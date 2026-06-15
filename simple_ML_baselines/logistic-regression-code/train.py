#!/usr/bin/env python3
"""Train a multinomial (multi-class) logistic-regression classifier on Voronoi features.

The model is a single linear layer; combined with the softmax implied by
``nn.CrossEntropyLoss`` this is multinomial logistic regression. It is trained
with the Adam optimizer and multi-class cross-entropy loss. The samples in the
given folder are split into a training and a validation set; per-epoch training
loss, validation loss and validation accuracy are reported.

Example
-------
    python train.py ~/Desktop/research/analysis/CEGAN/Cu64Zr36_MG_samples/\\
        Voronoi_analysis/samples_Voronoi_features/7by7by7/200each/fold1
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from data_utils import load_dataset


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("data_dir", help="Folder with <number>.txt sample files (e.g. .../fold1)")
    p.add_argument("--model-path", default="model.pt",
                   help="Where to save the trained model (default: model.pt)")
    p.add_argument("--epochs", type=int, default=200, help="Training epochs (default: 200)")
    p.add_argument("--lr", type=float, default=1e-2, help="Adam learning rate (default: 0.01)")
    p.add_argument("--batch-size", type=int, default=64, help="Mini-batch size (default: 64)")
    p.add_argument("--val-split", type=float, default=0.2,
                   help="Fraction of data held out for validation (default: 0.2)")
    p.add_argument("--weight-decay", type=float, default=0.0,
                   help="L2 regularization strength (default: 0.0)")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p.add_argument("--no-standardize", action="store_true",
                   help="Disable per-feature standardization (mean 0 / std 1)")
    p.add_argument("--log-every", type=int, default=1,
                   help="Print metrics every N epochs (default: 10)")
    return p.parse_args()


def stratified_split(y, val_split, seed):
    """Return ``(train_idx, val_idx)`` with each class's proportion preserved.

    Validation is the *last* ``val_split`` fraction of each class's files (in
    the numerically-sorted file order); the earlier files are used for training.
    """
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        # rng.shuffle(idx)
        n_val = int(round(len(idx) * val_split))
        n_train = len(idx) - n_val
        train_idx.extend(idx[:n_train].tolist())  # first part -> train
        val_idx.extend(idx[n_train:].tolist())    # last n_val -> val
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return np.array(train_idx), np.array(val_idx)


@torch.no_grad()
def evaluate(model, criterion, X, y):
    """Full-set cross-entropy loss and accuracy for the given tensors."""
    model.eval()
    logits = model(X)
    loss = criterion(logits, y).item()
    acc = (logits.argmax(dim=1) == y).float().mean().item()
    return loss, acc


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- load data ----
    X, labels, files, feature_names = load_dataset(args.data_dir)
    if any(label is None for label in labels):
        missing = [f for f, label in zip(files, labels) if label is None]
        raise SystemExit(
            f"{len(missing)} file(s) lack an integer class label on line 1, "
            f"e.g. {os.path.basename(missing[0])}"
        )
    classes = sorted(set(labels))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y = np.array([class_to_idx[label] for label in labels], dtype=np.int64)
    n_features, n_classes = X.shape[1], len(classes)
    print(f"Loaded {len(files)} samples | {n_features} features | "
          f"{n_classes} classes {classes}")

    # ---- train / validation split (stratified) ----
    train_idx, val_idx = stratified_split(y, args.val_split, args.seed)
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    print(f"Training samples: {len(train_idx)} | Validation samples: {len(val_idx)}")

    # ---- standardize features (statistics from the training split only) ----
    if args.no_standardize:
        mean = np.zeros(n_features, dtype=np.float32)
        std = np.ones(n_features, dtype=np.float32)
    else:
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)
        std[std == 0] = 1.0  # avoid divide-by-zero for constant features
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    # ---- tensors / loader ----
    X_train_t = torch.from_numpy(X_train.astype(np.float32))
    y_train_t = torch.from_numpy(y_train)
    X_val_t = torch.from_numpy(X_val.astype(np.float32))
    y_val_t = torch.from_numpy(y_val)
    loader = DataLoader(
        TensorDataset(X_train_t, y_train_t), batch_size=args.batch_size, shuffle=True
    )

    # ---- model / loss / optimizer ----
    model = nn.Linear(n_features, n_classes)            # multinomial logistic regression
    criterion = nn.CrossEntropyLoss()                   # multi-class cross-entropy (softmax + NLL)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # ---- training loop ----
    best_val_acc = -1.0
    log_file = open('log.txt', 'w')
    log_file.write("epoch train_loss val_loss val_acc\n")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, seen = 0.0, 0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
            seen += xb.size(0)
        train_loss = running_loss / seen
        val_loss, val_acc = evaluate(model, criterion, X_val_t, y_val_t)
        best_val_acc = max(best_val_acc, val_acc)

        if epoch == 1 or epoch == args.epochs or epoch % args.log_every == 0:
            log_file.write(f"{epoch} {train_loss} {val_loss} {val_acc}\n")
            print(f"epoch {epoch:4d}/{args.epochs} | train_loss {train_loss:.4f} | "
                  f"val_loss {val_loss:.4f} | val_acc {val_acc:.4f}")
                  #TODO: also write to a log file for plotting later

    # ---- final metrics ----
    train_loss_f, train_acc_f = evaluate(model, criterion, X_train_t, y_train_t)
    val_loss_f, val_acc_f = evaluate(model, criterion, X_val_t, y_val_t)
    log_file.close()
    print("-" * 64)
    print(f"Final | train_loss {train_loss_f:.4f} (acc {train_acc_f:.4f}) | "
          f"val_loss {val_loss_f:.4f} | val_acc {val_acc_f:.4f} "
          f"| best_val_acc {best_val_acc:.4f}")

    # ---- save checkpoint (everything predict.py needs to reproduce inference) ----
    checkpoint = {
        "state_dict": model.state_dict(),
        "feature_names": feature_names,   # fixes the input column order
        "classes": classes,              # maps model index -> original label
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
        "n_features": n_features,
        "n_classes": n_classes,
        "standardize": not args.no_standardize,
    }
    torch.save(checkpoint, args.model_path)
    print(f"Saved model to {os.path.abspath(args.model_path)}")


if __name__ == "__main__":
    main()
