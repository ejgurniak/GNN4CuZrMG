#!/usr/bin/env python3
"""Train a random-forest classifier on Voronoi features.

The model is a scikit-learn ``RandomForestClassifier`` -- an ensemble of decision
trees, each grown on a bootstrap sample of the data with splits chosen from a
random subset of features; predictions are the (probability-)averaged vote of the
trees. Unlike the logistic-regression baseline this model is *not* trained with a
differentiable loss and a gradient optimizer: a forest is fit in a single
``.fit()`` call, so there are no epochs and no Adam/cross-entropy training loop.
The samples in the given folder are split (stratified) into a training and a
validation set; training/validation accuracy, log-loss (cross-entropy, reported
as an evaluation metric only) and per-feature importances are reported.

Example
-------
    python train.py ./fold1
"""
import argparse
import os

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss

from data_utils import load_dataset


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("data_dir", help="Folder with <number>.txt sample files (e.g. .../fold1)")
    p.add_argument("--model-path", default="model.joblib",
                   help="Where to save the trained model (default: model.joblib)")
    p.add_argument("--n-estimators", type=int, default=100,
                   help="Number of trees in the forest (default: 100)")
    p.add_argument("--max-depth", type=int, default=None,
                   help="Maximum tree depth (default: None = grow until leaves are pure)")
    p.add_argument("--min-samples-split", type=int, default=2,
                   help="Min samples required to split an internal node (default: 2)")
    p.add_argument("--min-samples-leaf", type=int, default=1,
                   help="Min samples required at a leaf node (default: 1)")
    p.add_argument("--max-features", default="sqrt",
                   help="Features considered per split: 'sqrt', 'log2', 'all', an int, "
                        "or a float fraction (default: sqrt)")
    p.add_argument("--criterion", default="gini",
                   choices=["gini", "entropy", "log_loss"],
                   help="Split-quality (impurity) criterion used to choose splits -- "
                        "this is NOT a trainable loss (default: gini)")
    p.add_argument("--val-split", type=float, default=0.2,
                   help="Fraction of data held out for validation (default: 0.2)")
    p.add_argument("--n-jobs", type=int, default=-1,
                   help="Parallel jobs for fitting/predicting (-1 = all cores)")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p.add_argument("--no-standardize", action="store_true",
                   help="Disable per-feature standardization. NOTE: decision trees are "
                        "invariant to it, so this never changes predictions; it is kept "
                        "only for parity with the logistic-regression checkpoint format.")
    return p.parse_args()


def stratified_split(y, val_split, seed):
    """Return ``(train_idx, val_idx)`` with each class's proportion preserved."""
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        # rng.shuffle(idx)
        n_val = int(round(len(idx) * val_split))
        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return np.array(train_idx), np.array(val_idx)


def parse_max_features(value):
    """Coerce the --max-features string into the type scikit-learn expects."""
    if value in ("sqrt", "log2"):
        return value
    if value.lower() in ("all", "none"):
        return None  # consider all features at every split
    try:
        return int(value)
    except ValueError:
        return float(value)


def evaluate(clf, X, y, n_classes):
    """Full-set accuracy and log-loss (cross-entropy) for the given arrays."""
    proba = clf.predict_proba(X)
    preds = clf.classes_[proba.argmax(axis=1)]
    acc = accuracy_score(y, preds)
    loss = log_loss(y, proba, labels=list(range(n_classes)))
    return loss, acc


def main():
    args = parse_args()
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
    # NOTE: decision trees split on per-feature thresholds and are invariant to
    # this affine rescaling, so it does not change predictions. It is applied only
    # so the checkpoint mirrors the logistic-regression one and predict.py stays
    # parallel.
    if args.no_standardize:
        mean = np.zeros(n_features, dtype=np.float32)
        std = np.ones(n_features, dtype=np.float32)
    else:
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)
        std[std == 0] = 1.0  # avoid divide-by-zero for constant features
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    # ---- model ----
    # A random forest has no optimizer and no trainable loss; it is fit in one
    # shot. ``criterion`` below is the split-impurity measure, not a training loss.
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        criterion=args.criterion,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        max_features=parse_max_features(args.max_features),
        n_jobs=args.n_jobs,
        random_state=args.seed,
    )

    # ---- fit ----
    model.fit(X_train, y_train)

    # ---- metrics ----
    train_loss, train_acc = evaluate(model, X_train, y_train, n_classes)
    val_loss, val_acc = evaluate(model, X_val, y_val, n_classes)
    print("-" * 64)
    print(f"Final | train_acc {train_acc:.4f} (log_loss {train_loss:.4f}) | "
          f"val_acc {val_acc:.4f} | val_log_loss {val_loss:.4f}")

    # ---- feature importances (mean impurity decrease, ranked) ----
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]
    print("Top features by importance:")
    for j in order[:min(10, n_features)]:
        print(f"  {feature_names[j]:<30} {importances[j]:.4f}")

    # ---- log file for plotting / inspection later ----
    with open("log.txt", "w") as log_file:
        log_file.write("# random-forest training summary\n")
        log_file.write(f"train_acc {train_acc}\n")
        log_file.write(f"train_log_loss {train_loss}\n")
        log_file.write(f"val_acc {val_acc}\n")
        log_file.write(f"val_log_loss {val_loss}\n")
        log_file.write("# rank feature importance\n")
        for rank, j in enumerate(order, start=1):
            log_file.write(f"{rank} {feature_names[j]} {importances[j]}\n")

    # ---- save checkpoint (everything predict.py needs to reproduce inference) ----
    checkpoint = {
        "model": model,
        "feature_names": feature_names,   # fixes the input column order
        "classes": classes,              # maps model index -> original label
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
        "n_features": n_features,
        "n_classes": n_classes,
        "standardize": not args.no_standardize,
    }
    joblib.dump(checkpoint, args.model_path)
    print(f"Saved model to {os.path.abspath(args.model_path)}")


if __name__ == "__main__":
    main()
