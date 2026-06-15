#!/usr/bin/env python3
"""Predict classes for Voronoi-feature samples with a trained random-forest model.

Loads every ``<number>.txt`` sample in the given folder, applies the model saved
by ``train.py``, and prints one prediction per sample (file, predicted label,
confidence). If the samples carry true class labels on line 1, the overall
log-loss (cross-entropy) and accuracy are also reported.

Example
-------
    python predict.py ~/Desktop/research/analysis/CEGAN/Cu64Zr36_MG_samples/\\
        Voronoi_analysis/samples_Voronoi_features/7by7by7/200each/fold2 \\
        --model-path model.joblib --output predictions.csv
"""
import argparse
import os

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, log_loss

from data_utils import load_dataset


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("data_dir", help="Folder with <number>.txt sample files to predict")
    p.add_argument("--model-path", default="model.joblib",
                   help="Path to the trained model checkpoint (default: model.joblib)")
    p.add_argument("--output", default=None,
                   help="Optional CSV file to write predictions to (default: stdout)")
    return p.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.model_path):
        raise SystemExit(f"Model file not found: {args.model_path} "
                         f"(train one first with train.py)")

    checkpoint = joblib.load(args.model_path)
    model = checkpoint["model"]
    feature_names = checkpoint["feature_names"]
    classes = checkpoint["classes"]
    mean = np.asarray(checkpoint["mean"], dtype=np.float32)
    std = np.asarray(checkpoint["std"], dtype=np.float32)

    # Load using the training feature vocabulary so the columns line up exactly.
    X, labels, files, _ = load_dataset(args.data_dir, feature_names=feature_names)
    X = (X - mean) / std

    proba = model.predict_proba(X)
    pred_idx = model.classes_[proba.argmax(axis=1)]
    pred_labels = [classes[i] for i in pred_idx]
    confidences = proba.max(axis=1)

    # ---- write / print per-sample predictions ----
    header = "file,predicted_label,confidence"
    rows = [
        f"{os.path.basename(path)},{label},{conf:.6f}"
        for path, label, conf in zip(files, pred_labels, confidences)
    ]
    if args.output:
        with open(args.output, "w") as fh:
            fh.write(header + "\n")
            fh.write("\n".join(rows) + "\n")
        print(f"Wrote {len(rows)} predictions to {os.path.abspath(args.output)}")
    else:
        print(header)
        print("\n".join(rows))

    # ---- report loss / accuracy when ground-truth labels are present ----
    if labels and all(label is not None for label in labels):
        class_to_idx = {c: i for i, c in enumerate(classes)}
        unknown = sorted({label for label in labels if label not in class_to_idx})
        if unknown:
            print(f"\nNote: {len(unknown)} label(s) not seen during training "
                  f"{unknown}; excluded from metrics.")
        keep = [i for i, label in enumerate(labels) if label in class_to_idx]
        if keep:
            y_true = np.array([class_to_idx[labels[i]] for i in keep], dtype=np.int64)
            sub_proba = proba[keep]
            sub_preds = model.classes_[sub_proba.argmax(axis=1)]
            loss = log_loss(y_true, sub_proba, labels=list(range(len(classes))))
            acc = accuracy_score(y_true, sub_preds)
            print(f"\nEvaluated on {len(keep)} labeled samples | "
                  f"cross-entropy loss {loss:.4f} | accuracy {acc:.4f}")


if __name__ == "__main__":
    main()
