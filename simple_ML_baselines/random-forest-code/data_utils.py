"""Shared data-loading utilities for the Voronoi-feature random-forest scripts.

Data format (one file per sample, named ``<number>.txt``)::

    <class_label>                       # line 1: an integer class label
    <feature_label>: <value>            # remaining lines: a float feature value
    <feature_label>: <value>
    ...

Every sample in a dataset is expected to expose the same set of feature labels.
The feature *vocabulary* (a sorted list of labels) fixes the column order of the
design matrix, so the same ordering is reused at prediction time (it is stored in
the model checkpoint by ``train.py``).
"""
from __future__ import annotations

import glob
import os

import numpy as np


def parse_file(path):
    """Parse a single sample file.

    Returns ``(label, features)`` where ``label`` is an ``int`` (or ``None`` if the
    first line is not an integer, e.g. unlabeled data) and ``features`` is a dict
    mapping ``feature_label -> float value``.
    """
    with open(path) as fh:
        lines = [ln.strip() for ln in fh if ln.strip()]

    label = None
    features = {}
    if not lines:
        return label, features

    # Line 1 is the class label.
    try:
        label = int(lines[0])
    except ValueError:
        label = None

    # Remaining lines are "<feature_label>: <value>".
    for ln in lines[1:]:
        if ":" not in ln:
            continue
        key, value = ln.rsplit(":", 1)
        features[key.strip()] = float(value.strip())

    return label, features


def list_sample_files(data_dir):
    """Return the ``*.txt`` sample files in ``data_dir`` sorted numerically by stem."""
    files = glob.glob(os.path.join(data_dir, "*.txt"))

    def stem_key(p):
        stem = os.path.splitext(os.path.basename(p))[0]
        try:
            return (0, int(stem))  # numeric stems first, in numeric order
        except ValueError:
            return (1, stem)

    return sorted(files, key=stem_key)


def load_dataset(data_dir, feature_names=None):
    """Load every ``*.txt`` sample in ``data_dir`` into a design matrix.

    Parameters
    ----------
    data_dir : str
        Folder containing ``<number>.txt`` sample files.
    feature_names : list[str] or None
        If given, use exactly this column order (features missing from a sample
        become ``0.0``; features not in the list are ignored). This is how
        prediction reuses the training-time vocabulary. If ``None``, the
        vocabulary is inferred as the sorted union of labels across all files.

    Returns
    -------
    X : np.ndarray, shape (n_samples, n_features), float32
        The design matrix, row-aligned with ``files``.
    labels : list[int | None]
        Raw integer class labels (or ``None`` where a label was absent).
    files : list[str]
        Sample file paths, row-aligned with ``X``.
    feature_names : list[str]
        The feature column order actually used.
    """
    files = list_sample_files(data_dir)
    if not files:
        raise FileNotFoundError(f"No .txt sample files found in {data_dir!r}")

    parsed = [parse_file(p) for p in files]
    labels = [label for label, _ in parsed]
    feats = [feature for _, feature in parsed]

    if feature_names is None:
        vocab = set()
        for feature in feats:
            vocab.update(feature.keys())
        feature_names = sorted(vocab)

    col = {name: j for j, name in enumerate(feature_names)}
    X = np.zeros((len(files), len(feature_names)), dtype=np.float32)
    for i, feature in enumerate(feats):
        for name, value in feature.items():
            j = col.get(name)
            if j is not None:
                X[i, j] = value

    return X, labels, files, feature_names
