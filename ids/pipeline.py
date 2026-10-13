"""The 9-stage training pipeline.

    1. Load          read the raw NSL-KDD train and test files
    2. Clean         drop duplicate rows and check for missing values
    3. Label         map ~40 raw attack names to 4 alert categories
    4. Prune         drop features that never change in the training data
    5. Encode        one-hot encode protocol, service and TCP flag
    6. Select        keep the features that carry 99% of the importance
    7. Train         fit a class-balanced Random Forest
    8. Evaluate      score on the official test set (includes unseen attacks)
    9. Save          write model + preprocessing + metrics to one file
"""

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from ids import data

RANDOM_STATE = 42


class Preprocessor:
    """Stages 4-6, fitted on training data and replayed on any new connection."""

    def fit(self, raw):
        features = raw[data.FEATURE_COLUMNS]
        self.dropped = [c for c in features.columns if features[c].nunique() <= 1]
        self.encoded_columns = list(self._encode(raw).columns)
        self.selected = self.encoded_columns
        return self

    def transform(self, raw):
        encoded = self._encode(raw)
        # A category never seen in training (e.g. a new service) becomes all
        # zeros; a category seen in training but missing here becomes a zero column.
        encoded = encoded.reindex(columns=self.encoded_columns, fill_value=0)
        return encoded[self.selected].astype(float)

    def _encode(self, raw):
        features = raw[data.FEATURE_COLUMNS].drop(columns=self.dropped)
        return pd.get_dummies(features, columns=data.CATEGORICAL_COLUMNS, dtype=int)


def select_features(X, y, coverage=0.99):
    """Rank features with a quick forest and keep the smallest set covering `coverage`."""
    quick = RandomForestClassifier(
        n_estimators=50, class_weight="balanced_subsample",
        random_state=RANDOM_STATE, n_jobs=-1,
    ).fit(X, y)
    ranked = pd.Series(quick.feature_importances_, index=X.columns).sort_values(ascending=False)
    keep = (ranked.cumsum() <= coverage) | (np.arange(len(ranked)) == 0)
    return list(ranked[keep].index)


def run(data_dir="data", model_path="models/ids_model.joblib", n_estimators=200):
    def stage(n, text):
        print(f"[{n}/9] {text}")

    stage(1, "Loading NSL-KDD")
    data.download(data_dir)
    train_raw, test_raw = data.load("train", data_dir), data.load("test", data_dir)
    print(f"      train={len(train_raw):,} rows, test={len(test_raw):,} rows")

    stage(2, "Cleaning")
    before = len(train_raw)
    train_raw = train_raw.drop_duplicates(subset=data.FEATURE_COLUMNS + ["attack"])
    missing = int(train_raw.isna().sum().sum() + test_raw.isna().sum().sum())
    if missing:
        raise ValueError(f"{missing} missing values found; the raw files may be corrupt")
    print(f"      removed {before - len(train_raw):,} duplicate rows, 0 missing values")

    stage(3, "Mapping attack names to categories")
    y_train = data.to_category(train_raw["attack"])
    y_test = data.to_category(test_raw["attack"])
    print("      train:", y_train.value_counts().to_dict())

    stage(4, "Pruning constant features")
    prep = Preprocessor().fit(train_raw)
    print(f"      dropped {prep.dropped}")

    stage(5, "One-hot encoding protocol / service / flag")
    X_train = prep.transform(train_raw)
    print(f"      {len(data.FEATURE_COLUMNS)} raw features -> {X_train.shape[1]} encoded features")

    stage(6, "Selecting features")
    prep.selected = select_features(X_train, y_train)
    X_train, X_test = prep.transform(train_raw), prep.transform(test_raw)
    print(f"      kept {len(prep.selected)} features covering 99% of importance")

    stage(7, f"Training Random Forest ({n_estimators} trees)")
    start = time.time()
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced_subsample",  # Brute Force is <1% of training rows
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ).fit(X_train, y_train)
    print(f"      trained in {time.time() - start:.1f}s")

    stage(8, "Evaluating on the official test set")
    predicted = model.predict(X_test)
    labels = data.CATEGORIES
    metrics = {
        "accuracy": accuracy_score(y_test, predicted),
        "report": classification_report(y_test, predicted, labels=labels, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, predicted, labels=labels).tolist(),
        "labels": labels,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "unseen_attacks_in_test": sorted(set(test_raw["attack"]) - set(train_raw["attack"])),
    }
    print(classification_report(y_test, predicted, labels=labels, digits=3, zero_division=0))

    stage(9, f"Saving to {model_path}")
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "preprocessor": prep, "metrics": metrics}, model_path, compress=3)
    model_path.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2))
    return model, prep, metrics
