from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import roc_auc_score

from .config import MODELS_DIR, RANDOM_STATE, TARGET_COL
from .data import load_data, get_feature_columns
from .utils import log

MODELS_DIR.mkdir(parents=True, exist_ok=True)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ]
    )

    return preprocessor


def main() -> None:
    X, y, df = load_data()
    feature_cols = get_feature_columns(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    preprocessor = build_preprocessor(X)

    models = {
        "logreg": LogisticRegression(max_iter=1000),
        "rf": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
        "gb": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }

    best_name = None
    best_auc = -np.inf
    best_model = None

    for name, model in models.items():
        pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
        scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="roc_auc")
        mean_auc = float(scores.mean())
        log(f"Model {name}: CV AUC = {mean_auc:.4f}")
        if mean_auc > best_auc:
            best_auc = mean_auc
            best_name = name
            best_model = pipe

    if best_model is None:
        raise RuntimeError("No model was trained")

    log(f"Best model: {best_name} (CV AUC {best_auc:.4f})")
    best_model.fit(X_train, y_train)

    proba = best_model.predict_proba(X_test)
    classes = list(best_model.named_steps["model"].classes_)
    pos_index = classes.index("bad")
    y_score = proba[:, pos_index]
    test_auc = roc_auc_score((y_test == "bad").astype(int), y_score)
    log(f"Test AUC = {test_auc:.4f}")

    model_path = MODELS_DIR / "best_model.joblib"
    joblib.dump(best_model, model_path)
    log(f"Saved model to {model_path}")

    meta = {
        "best_model": best_name,
        "cv_auc": best_auc,
        "test_auc": float(test_auc),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "target_col": TARGET_COL,
    }
    meta_path = MODELS_DIR / "model_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    columns_path = MODELS_DIR / "feature_columns.json"
    columns_path.write_text(json.dumps(feature_cols, indent=2))
    log(f"Saved metadata to {meta_path}")


if __name__ == "__main__":
    main()
