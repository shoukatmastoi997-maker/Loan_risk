from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

from .config import MODELS_DIR, REPORTS_DIR, RANDOM_STATE
from .data import load_data
from .utils import log

REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    X, y, _ = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    model_path = MODELS_DIR / "best_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError("Model not found. Run training first.")

    model = joblib.load(model_path)
    proba = model.predict_proba(X_test)
    classes = list(model.named_steps["model"].classes_)
    pos_index = classes.index("bad")
    y_score = proba[:, pos_index]

    y_true = (y_test == "bad").astype(int)
    y_pred = (y_score >= 0.5).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "n_test": int(len(y_true)),
    }

    metrics_path = REPORTS_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    log(f"Saved metrics to {metrics_path}")

    md = [
        "# Model Metrics",
        "",
        f"- ROC AUC: {metrics['roc_auc']:.4f}",
        f"- F1: {metrics['f1']:.4f}",
        f"- Precision: {metrics['precision']:.4f}",
        f"- Recall: {metrics['recall']:.4f}",
        f"- Test size: {metrics['n_test']}",
        "",
        "Confusion matrix:",
        f"{metrics['confusion_matrix']}",
    ]
    metrics_md_path = REPORTS_DIR / "metrics.md"
    metrics_md_path.write_text("\n".join(md))
    log(f"Saved metrics summary to {metrics_md_path}")


if __name__ == "__main__":
    main()
