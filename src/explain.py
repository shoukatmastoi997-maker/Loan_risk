from __future__ import annotations

import csv
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from .config import MODELS_DIR, REPORTS_DIR, RANDOM_STATE
from .data import load_data
from .utils import log

REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def _get_feature_names(preprocessor, numeric_cols, categorical_cols) -> list[str]:
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_names = cat_encoder.get_feature_names_out(categorical_cols)
    return list(numeric_cols) + list(cat_names)


def main() -> None:
    X, y, _ = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    model_path = MODELS_DIR / "best_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError("Model not found. Run training first.")

    model = joblib.load(model_path)
    preprocessor = model.named_steps["preprocess"]

    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    feature_names = _get_feature_names(preprocessor, numeric_cols, categorical_cols)

    X_test_transformed = preprocessor.transform(X_test)
    if hasattr(X_test_transformed, "toarray"):
        X_test_transformed = X_test_transformed.toarray()
    perm = permutation_importance(
        model.named_steps["model"],
        X_test_transformed,
        (y_test == "bad").astype(int),
        n_repeats=10,
        random_state=RANDOM_STATE,
        scoring="roc_auc",
    )

    importances = perm.importances_mean
    indices = np.argsort(importances)[::-1]

    top_rows = []
    for i in indices[:20]:
        top_rows.append((feature_names[i], float(importances[i])))

    csv_path = REPORTS_DIR / "feature_importance.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["feature", "importance"])
        writer.writerows(top_rows)

    md_lines = ["# Feature Importance", "", "Top 20 features by permutation importance:", ""]
    for name, score in top_rows:
        md_lines.append(f"- {name}: {score:.6f}")

    md_path = REPORTS_DIR / "feature_importance.md"
    md_path.write_text("\n".join(md_lines))
    log(f"Saved feature importance to {md_path}")

    # Optional SHAP summary plot
    try:
        import shap
        import matplotlib.pyplot as plt

        sample = X_test.sample(n=min(200, len(X_test)), random_state=RANDOM_STATE)
        sample_transformed = preprocessor.transform(sample)
        if hasattr(sample_transformed, "toarray"):
            sample_transformed = sample_transformed.toarray()

        model_core = model.named_steps["model"]
        explainer = shap.Explainer(model_core, sample_transformed)
        shap_values = explainer(sample_transformed)

        shap.summary_plot(shap_values, feature_names=feature_names, show=False)
        shap_path = REPORTS_DIR / "shap_summary.png"
        plt.tight_layout()
        plt.savefig(shap_path, dpi=150)
        plt.close()
        log(f"Saved SHAP summary to {shap_path}")
    except Exception as e:
        log(f"SHAP summary skipped: {e}")


if __name__ == "__main__":
    main()
