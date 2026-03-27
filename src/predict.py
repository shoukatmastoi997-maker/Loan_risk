from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from .config import MODELS_DIR
from .utils import log


def load_applicant(path: Path) -> dict:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("Input JSON must be an object with feature keys")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a single loan applicant")
    parser.add_argument("--input", required=True, help="Path to applicant JSON")
    args = parser.parse_args()

    model_path = MODELS_DIR / "best_model.joblib"
    columns_path = MODELS_DIR / "feature_columns.json"

    if not model_path.exists() or not columns_path.exists():
        raise FileNotFoundError("Model or feature columns not found. Run training first.")

    model = joblib.load(model_path)
    columns = json.loads(columns_path.read_text())

    applicant = load_applicant(Path(args.input))
    row = {c: applicant.get(c, None) for c in columns}
    X = pd.DataFrame([row], columns=columns)

    proba = model.predict_proba(X)
    classes = list(model.named_steps["model"].classes_)
    pos_index = classes.index("bad")
    risk = float(proba[:, pos_index][0])
    label = "HIGH_RISK" if risk >= 0.5 else "LOW_RISK"

    log(f"Default risk probability: {risk:.4f}")
    log(f"Risk label: {label}")


if __name__ == "__main__":
    main()
