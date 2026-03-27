from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.config import MODELS_DIR

st.set_page_config(page_title="Credit Risk Scoring", page_icon="??", layout="centered")

st.title("Credit Risk Scoring")
st.write("Predict loan default risk and explain the decision.")

model_path = MODELS_DIR / "best_model.joblib"
columns_path = MODELS_DIR / "feature_columns.json"

if not model_path.exists() or not columns_path.exists():
    st.error("Model not found. Run `python -m src.train` first.")
    st.stop()

model = joblib.load(model_path)
columns = json.loads(columns_path.read_text())

st.subheader("Applicant Details")

sample_defaults = {
    "checking_status": "<0",
    "duration": 24,
    "credit_history": "existing paid",
    "purpose": "radio/tv",
    "credit_amount": 5000,
    "savings_status": "<100",
    "employment": "1<=X<4",
    "installment_commitment": 3,
    "personal_status": "male single",
    "other_parties": "none",
    "residence_since": 2,
    "property_magnitude": "car",
    "age": 35,
    "other_payment_plans": "none",
    "housing": "own",
    "existing_credits": 1,
    "job": "skilled",
    "num_dependents": 1,
    "own_telephone": "yes",
    "foreign_worker": "yes",
}

use_sample = st.checkbox("Autofill with sample applicant", value=False)

inputs = {}
for col in columns:
    if col in ["duration", "credit_amount", "age", "installment_commitment", "residence_since", "existing_credits", "num_dependents"]:
        default_val = sample_defaults.get(col, 1) if use_sample else 1
        inputs[col] = st.number_input(col, min_value=0, value=int(default_val))
    else:
        default_val = sample_defaults.get(col, "") if use_sample else ""
        inputs[col] = st.text_input(col, value=str(default_val))

if st.button("Score Applicant"):
    row = {c: inputs.get(c, None) for c in columns}
    X = pd.DataFrame([row], columns=columns)

    proba = model.predict_proba(X)
    classes = list(model.named_steps["model"].classes_)
    pos_index = classes.index("bad")
    risk = float(proba[:, pos_index][0])
    label = "HIGH_RISK" if risk >= 0.5 else "LOW_RISK"

    st.metric("Default Risk Probability", f"{risk:.4f}")
    st.success(f"Risk Label: {label}")

st.subheader("How to run")
st.code("streamlit run app.py", language="bash")
