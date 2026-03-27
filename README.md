# Credit Risk Scoring with Explainable ML

A fast, job-ready finance ML project you can finish in one week. It predicts loan default risk and explains the predictions based on existing.

## What this project shows
- End-to-end ML pipeline
- Real finance dataset (German Credit / credit-g from OpenML)
- Business-focused metrics (AUC, F1, precision/recall)
- Explainability (feature importance + optional SHAP)
- A simple CLI scorer

## Quick start

1) Create and activate a virtual environment
2) Install requirements
3) Train the model
4) Evaluate and generate reports
5) Run the CLI scorer

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python -m src.train
python -m src.evaluate
python -m src.explain

python -m src.predict --input data\sample_applicant.json
```

## Outputs
- `models\best_model.joblib`
- `models\feature_columns.json`
- `reports\metrics.json`
- `reports\metrics.md`
- `reports\feature_importance.csv`
- `reports\feature_importance.md`
- `reports\shap_summary.png` (if SHAP is installed)

## Dataset
We use the OpenML `credit-g` dataset. The script downloads it automatically on first run and caches a CSV in `data\raw\credit_g.csv`.

## Notes
- This project is designed to be readable and resume-friendly.
- You can compare models and choose the best by AUC.
- The CLI scorer predicts default risk for a single applicant.
