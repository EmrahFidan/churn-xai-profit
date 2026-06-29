# churn-xai-profit

Explainable and cost-sensitive customer churn prediction across several
independent public churn datasets. Tabular data, no GPU, gradient boosting
(XGBoost / LightGBM) with SHAP-based explanations and profit-aware evaluation.

Each dataset is processed independently in its own lane. **The datasets are
never merged** — every output (figures, tables, logs) is produced and stored
per dataset.

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Apple Silicon, LightGBM and XGBoost need OpenMP:

```bash
brew install libomp
```

## Layout

```
data/raw/          original CSVs (gitignored, place them here)
data/processed/    cleaned / derived data
src/               source modules (config, load, clean, eda, leakage)
notebooks/         exploratory notebooks
outputs/           figures, tables, logs (gitignored, per dataset)
tests/             tests
config.yaml        random seed + per-dataset registry (file, sector, target)
```

## Datasets

Drop the original CSVs into `data/raw/` using the file names declared in
`config.yaml`. Each dataset has its own target column (see `config.yaml`).
