# Data

Original CSVs are not tracked in git. Place each dataset's CSV into
`data/raw/` using the file name declared in `../config.yaml`.

| key       | file           | sector   | target |
|-----------|----------------|----------|--------|
| telco     | telco.csv      | telecom  | Churn  |
| cell2cell | cell2cell.csv  | telecom  | Churn  |
| ecommerce | ecommerce.csv  | ecommerce| Churn  |
| iranian   | iranian.csv    | telecom  | Churn  |
| bank      | bank.csv       | banking  | Exited |

Sources (public datasets):
- Telco Customer Churn — IBM sample dataset (Kaggle)
- Cell2Cell — Teradata Center / Kaggle
- E-commerce Churn — Kaggle
- Iranian Churn — UCI Machine Learning Repository
- Bank Customer Churn — Kaggle

`data/processed/` holds cleaned and derived data produced by the pipeline.
Each dataset stays in its own lane; datasets are never merged.
