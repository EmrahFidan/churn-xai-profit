"""Dataset loading and standardization.

Loads each dataset from data/raw/ (the unlabeled holdout from data/holdout/),
normalizes the target to a single name ('churn', 0/1), and drops identifier
columns (logged). One dataset at a time; datasets are never merged. NO ENCODING
is done in this step — categoricals are left raw.
"""
import pandas as pd
from pandas.api.types import is_numeric_dtype

from . import config as cfg

# Identifier columns to drop, per dataset
ID_KOLON = {
    "telco": ["customerID"],
    "cell2cell": ["CustomerID"],
    "bank": ["RowNumber", "CustomerId", "Surname"],
    "ecommerce": [],
    "iranian": [],
}
_POZITIF = {"yes", "true", "1", "churn", "churned"}
_NEGATIF = {"no", "false", "0"}


def _ikili_churn(s: pd.Series) -> pd.Series:
    """Converts the target series to 0/1. Values that cannot be converted (unlabeled) remain NaN.

    Accepts numeric 0/1 or textual Yes/No / True/False. For the unlabeled holdout
    all values are empty and are preserved as NaN (nullable Int64).
    """
    if is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").astype("Int64")
    m = s.astype("string").str.strip().str.lower()
    out = m.map(lambda v: 1 if v in _POZITIF else (0 if v in _NEGATIF else pd.NA))
    return out.astype("Int64")


def standardize(df: pd.DataFrame, key: str, hedef_kolon: str):
    """Drops identifier columns and converts the target to 'churn'.

    Returns: (new_df, dropped_identifier_columns). The 'churn' column is appended
    at the end; feature column names are left untouched.
    """
    df = df.copy()
    dusurulen = [c for c in ID_KOLON.get(key, []) if c in df.columns]
    if dusurulen:
        df = df.drop(columns=dusurulen)
    churn = _ikili_churn(df[hedef_kolon])
    df = df.drop(columns=[hedef_kolon])
    df["churn"] = churn
    return df, dusurulen


def yukle_etiketli():
    """Loads and standardizes the 5 labeled datasets.

    Returns: key -> {df, sector, dropped, raw_shape, raw}. 'df' is standardized
    (churn int), 'raw' is the raw loaded copy (for the missingness figure/overview).
    """
    out = {}
    for key, d in cfg.DATASETS.items():
        raw = pd.read_csv(cfg.RAW / d["file"], low_memory=False)
        std, dusurulen = standardize(raw, key, d["target"])
        assert std["churn"].notna().all(), f"{key}: empty churn in labeled dataset"
        std["churn"] = std["churn"].astype(int)
        out[key] = {
            "df": std,
            "sector": d["sector"],
            "dropped": dusurulen,
            "raw_shape": raw.shape,
            "raw": raw,
        }
    return out


def yukle_holdout():
    """Loads the unlabeled cell2cell test holdout (same schema as cell2cell).

    The churn column is entirely NaN; it is not included in the churn-based parts
    of EDA/leakage, it is only cleaned and stored separately.
    """
    raw = pd.read_csv(cfg.HOLDOUT / cfg.HOLDOUT_DOSYA, low_memory=False)
    std, dusurulen = standardize(raw, "cell2cell", cfg.DATASETS["cell2cell"]["target"])
    return {"df": std, "dropped": dusurulen, "raw_shape": raw.shape, "raw": raw}
