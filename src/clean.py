"""Per-dataset cleaning (safe operations only: missing values + type fixes).

NO ENCODING. Categoricals are left raw. Each dataset is cleaned in its own lane;
output data/processed/<set>_clean.csv. The cell2cell holdout is filled with train
statistics (consistency + leakage prevention).
"""
import pandas as pd
from pandas.api.types import is_numeric_dtype

from . import config as cfg


def _fit_doldurma(df: pd.DataFrame, haric=("churn",)) -> dict:
    """Learns the fill statistic for each column that contains missing values.

    Numeric -> median, categorical -> mode (if none, 'Unknown'). Returns: column ->
    (method_name, value).
    """
    stats = {}
    for c in df.columns:
        if c in haric or not df[c].isna().any():
            continue
        if is_numeric_dtype(df[c]):
            stats[c] = ("median", df[c].median())
        else:
            mod = df[c].mode(dropna=True)
            stats[c] = ("mode", mod.iloc[0] if len(mod) else "Unknown")
    return stats


def _uygula_doldurma(df: pd.DataFrame, stats: dict):
    """Fills missing values with the learned statistics. Returns: (df, log_rows)."""
    df = df.copy()
    log = []
    for c, (yontem, deger) in stats.items():
        if c not in df.columns:
            continue
        n = int(df[c].isna().sum())
        if n:
            df[c] = df[c].fillna(deger)
            gosterim = f"{deger:.2f}" if isinstance(deger, float) else str(deger)
            log.append((c, f"missing -> {yontem}", f"{n} cells filled ({gosterim})"))
    return df, log


def temizle_telco(df: pd.DataFrame):
    """Telco: TotalCharges whitespace embedded in text -> numeric; tenure=0 -> 0, rest median."""
    df = df.copy()
    log = []
    tc = pd.to_numeric(df["TotalCharges"], errors="coerce")
    n_bos = int(tc.isna().sum())
    sifir_mask = (df["tenure"] == 0) & tc.isna()
    n_sifir = int(sifir_mask.sum())
    tc[sifir_mask] = 0.0
    med = tc.median()
    n_med = int(tc.isna().sum())
    tc = tc.fillna(med)
    df["TotalCharges"] = tc
    log.append(("TotalCharges", "text -> numeric + missing fill",
                f"{n_bos} blank; tenure=0 -> 0 ({n_sifir} rows); remaining {n_med} -> median ({med:.2f})"))
    return df, log


def temizle_hepsi(etiketli: dict, holdout: dict):
    """Cleans all datasets, writes them under data/processed/.

    Returns: (processed: name->df, log: list of (set, column, operation, detail)).
    """
    cfg.klasorleri_hazirla()
    processed = {}
    log = []

    # Telco
    d, lg = temizle_telco(etiketli["telco"]["df"])
    processed["telco"] = d
    log += [("telco", *r) for r in lg]

    # Cell2Cell: learn statistics from TRAIN, apply to train + test
    tr = etiketli["cell2cell"]["df"]
    stats = _fit_doldurma(tr)
    tr_c, lg = _uygula_doldurma(tr, stats)
    processed["cell2cell"] = tr_c
    log += [("cell2cell", *r) for r in lg]
    te_c, lg = _uygula_doldurma(holdout["df"], stats)
    processed["cell2cell_test"] = te_c
    log += [("cell2cell_test", *r) for r in lg]

    # E-commerce: numeric missing values -> median
    d = etiketli["ecommerce"]["df"]
    stats = _fit_doldurma(d)
    d_c, lg = _uygula_doldurma(d, stats)
    processed["ecommerce"] = d_c
    log += [("ecommerce", *r) for r in lg]

    # Bank and Iranian: no missing values -> validate, do not touch
    for k in ("bank", "iranian"):
        d = etiketli[k]["df"].copy()
        eksik = int(d.drop(columns=["churn"]).isna().sum().sum())
        assert eksik == 0, f"{k}: unexpected missing ({eksik})"
        processed[k] = d
        log.append((k, "-", "clean", "no missing, untouched"))

    # Write
    for ad, df in processed.items():
        df.to_csv(cfg.PROCESSED / f"{ad}_clean.csv", index=False)

    return processed, log
