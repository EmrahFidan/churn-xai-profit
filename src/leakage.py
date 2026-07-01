"""Leakage audit — REPORT ONLY, does not drop columns.

For each feature in each labeled dataset the univariate churn AUC is measured;
>= 0.90 is flagged as SUSPICIOUS. The drop decision is left to the user. Output:
outputs/tables/leakage_audit.csv.
"""
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from . import config as cfg
from . import strings as S

ESIK = 0.90  # suspicion threshold


def _yon_bagimsiz(a: float) -> float:
    """Direction-independent AUC: max(a, 1-a)."""
    return max(a, 1.0 - a)


def _sayisal_auc(x: pd.Series, y: np.ndarray) -> float:
    """Univariate AUC for a numeric feature (filled with the median)."""
    xx = pd.to_numeric(x, errors="coerce")
    xx = xx.fillna(xx.median())
    return _yon_bagimsiz(roc_auc_score(y, xx.to_numpy()))


def _kategorik_auc(x: pd.Series, y: np.ndarray, seed: int) -> float:
    """OOF AUC for a categorical feature with 5-fold CV target-mean encoding.

    To avoid in-sample inflation, the category churn rate is learned from each
    fold's training part and applied to the validation part (unseen category ->
    global rate).
    """
    xv = x.astype("string").fillna("__NA__").to_numpy()
    oof = np.full(len(y), np.nan)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, va in skf.split(xv, y):
        oran = pd.Series(y[tr]).groupby(pd.Series(xv[tr])).mean()
        glob = y[tr].mean()
        oof[va] = pd.Series(xv[va]).map(oran).fillna(glob).to_numpy()
    return _yon_bagimsiz(roc_auc_score(y, oof))


def denetle_set(df: pd.DataFrame, key: str, seed: int):
    """Returns a list of rows (dictionaries) for a single labeled dataset."""
    y = df["churn"].to_numpy()
    satirlar = []
    for c in df.columns:
        if c == "churn":
            continue
        if is_numeric_dtype(df[c]):
            auc = _sayisal_auc(df[c], y)
        else:
            auc = _kategorik_auc(df[c], y, seed)
        supheli = auc >= ESIK
        alan = S.GEREKCE_ALAN.get((key, c))
        if supheli:
            gerekce = S.GEREKCE_SUPHELI + (f" {alan}" if alan else "")
        else:
            gerekce = alan if alan else S.GEREKCE_NORMAL
        satirlar.append({
            S.KOLON["veri_seti"]: key,
            S.KOLON["ozellik"]: c,
            S.KOLON["tekil_auc"]: round(float(auc), 4),
            S.KOLON["bayrak"]: S.BAYRAK_SUPHELI if supheli else S.BAYRAK_NORMAL,
            S.KOLON["aksiyon"]: S.AKSIYON_INCELE if supheli else S.AKSIYON_TUT,
            S.KOLON["gerekce"]: gerekce,
        })
    return satirlar


def denetle_hepsi(processed: dict, seed: int = None):
    """Audits all labeled datasets, writes leakage_audit.csv, returns a DataFrame.

    cell2cell_test (unlabeled) is excluded.
    """
    seed = cfg.SEED if seed is None else seed
    cfg.klasorleri_hazirla()
    satirlar = []
    for key in cfg.DATASETS:  # the 5 labeled datasets
        satirlar += denetle_set(processed[key], key, seed)
    df = pd.DataFrame(satirlar).sort_values(
        [S.KOLON["veri_seti"], S.KOLON["tekil_auc"]], ascending=[True, False]
    ).reset_index(drop=True)
    df.to_csv(cfg.TABLES / "leakage_audit.csv", index=False)
    return df
