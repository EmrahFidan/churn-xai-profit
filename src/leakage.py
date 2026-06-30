"""Leakage (sızıntı) denetimi — SADECE RAPOR, kolon düşürmez.

Her etiketli sette her feature için tek-değişkenli churn AUC'si ölçülür;
>= 0.90 ŞÜPHELİ işaretlenir. Düşürme kararı kullanıcıya bırakılır. Çıktı:
outputs/tables/leakage_audit.csv.
"""
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from . import config as cfg
from . import strings_tr as S

ESIK = 0.90  # şüphe eşiği


def _yon_bagimsiz(a: float) -> float:
    """AUC yön bağımsız: max(a, 1-a)."""
    return max(a, 1.0 - a)


def _sayisal_auc(x: pd.Series, y: np.ndarray) -> float:
    """Sayısal feature için tek-değişkenli AUC (medyan ile doldurulmuş)."""
    xx = pd.to_numeric(x, errors="coerce")
    xx = xx.fillna(xx.median())
    return _yon_bagimsiz(roc_auc_score(y, xx.to_numpy()))


def _kategorik_auc(x: pd.Series, y: np.ndarray, seed: int) -> float:
    """Kategorik feature için 5-kat CV hedef-ortalama kodlamasıyla OOF AUC.

    In-sample şişmeyi önlemek için her kat eğitim kısmından kategori churn oranı
    öğrenilir, doğrulama kısmına uygulanır (görülmeyen kategori -> global oran).
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
    """Tek bir etiketli set için satır listesi döndürür (sözlükler)."""
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
    """Etiketli setlerin tümünü denetler, leakage_audit.csv yazar, DataFrame döndürür.

    cell2cell_test (etiketsiz) hariç tutulur.
    """
    seed = cfg.SEED if seed is None else seed
    cfg.klasorleri_hazirla()
    satirlar = []
    for key in cfg.DATASETS:  # etiketli 5 set
        satirlar += denetle_set(processed[key], key, seed)
    df = pd.DataFrame(satirlar).sort_values(
        [S.KOLON["veri_seti"], S.KOLON["tekil_auc"]], ascending=[True, False]
    ).reset_index(drop=True)
    df.to_csv(cfg.TABLES / "leakage_audit.csv", index=False)
    return df
