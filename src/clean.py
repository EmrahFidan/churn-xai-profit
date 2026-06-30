"""Set bazında temizlik (yalnızca güvenli işlemler: eksik + tip düzeltme).

ENCODING YOK. Kategorikler ham bırakılır. Her set kendi şeridinde temizlenir;
çıktı data/processed/<set>_clean.csv. cell2cell holdout'u train istatistikleriyle
doldurulur (tutarlılık + sızıntı önleme).
"""
import pandas as pd
from pandas.api.types import is_numeric_dtype

from . import config as cfg


def _fit_doldurma(df: pd.DataFrame, haric=("churn",)) -> dict:
    """Eksik içeren her kolon için doldurma istatistiğini öğrenir.

    Sayısal -> medyan, kategorik -> mod (yoksa 'Unknown'). Dönüş: kolon ->
    (yontem_adi, deger).
    """
    stats = {}
    for c in df.columns:
        if c in haric or not df[c].isna().any():
            continue
        if is_numeric_dtype(df[c]):
            stats[c] = ("medyan", df[c].median())
        else:
            mod = df[c].mode(dropna=True)
            stats[c] = ("mod", mod.iloc[0] if len(mod) else "Unknown")
    return stats


def _uygula_doldurma(df: pd.DataFrame, stats: dict):
    """Öğrenilen istatistiklerle eksikleri doldurur. Dönüş: (df, log_satirlari)."""
    df = df.copy()
    log = []
    for c, (yontem, deger) in stats.items():
        if c not in df.columns:
            continue
        n = int(df[c].isna().sum())
        if n:
            df[c] = df[c].fillna(deger)
            gosterim = f"{deger:.2f}" if isinstance(deger, float) else str(deger)
            log.append((c, f"eksik -> {yontem}", f"{n} hücre dolduruldu ({gosterim})"))
    return df, log


def temizle_telco(df: pd.DataFrame):
    """Telco: TotalCharges metne gömülü boşluk -> sayısal; tenure=0 -> 0, kalan medyan."""
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
    log.append(("TotalCharges", "metin -> sayısal + eksik doldurma",
                f"{n_bos} boş; tenure=0 -> 0 ({n_sifir} satır); kalan {n_med} -> medyan ({med:.2f})"))
    return df, log


def temizle_hepsi(etiketli: dict, holdout: dict):
    """Tüm setleri temizler, data/processed/ altına yazar.

    Dönüş: (processed: ad->df, log: (set,kolon,islem,detay) listesi).
    """
    cfg.klasorleri_hazirla()
    processed = {}
    log = []

    # Telco
    d, lg = temizle_telco(etiketli["telco"]["df"])
    processed["telco"] = d
    log += [("telco", *r) for r in lg]

    # Cell2Cell: istatistikleri TRAIN'den öğren, train + test'e uygula
    tr = etiketli["cell2cell"]["df"]
    stats = _fit_doldurma(tr)
    tr_c, lg = _uygula_doldurma(tr, stats)
    processed["cell2cell"] = tr_c
    log += [("cell2cell", *r) for r in lg]
    te_c, lg = _uygula_doldurma(holdout["df"], stats)
    processed["cell2cell_test"] = te_c
    log += [("cell2cell_test", *r) for r in lg]

    # E-commerce: sayısal eksikler -> medyan
    d = etiketli["ecommerce"]["df"]
    stats = _fit_doldurma(d)
    d_c, lg = _uygula_doldurma(d, stats)
    processed["ecommerce"] = d_c
    log += [("ecommerce", *r) for r in lg]

    # Bank ve Iranian: eksik yok -> doğrula, dokunma
    for k in ("bank", "iranian"):
        d = etiketli[k]["df"].copy()
        eksik = int(d.drop(columns=["churn"]).isna().sum().sum())
        assert eksik == 0, f"{k}: beklenmeyen eksik ({eksik})"
        processed[k] = d
        log.append((k, "-", "temiz", "eksik yok, dokunulmadı"))

    # Yaz
    for ad, df in processed.items():
        df.to_csv(cfg.PROCESSED / f"{ad}_clean.csv", index=False)

    return processed, log
