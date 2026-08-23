"""Per-dataset cleaning (deterministic operations only: type fixes and coding rules).

NO ENCODING and NO STATISTICAL IMPUTATION. Categoricals are left raw and missing cells
are kept as NaN; median / most-frequent imputation is applied inside the cross-validation
fold by encode.MissingValueImputer. Each dataset is cleaned in its own lane; output
data/processed/<set>_clean.csv.
"""
import pandas as pd

from . import config as cfg


def temizle_telco(df: pd.DataFrame):
    """Telco: TotalCharges is stored as text with blanks; convert to numeric.

    Rows with tenure=0 are set to 0 (no billing cycle has closed yet). Any remaining
    blank is left as NaN and imputed inside the fold.
    """
    df = df.copy()
    log = []
    tc = pd.to_numeric(df["TotalCharges"], errors="coerce")
    n_bos = int(tc.isna().sum())
    sifir_mask = (df["tenure"] == 0) & tc.isna()
    n_sifir = int(sifir_mask.sum())
    tc[sifir_mask] = 0.0
    n_kalan = int(tc.isna().sum())
    df["TotalCharges"] = tc
    log.append(("TotalCharges", "text -> numeric",
                f"{n_bos} blank; tenure=0 -> 0 ({n_sifir} rows); remaining {n_kalan} kept as NaN"))
    return df, log


def temizle_hepsi(etiketli: dict, holdout: dict):
    """Cleans all datasets, writes them under data/processed/.

    Only deterministic fixes are applied here; missing cells are carried forward as NaN
    and imputed within the fold. Returns: (processed: name->df, log: list of
    (set, column, operation, detail)).
    """
    cfg.klasorleri_hazirla()
    processed = {}
    log = []

    # Telco: TotalCharges type fix
    d, lg = temizle_telco(etiketli["telco"]["df"])
    processed["telco"] = d
    log += [("telco", *r) for r in lg]

    # Cell2Cell (train + unlabeled holdout) and E-commerce: missing cells kept as NaN
    for ad, df in (("cell2cell", etiketli["cell2cell"]["df"]),
                   ("cell2cell_test", holdout["df"]),
                   ("ecommerce", etiketli["ecommerce"]["df"])):
        d = df.copy()
        eksik = int(d.drop(columns=["churn"]).isna().sum().sum())
        processed[ad] = d
        log.append((ad, "-", "missing kept", f"{eksik} cells left as NaN, imputed within fold"))

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
