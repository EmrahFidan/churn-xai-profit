# %% [markdown]
# # STEP 1 — Diagnosis & Cleaning + EDA + Leakage audit
#
# 5 churn datasets are processed **independently**; sets are never merged, outputs
# are separate per set. In this step **no encoding is done** (categoricals are left raw)
# and **no modeling is done**. The heavy logic is in the `src/` functions; this script
# only calls them in order (can be run cell by cell with Shift+Enter).

# %% Setup
# Common setup: imports + seed(42) + shared figure style.
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _bul_kok():
    for c in [Path.cwd(), *Path.cwd().parents]:
        if (c / "config.yaml").exists():
            return c
    raise RuntimeError("config.yaml not found — run from the repo root")


KOK = _bul_kok()
if str(KOK) not in sys.path:
    sys.path.insert(0, str(KOK))

from src import clean as clean_mod
from src import config as cfg
from src import eda as eda_mod
from src import leakage as leak_mod
from src import load as load_mod
from src import plotstyle as ps
from src import strings as S

random.seed(cfg.SEED)
np.random.seed(cfg.SEED)
ps.uygula()
cfg.klasorleri_hazirla()
pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 30)

CIKTI = []  # console summary; written under outputs/logs/ at the end


def yaz(s=""):
    print(s)
    CIKTI.append(str(s))


# %% [markdown]
# ## 1. Pre-check and profile
# Do the files exist, does the actual profile (rows/columns/churn) match the reference values?

# %% Pre-check and profile
yaz(S.MSG["bolum"].format(ad="PRE-CHECK & PROFILE"))
_eksik = [d["file"] for d in cfg.DATASETS.values() if not (cfg.RAW / d["file"]).exists()]
if not (cfg.HOLDOUT / cfg.HOLDOUT_DOSYA).exists():
    _eksik.append(cfg.HOLDOUT_DOSYA)
assert not _eksik, S.MSG["on_kontrol_eksik"].format(dosya=", ".join(_eksik))

etiketli = load_mod.yukle_etiketli()
holdout = load_mod.yukle_holdout()

for key, d in cfg.DATASETS.items():
    info = etiketli[key]
    churn = 100 * info["df"]["churn"].mean()
    r = cfg.REFERANS[key]
    assert info["raw_shape"][0] == r[0], f"{key}: row {info['raw_shape'][0]} != ref {r[0]}"
    yaz(S.MSG["profil_satiri"].format(
        set=key, satir=info["raw_shape"][0], sutun=info["raw_shape"][1],
        churn=churn, ref=f"{r[2]}%", hedef=d["target"]))
    if info["dropped"]:
        yaz("    " + S.MSG["id_dusuruldu"].format(set=key, kolonlar=info["dropped"]))

# unlabeled holdout validation
assert holdout["df"]["churn"].isna().all(), "cell2cell_test appears to be labeled"
yaz(f"{'cell2cell_test':11s} {holdout['raw_shape'][0]:>6d}x{holdout['raw_shape'][1]:<3d}  UNLABELED holdout")
yaz(S.MSG["on_kontrol_ok"].format(n=len(cfg.DATASETS)))

# %% [markdown]
# ## 2. Cleaning
# Only safe operations per set (missing-value imputation + type correction). Output:
# `data/processed/<set>_clean.csv`. The cell2cell holdout is imputed with train statistics.

# %% Cleaning
yaz(S.MSG["bolum"].format(ad="CLEANING"))
processed, temizlik_log = clean_mod.temizle_hepsi(etiketli, holdout)
for s, kolon, islem, detay in temizlik_log:
    yaz(S.MSG["temizlik_satiri"].format(set=s, kolon=kolon, islem=f"{islem} | {detay}"))

# temizlik_log -> CSV (Turkish headers)
clean_df = pd.DataFrame(temizlik_log, columns=[
    S.KOLON["veri_seti"], S.KOLON["kolon"], S.KOLON["islem"], S.KOLON["detay"]])
clean_df.to_csv(cfg.TABLES / "cleaning_log.csv", index=False)
yaz(S.MSG["kayit"].format(yol=cfg.TABLES / "cleaning_log.csv"))

# %% [markdown]
# ## 3. Leakage audit
# For each labeled set, univariate churn AUC per feature; ≥ 0.90 SUSPICIOUS.
# **Report only** — no column is dropped, the decision is left to the user.

# %% Leakage audit
yaz(S.MSG["bolum"].format(ad="LEAKAGE AUDIT"))
leak_df = leak_mod.denetle_hepsi(processed)
supheli = leak_df[leak_df[S.KOLON["bayrak"]] == S.BAYRAK_SUPHELI]
if len(supheli):
    for _, r in supheli.iterrows():
        yaz(S.MSG["leakage_supheli"].format(
            set=r[S.KOLON["veri_seti"]], ozellik=r[S.KOLON["ozellik"]],
            auc=r[S.KOLON["tekil_auc"]], aksiyon=r[S.KOLON["aksiyon"]]))
else:
    yaz("No suspicious (AUC ≥ 0.90) feature found.")
yaz(S.MSG["kayit"].format(yol=cfg.TABLES / "leakage_audit.csv"))

# %% [markdown]
# ## 4. EDA figures and overview
# For each labeled set, 5 figures + an overview table under `outputs/figures/<set>/`.

# %% EDA figures
yaz(S.MSG["bolum"].format(ad="EDA FIGURES"))
for key in cfg.DATASETS:
    yollar, atlanan = eda_mod.figurler_set(key, processed[key], etiketli[key]["df"])
    yaz(f"{key}: {len(yollar)} figures -> {cfg.FIGURES / key}")
    if atlanan:
        yaz(f"    (high cardinality, skipped in categorical figure: {atlanan})")

overview = eda_mod.genel_bakis(etiketli, processed, holdout)
yaz(S.MSG["kayit"].format(yol=cfg.TABLES / "dataset_overview.csv"))
yaz(overview.to_string(index=False))

# %% [markdown]
# ## 5. Summary
# The console summary is also written under `outputs/logs/`. The drop decision (leakage)
# was left to the user; no modeling was done.

# %% Summary
yaz(S.MSG["bolum"].format(ad="SUMMARY"))
yaz(f"Processed labeled sets: {len(cfg.DATASETS)} | unlabeled holdout: 1")
yaz(f"Cleaning decision rows: {len(temizlik_log)} | suspicious features: {len(supheli)}")
yaz(S.MSG["bitti"])

_log_yol = cfg.LOGS / "adim1_ozet.log"
_log_yol.write_text("\n".join(CIKTI) + "\n", encoding="utf-8")
print(S.MSG["kayit"].format(yol=_log_yol))
