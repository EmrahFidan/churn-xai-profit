"""ADIM 7 — Sağlamlık: kâr-zinciri ablation + tekrarlı koşu (CI) + anlamlılık.

5 tohum (config.seeds) üstünden ortalama ± std + %95 CI. Model: HAM LightGBM
(Adım 2 best_params). Setler ayrı; birleştirme yok. OOF olasılıklar (set, model, seed)
bazında memoize edilir (parçalar arası tekrar hesap yok).
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.stats import t as t_dist
from scipy.stats import wilcoxon
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from . import config as cfg
from . import encode
from . import evaluate as ev
from . import imbalance as imb
from . import plotstyle as ps
from . import profit as pr
from . import strings_tr as S

SEEDS = cfg.CFG["seeds"]
C_ORAN = cfg.CFG["profit"]["emp_c_oran"]   # referans maliyet oranı (%5)
GAMMA = 0.30
ABLATION_SETLERI = ["telco", "cell2cell"]
_OOF = {}  # (set, model, seed) -> (p, y)


def _params(set_adi, anahtar):
    bp = json.loads((cfg.TABLES / f"best_params_{set_adi}.json").read_text(encoding="utf-8")).get(anahtar, {})
    return {k.replace("model__", ""): v for k, v in bp.items()}


def _pipeline(model_adi, set_adi, df, seed):
    """Koşula göre fit edilmemiş pipeline. model_adi: lgbm/logreg/xgboost/class_weight/smote."""
    parts = encode.parcalar(set_adi, df, olcekle=False)
    y = df["churn"].to_numpy()
    spw = float((y == 0).sum() / max(1, (y == 1).sum()))
    lp = _params(set_adi, "lightgbm")
    if model_adi == "lgbm":
        return imb._pipeline(parts, "baseline", lp, seed, spw)
    if model_adi == "class_weight":
        return imb._pipeline(parts, "class_weight", lp, seed, spw)
    if model_adi == "smote":
        return imb._pipeline(parts, "smote", lp, seed, spw)
    if model_adi == "logreg":
        pre = encode.on_isleyici(set_adi, df, olcekle=True)[0]
        C = _params(set_adi, "logreg").get("C", 1.0)
        return Pipeline([("pre", pre), ("model", LogisticRegression(C=C, max_iter=2000,
                         solver="liblinear", random_state=seed))])
    if model_adi == "xgboost":
        pre = encode.on_isleyici(set_adi, df, olcekle=False)[0]
        return Pipeline([("pre", pre), ("model", XGBClassifier(tree_method="hist", n_jobs=-1,
                         random_state=seed, eval_metric="logloss", **_params(set_adi, "xgboost")))])
    raise ValueError(model_adi)


def oof(set_adi, df, model_adi, seed):
    """5-kat OOF olasılık (memoize). Dönüş: (p, y)."""
    key = (set_adi, model_adi, seed)
    if key in _OOF:
        return _OOF[key]
    X = df.drop(columns=["churn"])
    y = df["churn"].to_numpy()
    p = np.zeros(len(y))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, va in skf.split(X, y):
        pipe = clone(_pipeline(model_adi, set_adi, df, seed))
        pipe.fit(X.iloc[tr], y[tr])
        p[va] = pipe.predict_proba(X.iloc[va])[:, 1]
    _OOF[key] = (p, y)
    return p, y


def _ci(vals):
    a = np.asarray(vals, float)
    m, s = a.mean(), (a.std(ddof=1) if len(a) > 1 else 0.0)
    h = t_dist.ppf(0.975, len(a) - 1) * s / np.sqrt(len(a)) if len(a) > 1 else 0.0
    return m, s, m - h, m + h


# ----------------------------- PARÇA 1: ablation -----------------------------
KOSUL_MODEL = {"K0": ("lgbm", "profit"), "K1": ("lgbm", "acc"),
               "K2": ("class_weight", "profit"), "K3": ("logreg", "profit"),
               "K4": ("smote", "profit")}


def ablation_set(set_adi, df):
    clv, _ = pr.clv_hesapla(set_adi, df)
    c = float(np.mean(clv)) * C_ORAN
    esikler = np.linspace(0.0, 1.0, 101)
    cikti = {}
    for kosul, (model_adi, strat) in KOSUL_MODEL.items():
        kar_l, roi_l, ece_l, esik_l = [], [], [], []
        for seed in SEEDS:
            p, y = oof(set_adi, df, model_adi, seed)
            if strat == "acc":
                t = 0.5
            else:
                t, _ = pr.en_iyi_esik(p, y, clv, c, GAMMA, esikler)
            kar_l.append(pr.kar(p, y, clv, c, GAMMA, t))
            roi_l.append(pr.roi(p, y, clv, c, GAMMA, t))
            ece_l.append(ev.ece(y, p))
            esik_l.append(t)
        m, s, lo, hi = _ci(kar_l)
        cikti[kosul] = {"kar": m, "kar_lo": lo, "kar_hi": hi, "roi": np.mean(roi_l),
                        "ece": np.mean(ece_l), "esik": np.mean(esik_l)}
    k0 = cikti["K0"]["kar"]
    for kosul in cikti:
        cikti[kosul]["dkar"] = cikti[kosul]["kar"] - k0
        cikti[kosul]["dkar_yuzde"] = 100 * (cikti[kosul]["kar"] - k0) / abs(k0) if k0 else np.nan
    return cikti


def tablo_ablation(tum_ablation):
    K = S.KOLON7
    rows = []
    for s, d in tum_ablation.items():
        for kosul, v in d.items():
            rows.append({
                K["veri_seti"]: s, K["kosul"]: S.KOSUL_ABLATION[kosul],
                K["kar_ort"]: round(v["kar"], 1), K["kar_ci_low"]: round(v["kar_lo"], 1),
                K["kar_ci_high"]: round(v["kar_hi"], 1), K["roi"]: round(v["roi"], 3),
                K["ece"]: round(v["ece"], 4), K["esik"]: round(v["esik"], 3),
                K["dkar"]: round(v["dkar"], 1), K["dkar_yuzde"]: round(v["dkar_yuzde"], 1),
            })
    df = pd.DataFrame(rows)
    df.to_csv(cfg.TABLES / "ablation_profit.csv", index=False)
    return df


# ----------------------------- PARÇA 2: CI -----------------------------
def robustness_set(set_adi, df):
    clv, _ = pr.clv_hesapla(set_adi, df)
    c = float(np.mean(clv)) * C_ORAN
    esikler = np.linspace(0.0, 1.0, 101)
    met = {m: [] for m in ["PR-AUC", "ROC-AUC", "recall", "precision", "F1", "EMP"]}
    for seed in SEEDS:
        p, y = oof(set_adi, df, "lgbm", seed)
        met["PR-AUC"].append(average_precision_score(y, p))
        met["ROC-AUC"].append(roc_auc_score(y, p))
        t, _ = pr.en_iyi_esik(p, y, clv, c, GAMMA, esikler)
        pred = (p >= t).astype(int)
        met["recall"].append(recall_score(y, pred, zero_division=0))
        met["precision"].append(precision_score(y, pred, zero_division=0))
        met["F1"].append(f1_score(y, pred, zero_division=0))
        met["EMP"].append(pr.emp(p, y, clv, c))
    return {m: _ci(v) for m, v in met.items()}


def tablo_ci(tum_ci):
    K = S.KOLON7
    rows = []
    for s, d in tum_ci.items():
        for metrik, (m, std, lo, hi) in d.items():
            rows.append({K["veri_seti"]: s, K["metrik"]: metrik, K["ort"]: round(m, 4),
                         K["std"]: round(std, 4), K["ci_low"]: round(lo, 4), K["ci_high"]: round(hi, 4)})
    df = pd.DataFrame(rows)
    df.to_csv(cfg.TABLES / "robustness_ci.csv", index=False)
    return df


# ----------------------------- PARÇA 3: anlamlılık -----------------------------
def anlamlilik(veriler):
    """Model çiftleri (set×seed OOF PR-AUC eşli) Wilcoxon + kâr-eşiği vs accuracy eşli test."""
    modeller = ["lgbm", "logreg", "xgboost"]
    ap = {m: [] for m in modeller}
    for s in cfg.DATASETS:
        for seed in SEEDS:
            for m in modeller:
                p, y = oof(s, veriler[s], m, seed)
                ap[m].append(average_precision_score(y, p))
    K = S.KOLON7
    rows = []
    pmat = {}
    for i, ma in enumerate(modeller):
        for mb in modeller[i + 1:]:
            st, pv = wilcoxon(ap[ma], ap[mb])
            pmat[(ma, mb)] = pv
            rows.append({K["kiyas"]: f"{ma} vs {mb} (PR-AUC)", K["test"]: "Wilcoxon",
                         K["istatistik"]: round(float(st), 2), K["p"]: round(float(pv), 5),
                         K["sonuc"]: S.ANLAMLI[pv < 0.05]})
    # kâr-eşiği (K0) vs accuracy (K1): telco+cell2cell × seed eşli kâr
    k0, k1 = [], []
    for s in ABLATION_SETLERI:
        clv, _ = pr.clv_hesapla(s, veriler[s])
        c = float(np.mean(clv)) * C_ORAN
        esikler = np.linspace(0.0, 1.0, 101)
        for seed in SEEDS:
            p, y = oof(s, veriler[s], "lgbm", seed)
            t, _ = pr.en_iyi_esik(p, y, clv, c, GAMMA, esikler)
            k0.append(pr.kar(p, y, clv, c, GAMMA, t))
            k1.append(pr.kar(p, y, clv, c, GAMMA, 0.5))
    st, pv = wilcoxon(k0, k1)
    rows.append({K["kiyas"]: "Kâr-eşiği (K0) vs accuracy (K1) — kâr", K["test"]: "Wilcoxon",
                 K["istatistik"]: round(float(st), 2), K["p"]: round(float(pv), 5),
                 K["sonuc"]: S.ANLAMLI[pv < 0.05]})
    df = pd.DataFrame(rows)
    df.to_csv(cfg.TABLES / "significance_tests.csv", index=False)
    return df, pmat, modeller, len(ap["lgbm"])


# ----------------------------- figürler -----------------------------
def _kaydet(fig, dosya):
    d = cfg.FIGURES / "_robust"
    d.mkdir(parents=True, exist_ok=True)
    yol = d / dosya
    fig.savefig(yol)
    return yol


def figur_ablation(tum_ablation):
    setler = list(tum_ablation.keys())
    kosullar = list(S.KOSUL_ABLATION.keys())
    fig, axes = plt.subplots(1, len(setler), figsize=(6.5 * len(setler), 5), squeeze=False)
    for ax, s in zip(axes[0], setler):
        d = tum_ablation[s]
        kar = [d[k]["kar"] for k in kosullar]
        err = [[d[k]["kar"] - d[k]["kar_lo"] for k in kosullar],
               [d[k]["kar_hi"] - d[k]["kar"] for k in kosullar]]
        renk = [ps.CHURN_RENK[0]] + [ps.CHURN_RENK[1]] * (len(kosullar) - 1)
        ax.bar(kosullar, kar, yerr=err, capsize=4, color=renk)
        ax.axhline(d["K0"]["kar"], color="#444444", linestyle="--", linewidth=1)
        for i, k in enumerate(kosullar):
            if k != "K0":
                ax.text(i, kar[i], f"{d[k]['dkar_yuzde']:+.0f}%", ha="center",
                        va="bottom" if kar[i] >= 0 else "top", fontsize=9)
        ax.set_title(s)
        ax.set_ylabel(S.EKSEN7["kar"])
        ax.axhline(0, color="black", linewidth=0.8)
    fig.suptitle(S.FIG7_BASLIK["ablation"])
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _kaydet(fig, S.FIG7_DOSYA["ablation"])


def figur_ci(tum_ci):
    setler = list(tum_ci.keys())
    m = [tum_ci[s]["PR-AUC"][0] for s in setler]
    lo = [tum_ci[s]["PR-AUC"][2] for s in setler]
    hi = [tum_ci[s]["PR-AUC"][3] for s in setler]
    y = np.arange(len(setler))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(m, y, xerr=[np.array(m) - np.array(lo), np.array(hi) - np.array(m)],
                fmt="o", color=ps.CHURN_RENK[0], capsize=5, markersize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(setler)
    ax.set_xlabel(S.EKSEN7["prauc"])
    ax.set_title(S.FIG7_BASLIK["ci"])
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return _kaydet(fig, S.FIG7_DOSYA["ci"])


def figur_significance(pmat, modeller):
    n = len(modeller)
    M = np.full((n, n), np.nan)
    for (a, b), pv in pmat.items():
        i, j = modeller.index(a), modeller.index(b)
        M[i, j] = pv
        M[j, i] = pv
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(M, cmap="RdYlGn_r", vmin=0, vmax=0.1)
    ax.set_xticks(range(n)); ax.set_xticklabels(modeller)
    ax.set_yticks(range(n)); ax.set_yticklabels(modeller)
    for i in range(n):
        for j in range(n):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=10,
                        color="white" if M[i, j] < 0.05 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="p-değeri")
    ax.set_title(S.FIG7_BASLIK["significance"])
    fig.tight_layout()
    return _kaydet(fig, S.FIG7_DOSYA["significance"])
