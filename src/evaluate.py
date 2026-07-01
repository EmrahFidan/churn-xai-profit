"""Evaluation: protocol, metrics, figures and tables.

Each labeled set × each model: stratified 5-fold CV (seed=42), primary score PR-AUC.
Encoding + (scaling) + model + calibration in a single pipeline, fit within the fold
(no leakage). Calibration quality is computed separately with Brier + ECE for
raw/Platt/Isotonic. NO RESAMPLING — an honest baseline on the natural distribution.
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier

from . import calibration as cal_mod
from . import config as cfg
from . import encode
from . import models as models_mod
from . import plotstyle as ps
from . import strings as S

YONTEM_ANAHTAR = ["ham", "Platt", "Isotonic"]


# ----------------------------- metrics -----------------------------
def ece(y, p, bins: int = 10) -> float:
    """Expected Calibration Error (10 equal-width bins, weighted |accuracy-confidence|)."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    kenar = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, kenar[1:-1]), 0, bins - 1)
    toplam = 0.0
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        toplam += abs(y[m].mean() - p[m].mean()) * m.sum() / len(p)
    return float(toplam)


def _ozet(deg):
    a = np.asarray(deg, dtype=float)
    return a.mean(), a.std()


def _fmt(deg) -> str:
    m, s = _ozet(deg)
    return f"{m:.4f} ± {s:.4f}"


# --------------------------- fold evaluation ---------------------------
def degerlendir(fab, best_params, X, y, seed):
    """5-fold evaluation + calibration for one (set, model).

    Return: {"oof": {ham,Platt,Isotonic}, "perfold": {...metric lists...}}.
    Discrimination metrics (PR/ROC, 0.5 threshold) from the uncalibrated probability;
    Brier/ECE separately for all three methods.
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    n = len(y)
    oof = {k: np.full(n, np.nan) for k in YONTEM_ANAHTAR}
    pf = {
        "ham": {k: [] for k in ["PR-AUC", "ROC-AUC", "recall", "precision", "F1", "Brier", "ECE"]},
        "Platt": {"Brier": [], "ECE": []},
        "Isotonic": {"Brier": [], "ECE": []},
    }
    for tr, va in skf.split(X, y):
        Xtr, Xva = X.iloc[tr], X.iloc[va]
        ytr, yva = y[tr], y[va]
        base = fab()
        base.set_params(**best_params)
        base.fit(Xtr, ytr)
        p = base.predict_proba(Xva)[:, 1]
        oof["ham"][va] = p
        pred = (p >= 0.5).astype(int)
        pf["ham"]["PR-AUC"].append(average_precision_score(yva, p))
        pf["ham"]["ROC-AUC"].append(roc_auc_score(yva, p))
        pf["ham"]["recall"].append(recall_score(yva, pred, zero_division=0))
        pf["ham"]["precision"].append(precision_score(yva, pred, zero_division=0))
        pf["ham"]["F1"].append(f1_score(yva, pred, zero_division=0))
        pf["ham"]["Brier"].append(brier_score_loss(yva, p))
        pf["ham"]["ECE"].append(ece(yva, p))
        for method, key in cal_mod.YONTEMLER:
            c = cal_mod.kalibre(fab, best_params, method)
            c.fit(Xtr, ytr)
            pc = c.predict_proba(Xva)[:, 1]
            oof[key][va] = pc
            pf[key]["Brier"].append(brier_score_loss(yva, pc))
            pf[key]["ECE"].append(ece(yva, pc))
    return {"oof": oof, "perfold": pf}


def calistir_set(set_adi, df, seed, model_adlari=None, kaydet_sema=True):
    """Runs all models for one set (HPO + evaluation).

    Writes encoding_schema_<set>.csv and best_params_<set>.json. Return:
    {"sonuc": {model: degerlendir output + hpo_skor}, "best_params": {...}, "sema": ...}.
    """
    model_adlari = model_adlari or models_mod.MODEL_ADLARI
    X = df.drop(columns=["churn"])
    y = df["churn"].to_numpy()

    pre_olcekli = encode.on_isleyici(set_adi, df, True)[0]
    pre_olceksiz, sema = encode.on_isleyici(set_adi, df, False)
    if kaydet_sema:
        encode.sema_yaz(set_adi, sema)

    def fabrika(est, olcekle):
        pre = pre_olcekli if olcekle else pre_olceksiz
        return lambda: Pipeline([("pre", clone(pre)), ("model", clone(est))])

    sonuc = {}
    best_params = {}
    for madi in model_adlari:
        est, dist, n_iter, olcekle = models_mod.model_uzayi(madi, seed)
        fab = fabrika(est, olcekle)
        bp, bscore = models_mod.hpo(fab(), dist, n_iter, X, y, seed)
        best_params[madi] = bp
        res = degerlendir(fab, bp, X, y, seed)
        res["hpo_skor"] = bscore
        sonuc[madi] = res

    (cfg.TABLES / f"best_params_{set_adi}.json").write_text(
        json.dumps(best_params, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"sonuc": sonuc, "best_params": best_params, "sema": sema}


# ----------------------------- helpers -----------------------------
def pr_ortalama(res):
    return float(np.mean(res["perfold"]["ham"]["PR-AUC"]))


def ece_ortalamalari(res):
    return {k: float(np.mean(res["perfold"][k]["ECE"])) for k in YONTEM_ANAHTAR}


def en_iyi_model(set_sonuc):
    """Returns the best model key by mean PR-AUC."""
    return max(set_sonuc["sonuc"], key=lambda m: pr_ortalama(set_sonuc["sonuc"][m]))


def en_iyi_kalibrasyon(res):
    """Returns the method with the lowest ECE (ham/Platt/Isotonic)."""
    e = ece_ortalamalari(res)
    return min(e, key=e.get), e


# ----------------------------- tables -----------------------------
def tablo_performans(tum):
    """model_performance.csv — set × model, mean ± std."""
    K = S.KOLON2
    rows = []
    for s, r in tum.items():
        for m, res in r["sonuc"].items():
            pf = res["perfold"]["ham"]
            rows.append({
                K["veri_seti"]: s, K["model"]: S.MODEL_AD[m],
                K["pr_auc"]: _fmt(pf["PR-AUC"]), K["roc_auc"]: _fmt(pf["ROC-AUC"]),
                K["recall"]: _fmt(pf["recall"]), K["precision"]: _fmt(pf["precision"]),
                K["f1"]: _fmt(pf["F1"]),
            })
    df = pd.DataFrame(rows)
    df.to_csv(cfg.TABLES / "model_performance.csv", index=False)
    return df


def tablo_kalibrasyon(tum):
    """calibration_comparison.csv — set × model × method, Brier + ECE."""
    K = S.KOLON2
    rows = []
    for s, r in tum.items():
        for m, res in r["sonuc"].items():
            pf = res["perfold"]
            for key in YONTEM_ANAHTAR:
                rows.append({
                    K["veri_seti"]: s, K["model"]: S.MODEL_AD[m],
                    K["yontem"]: S.YONTEM_AD[key],
                    K["brier"]: _fmt(pf[key]["Brier"]), K["ece"]: _fmt(pf[key]["ECE"]),
                })
    df = pd.DataFrame(rows)
    df.to_csv(cfg.TABLES / "calibration_comparison.csv", index=False)
    return df


# ----------------------------- figures -----------------------------
def _kaydet(fig, set_adi, dosya):
    d = cfg.FIGURES / set_adi
    d.mkdir(parents=True, exist_ok=True)
    yol = d / dosya
    fig.savefig(yol)
    return yol


def figur_pr(set_adi, set_sonuc, y):
    """pr_curve.png — all models together (out-of-fold)."""
    fig, ax = plt.subplots(figsize=(6, 5))
    for m, res in set_sonuc["sonuc"].items():
        p = res["oof"]["ham"]
        prec, rec, _ = precision_recall_curve(y, p)
        ap = pr_ortalama(res)
        ax.plot(rec, prec, color=ps.MODEL_RENK[m], label=f"{S.MODEL_AD[m]} (PR-AUC={ap:.3f})")
    taban = y.mean()
    ax.axhline(taban, color="#7F7F7F", linestyle=":", label=f"Baseline (prevalence={taban:.3f})")
    ax.set_title(S.FIG2_BASLIK["pr"].format(set=set_adi))
    ax.set_xlabel(S.EKSEN2["recall"])
    ax.set_ylabel(S.EKSEN2["precision"])
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    return _kaydet(fig, set_adi, S.FIG2_DOSYA["pr"])


def figur_roc(set_adi, set_sonuc, y):
    """roc_curve.png — all models together (out-of-fold)."""
    fig, ax = plt.subplots(figsize=(6, 5))
    for m, res in set_sonuc["sonuc"].items():
        p = res["oof"]["ham"]
        fpr, tpr, _ = roc_curve(y, p)
        auc = float(np.mean(res["perfold"]["ham"]["ROC-AUC"]))
        ax.plot(fpr, tpr, color=ps.MODEL_RENK[m], label=f"{S.MODEL_AD[m]} (ROC-AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], color="#7F7F7F", linestyle=":", label="Random")
    ax.set_title(S.FIG2_BASLIK["roc"].format(set=set_adi))
    ax.set_xlabel(S.EKSEN2["fpr"])
    ax.set_ylabel(S.EKSEN2["tpr"])
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    return _kaydet(fig, set_adi, S.FIG2_DOSYA["roc"])


def figur_kalibrasyon(set_adi, model_adi, res, y):
    """calibration_curves.png — the best model's ham/Platt/Isotonic reliability curves."""
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot([0, 1], [0, 1], color="black", linestyle="-", linewidth=0.8, label="Perfect")
    for key in YONTEM_ANAHTAR:
        p = res["oof"][key]
        frac, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
        brier = float(np.mean(res["perfold"][key]["Brier"]))
        e = float(np.mean(res["perfold"][key]["ECE"]))
        st = ps.YONTEM_STIL[key]
        ax.plot(mean_pred, frac, marker="o", markersize=4, color=st["color"],
                linestyle=st["linestyle"],
                label=f"{S.YONTEM_AD[key]} (Brier={brier:.3f}, ECE={e:.3f})")
    ax.set_title(S.FIG2_BASLIK["calibration"].format(model=S.MODEL_AD[model_adi], set=set_adi))
    ax.set_xlabel(S.EKSEN2["tahmin_olasilik"])
    ax.set_ylabel(S.EKSEN2["gercek_oran"])
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    return _kaydet(fig, set_adi, S.FIG2_DOSYA["calibration"])


def figur_model_kiyas(set_adi, set_sonuc):
    """model_comparison.png — PR-AUC / Recall / F1 (models side by side)."""
    modeller = list(set_sonuc["sonuc"].keys())
    metr = [("PR-AUC", "PR-AUC"), ("recall", S.KOLON2["recall"]), ("F1", "F1")]
    x = np.arange(len(metr))
    w = 0.8 / len(modeller)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, m in enumerate(modeller):
        pf = set_sonuc["sonuc"][m]["perfold"]["ham"]
        deger = [np.mean(pf[k]) for k, _ in metr]
        hata = [np.std(pf[k]) for k, _ in metr]
        ax.bar(x + i * w, deger, w, yerr=hata, capsize=3,
               color=ps.MODEL_RENK[m], label=S.MODEL_AD[m])
    ax.set_xticks(x + 0.4 - w / 2)
    ax.set_xticklabels([etk for _, etk in metr])
    ax.set_ylabel(S.EKSEN2["metrik_deger"])
    ax.set_title(S.FIG2_BASLIK["model_comparison"].format(set=set_adi))
    ax.legend(fontsize=9)
    fig.tight_layout()
    return _kaydet(fig, set_adi, S.FIG2_DOSYA["model_comparison"])


def figurler_set(set_adi, set_sonuc, y):
    """Produces the 4 figures for one set. Return: {key: path}."""
    en_iyi = en_iyi_model(set_sonuc)
    yollar = {
        "pr": figur_pr(set_adi, set_sonuc, y),
        "roc": figur_roc(set_adi, set_sonuc, y),
        "calibration": figur_kalibrasyon(set_adi, en_iyi, set_sonuc["sonuc"][en_iyi], y),
        "model_comparison": figur_model_kiyas(set_adi, set_sonuc),
    }
    return yollar, en_iyi


# ------------------------ iranian Status dominance ------------------------
def iranian_status_etkisi(df, seed):
    """PR-AUC change when 'Status' is dropped in iranian (LGBM, 5-fold). Return: (full, drop)."""
    def ap(d):
        X = d.drop(columns=["churn"])
        y = d["churn"].to_numpy()
        pre = encode.on_isleyici("iranian", d, False)[0]
        pipe = Pipeline([("pre", pre), ("model", LGBMClassifier(n_jobs=-1, random_state=seed, verbose=-1))])
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        return float(cross_val_score(pipe, X, y, scoring="average_precision", cv=cv).mean())
    return ap(df), ap(df.drop(columns=["Status"]))
