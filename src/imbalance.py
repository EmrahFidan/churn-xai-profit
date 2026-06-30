"""ADIM 3 — RQ1: Dengeleme yöntemleri karşılaştırması (LightGBM sabit).

Koşullar: baseline (doğal, eşik 0.5), class-weight (scale_pos_weight), SMOTENC/SMOTE,
ADASYN, eşik kaydırma (max-F1; istatistiksel, kâr değil). KAZANAN SEÇİLMEZ — trade-off
raporlanır.

SIZINTI KURALI: tüm resampling, scale_pos_weight ve eşik seçimi YALNIZ fold-içi eğitimde.
imblearn Pipeline: [prep] -> [resampler] -> [encode] -> [LightGBM]; resampler ham feature
üstünde (ADASYN istisna: encode'dan sonra). Validation katı doğal bırakılır (kalibrasyon
gerçek dağılımda ölçülür). Stratified 5-kat (seed=42), birincil skor PR-AUC.
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN, SMOTE, SMOTENC
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold

from . import config as cfg
from . import encode
from . import evaluate as ev
from . import plotstyle as ps
from . import strings_tr as S

KOSULLAR = ["baseline", "class_weight", "smote", "adasyn", "threshold"]
MODEL_KOSUL = ["baseline", "class_weight", "smote", "adasyn"]  # fit gereken (threshold baseline'ı kullanır)


def _lgbm_params(set_adi):
    """best_params_<set>.json'dan LightGBM hiperparametrelerini okur (model__ öneki sıyrılır)."""
    yol = cfg.TABLES / f"best_params_{set_adi}.json"
    bp = json.loads(yol.read_text(encoding="utf-8")).get("lightgbm", {})
    return {k.replace("model__", ""): v for k, v in bp.items()}


def _lgbm(params, seed, scale_pos_weight=None):
    p = dict(params)
    if scale_pos_weight is not None:
        p["scale_pos_weight"] = scale_pos_weight
    return LGBMClassifier(random_state=seed, n_jobs=-1, verbose=-1, **p)


def _resampler(parts, seed):
    """Kategorik varsa SMOTENC, tam sayısalsa SMOTE."""
    if parts["nominal"]:
        return SMOTENC(categorical_features=parts["kat_idx"], random_state=seed)
    return SMOTE(random_state=seed)


def _pipeline(parts, kosul, params, seed, spw):
    """Bir koşul için imblearn Pipeline kurar (fit edilmemiş)."""
    adimlar = []
    if parts["prep"] is not None:
        adimlar.append(("hazirla", clone(parts["prep"])))
    if kosul == "smote":
        adimlar.append(("resample", _resampler(parts, seed)))
        adimlar.append(("ct", clone(parts["ct"])))
        adimlar.append(("model", _lgbm(params, seed)))
    elif kosul == "adasyn":
        adimlar.append(("ct", clone(parts["ct"])))
        adimlar.append(("resample", ADASYN(random_state=seed)))
        adimlar.append(("model", _lgbm(params, seed)))
    elif kosul == "class_weight":
        adimlar.append(("ct", clone(parts["ct"])))
        adimlar.append(("model", _lgbm(params, seed, scale_pos_weight=spw)))
    else:  # baseline
        adimlar.append(("ct", clone(parts["ct"])))
        adimlar.append(("model", _lgbm(params, seed)))
    return ImbPipeline(adimlar)


def _f1_esik(y, p):
    """max-F1 veren eşiği döndürür (precision_recall_curve üzerinden)."""
    prec, rec, thr = precision_recall_curve(y, p)
    if len(thr) == 0:
        return 0.5
    f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
    return float(thr[int(np.nanargmax(f1))])


def _isletim_metrik(y, p, esik):
    pred = (p >= esik).astype(int)
    return (recall_score(y, pred, zero_division=0),
            precision_score(y, pred, zero_division=0),
            f1_score(y, pred, zero_division=0))


def calistir_set_rq1(set_adi, df, seed):
    """Bir set için tüm koşulları 5-kat CV ile değerlendirir.

    Dönüş: {kosul: {"oof": arr, "perfold": {metrik: [..]}}}. baseline'ın olasılığı
    eşik-kaydırma koşulunda yeniden kullanılır (aynı model).
    """
    X = df.drop(columns=["churn"])
    y = df["churn"].to_numpy()
    parts = encode.parcalar(set_adi, df, olcekle=False)
    params = _lgbm_params(set_adi)
    spw = float((y == 0).sum() / max(1, (y == 1).sum()))

    n = len(y)
    metr = ["PR-AUC", "ROC-AUC", "recall", "precision", "F1", "ECE", "Brier", "esik"]
    res = {k: {"oof": np.full(n, np.nan), "perfold": {m: [] for m in metr}} for k in KOSULLAR}

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, va in skf.split(X, y):
        Xtr, Xva = X.iloc[tr], X.iloc[va]
        ytr, yva = y[tr], y[va]
        p_baseline = None
        for kosul in MODEL_KOSUL:
            pipe = _pipeline(parts, kosul, params, seed, spw)
            pipe.fit(Xtr, ytr)
            p = pipe.predict_proba(Xva)[:, 1]
            res[kosul]["oof"][va] = p
            if kosul == "baseline":
                p_baseline = p
            esik = 0.5
            rec, pre, f1 = _isletim_metrik(yva, p, esik)
            pf = res[kosul]["perfold"]
            pf["PR-AUC"].append(average_precision_score(yva, p))
            pf["ROC-AUC"].append(roc_auc_score(yva, p))
            pf["recall"].append(rec); pf["precision"].append(pre); pf["F1"].append(f1)
            pf["ECE"].append(ev.ece(yva, p)); pf["Brier"].append(brier_score_loss(yva, p))
            pf["esik"].append(esik)
        # eşik kaydırma: baseline modeli, max-F1 eşiği
        esik = _f1_esik(yva, p_baseline)
        res["threshold"]["oof"][va] = p_baseline
        rec, pre, f1 = _isletim_metrik(yva, p_baseline, esik)
        pf = res["threshold"]["perfold"]
        pf["PR-AUC"].append(average_precision_score(yva, p_baseline))
        pf["ROC-AUC"].append(roc_auc_score(yva, p_baseline))
        pf["recall"].append(rec); pf["precision"].append(pre); pf["F1"].append(f1)
        pf["ECE"].append(ev.ece(yva, p_baseline)); pf["Brier"].append(brier_score_loss(yva, p_baseline))
        pf["esik"].append(esik)
    return res


# ----------------------------- tablolar -----------------------------
def _ort(x):
    a = np.asarray(x, float)
    return a.mean(), a.std()


def _fmt(x):
    m, s = _ort(x)
    return f"{m:.4f} ± {s:.4f}"


def tablo_karsilastirma(tum):
    """rq1_imbalance_comparison.csv (set × yöntem, mean ± std)."""
    K = S.KOLON3
    rows = []
    for s, res in tum.items():
        for k in KOSULLAR:
            pf = res[k]["perfold"]
            rows.append({
                K["veri_seti"]: s, K["yontem"]: S.KOSUL_AD[k],
                K["pr_auc"]: _fmt(pf["PR-AUC"]), K["roc_auc"]: _fmt(pf["ROC-AUC"]),
                K["recall"]: _fmt(pf["recall"]), K["precision"]: _fmt(pf["precision"]),
                K["f1"]: _fmt(pf["F1"]), K["ece"]: _fmt(pf["ECE"]), K["brier"]: _fmt(pf["Brier"]),
                K["esik"]: _fmt(pf["esik"]),
            })
    df = pd.DataFrame(rows)
    df.to_csv(cfg.TABLES / "rq1_imbalance_comparison.csv", index=False)
    return df


def tablo_esikler(tum):
    """rq1_thresholds.csv (set × yöntem, seçilen eşik mean ± std)."""
    K = S.KOLON3
    rows = []
    for s, res in tum.items():
        for k in KOSULLAR:
            rows.append({K["veri_seti"]: s, K["yontem"]: S.KOSUL_AD[k],
                         K["esik"]: _fmt(res[k]["perfold"]["esik"])})
    df = pd.DataFrame(rows)
    df.to_csv(cfg.TABLES / "rq1_thresholds.csv", index=False)
    return df


# ----------------------------- figürler -----------------------------
def _ortalama(res, k, metrik):
    return float(np.mean(res[k]["perfold"][metrik]))


def _std(res, k, metrik):
    return float(np.std(res[k]["perfold"][metrik]))


def figur_prauc(set_adi, res):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    deger = [_ortalama(res, k, "PR-AUC") for k in KOSULLAR]
    hata = [_std(res, k, "PR-AUC") for k in KOSULLAR]
    ax.bar([S.KOSUL_AD[k] for k in KOSULLAR], deger, yerr=hata, capsize=4,
           color=[ps.KOSUL_RENK[k] for k in KOSULLAR])
    ax.set_ylabel("PR-AUC")
    ax.set_title(S.FIG3_BASLIK["prauc"].format(set=set_adi))
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    return ev._kaydet(fig, set_adi, S.FIG3_DOSYA["prauc"])


def figur_recall_precision(set_adi, res):
    fig, ax = plt.subplots(figsize=(6, 5))
    for k in KOSULLAR:
        rec, pre = _ortalama(res, k, "recall"), _ortalama(res, k, "precision")
        ax.errorbar(rec, pre, xerr=_std(res, k, "recall"), yerr=_std(res, k, "precision"),
                    fmt="o", markersize=9, capsize=3, color=ps.KOSUL_RENK[k], label=S.KOSUL_AD[k])
    ax.set_xlabel(S.KOLON3["recall"] + " (recall)")
    ax.set_ylabel(S.KOLON3["precision"] + " (precision)")
    ax.set_title(S.FIG3_BASLIK["recall_precision"].format(set=set_adi))
    ax.legend(fontsize=9)
    fig.tight_layout()
    return ev._kaydet(fig, set_adi, S.FIG3_DOSYA["recall_precision"])


def figur_calibration(set_adi, res):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    deger = [_ortalama(res, k, "ECE") for k in KOSULLAR]
    hata = [_std(res, k, "ECE") for k in KOSULLAR]
    ax.bar([S.KOSUL_AD[k] for k in KOSULLAR], deger, yerr=hata, capsize=4,
           color=[ps.KOSUL_RENK[k] for k in KOSULLAR])
    ax.set_ylabel("ECE")
    ax.set_title(S.FIG3_BASLIK["calibration"].format(set=set_adi))
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    return ev._kaydet(fig, set_adi, S.FIG3_DOSYA["calibration"])


def figur_pr_operating(set_adi, res, y):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for k in KOSULLAR:
        p = res[k]["oof"]
        prec, rec, _ = precision_recall_curve(y, p)
        ap = _ortalama(res, k, "PR-AUC")
        ax.plot(rec, prec, color=ps.KOSUL_RENK[k], alpha=0.85,
                label=f"{S.KOSUL_AD[k]} (PR-AUC={ap:.3f})")
        # işletim noktası
        r_op, p_op = _ortalama(res, k, "recall"), _ortalama(res, k, "precision")
        ax.scatter([r_op], [p_op], color=ps.KOSUL_RENK[k], edgecolor="black", zorder=5, s=60)
    ax.axhline(y.mean(), color="#444444", linestyle=":", label=f"Taban (prevalans={y.mean():.3f})")
    ax.set_xlabel(S.KOLON3["recall"] + " (recall)")
    ax.set_ylabel(S.KOLON3["precision"] + " (precision)")
    ax.set_ylim(0, 1)
    ax.set_title(S.FIG3_BASLIK["pr_operating"].format(set=set_adi))
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    return ev._kaydet(fig, set_adi, S.FIG3_DOSYA["pr_operating"])


def figurler_set(set_adi, res, y):
    return {
        "prauc": figur_prauc(set_adi, res),
        "recall_precision": figur_recall_precision(set_adi, res),
        "calibration": figur_calibration(set_adi, res),
        "pr_operating": figur_pr_operating(set_adi, res, y),
    }
