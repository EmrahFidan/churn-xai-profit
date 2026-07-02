"""STEP 6 — Conditional transfer probe (thresholded, honest).

Measures how well a model trained on one set generalizes to another set. Sets are
not merged; transfer = "fit on source, predict on target". Shared feature space:
concept_map concepts that have a NUMERIC representative in BOTH sets (one
representative per concept, the highest-importance numeric feature in source/target).
For the scale difference, the scaler is fit ONLY on the source (no leakage from
target). Model: RAW LightGBM (source best_params). The result is assigned to three
categories by threshold: INCLUDED / PARTIAL / WEAK.
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from pandas.api.types import is_numeric_dtype
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from . import concept_map as km
from . import config as cfg
from . import plotstyle as ps
from . import evaluate as ev
from . import strings as S

# scenarios: (anahtar, ad, kaynak, hedef, tip)
SENARYOLAR = [
    ("A1", "Cell2Cell -> Iranian", "cell2cell", "iranian", "intra-sector"),
    ("A2", "Iranian -> Cell2Cell", "iranian", "cell2cell", "intra-sector"),
    ("B1", "telco -> bank", "telco", "bank", "cross-sector"),
    ("B2", "bank -> telco", "bank", "telco", "cross-sector"),
]
ESIK_DAHIL = 0.70    # retention ratio threshold
TRIVIAL_KAT = 1.10   # coefficient for "clearly beating" trivial


def _lgbm_params(set_adi):
    bp = json.loads((cfg.TABLES / f"best_params_{set_adi}.json").read_text(encoding="utf-8")).get("lightgbm", {})
    return {k.replace("model__", ""): v for k, v in bp.items()}


def temsilciler(set_adi, df):
    """Concept -> highest-importance NUMERIC representative feature (in that set)."""
    imp = pd.read_csv(cfg.TABLES / f"rq2_global_importance_{set_adi}.csv")
    skor = dict(zip(imp[S.KOLON4["feature"]], imp[S.KOLON4["mean_abs_shap"]]))
    rep = {}
    for c in df.columns:
        if c == "churn" or not is_numeric_dtype(df[c]):
            continue
        k = km.kavram(c)
        s = skor.get(c, 0.0)
        if k not in rep or s > rep[k][1]:
            rep[k] = (c, s)
    return {k: v[0] for k, v in rep.items()}


def ortak_kavramlar(kaynak_set, kaynak_df, hedef_set, hedef_df):
    """Concepts that have a numeric representative in both sets + mapping. Returns: (concepts, map)."""
    rk = temsilciler(kaynak_set, kaynak_df)
    rh = temsilciler(hedef_set, hedef_df)
    ortak = [k for k in km.KAVRAM_SIRA if k in rk and k in rh]
    esleme = {k: (rk[k], rh[k]) for k in ortak}
    return ortak, esleme


def _matris(df, esleme, rol):
    """Matrix in the shared concept space (columns = concepts, ordered)."""
    idx = 0 if rol == "kaynak" else 1
    kols = [esleme[k][idx] for k in esleme]
    X = df[kols].astype(float).copy()
    X.columns = list(esleme.keys())
    return X


def in_domain_ref(hedef_set, hedef_df, esleme, seed):
    """5-fold OOF PR-AUC WITHIN the target set itself, in the shared-concept space (ceiling)."""
    X = _matris(hedef_df, esleme, "hedef")
    y = hedef_df["churn"].to_numpy()
    params = _lgbm_params(hedef_set)
    p = np.zeros(len(y))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, va in skf.split(X, y):
        sc = StandardScaler().fit(X.iloc[tr])
        m = LGBMClassifier(random_state=seed, n_jobs=-1, verbose=-1, **params)
        m.fit(sc.transform(X.iloc[tr]), y[tr])
        p[va] = m.predict_proba(sc.transform(X.iloc[va]))[:, 1]
    return float(average_precision_score(y, p))


def tam_ref(hedef_set):
    """Step 2 full-feature LightGBM PR-AUC (context; 'x.xxxx ± ...' -> float)."""
    perf = pd.read_csv(cfg.TABLES / "model_performance.csv")
    K = S.KOLON2
    satir = perf[(perf[K["veri_seti"]] == hedef_set) & (perf[K["model"]] == "LightGBM")]
    if len(satir):
        return float(str(satir.iloc[0][K["pr_auc"]]).split("±")[0])
    return np.nan


# semantic (hand-fixed) representatives — not selected by importance; same operational phenomenon
SEMANTIK = [
    ("Usage volume (duration)", "MonthlyMinutes", "Seconds of Use", "minute/duration-based usage volume"),
    ("Usage frequency (call count)", "PeakCallsInOut", "Frequency of use",
     "peak incoming/outgoing call count: the single measure of usage frequency (minutes is a separate concept)"),
    ("Complaint/support", "CustomerCareCalls", "Complains", "customer service contact ~ complaint signal"),
    ("Relationship duration", "MonthsInService", "Subscription  Length", "subscription/service tenure"),
    ("Monetary value", "MonthlyRevenue", "Customer Value", "the customer's monetary value"),
]


def _karar(transfer, ref, trivial, oran):
    if transfer < trivial * TRIVIAL_KAT:
        return "zayif"
    return "dahil" if oran >= ESIK_DAHIL else "kismi"


def _degerlendir(anahtar, ad, kaynak_set, hedef_set, tip, esleme, esleme_tipi, veriler, seed):
    """Measures transfer using the given concept->(source_col, target_col) mapping."""
    ksrc, ktgt = veriler[kaynak_set], veriler[hedef_set]
    Xk = _matris(ksrc, esleme, "kaynak")
    yk = ksrc["churn"].to_numpy()
    Xh = _matris(ktgt, esleme, "hedef")
    yh = ktgt["churn"].to_numpy()

    sc = StandardScaler().fit(Xk)
    model = LGBMClassifier(random_state=seed, n_jobs=-1, verbose=-1, **_lgbm_params(kaynak_set))
    model.fit(sc.transform(Xk), yk)
    p = model.predict_proba(sc.transform(Xh))[:, 1]

    transfer = float(average_precision_score(yh, p))
    ref = in_domain_ref(hedef_set, ktgt, esleme, seed)
    trivial = float(yh.mean())
    oran = transfer / ref if ref > 0 else 0.0
    pred = (p >= 0.5).astype(int)
    return {
        "anahtar": anahtar, "ad": ad, "tip": tip, "esleme_tipi": esleme_tipi,
        "ortak": list(esleme.keys()), "esleme": esleme,
        "transfer": transfer, "ref": ref, "tam_ref": tam_ref(hedef_set),
        "trivial": trivial, "oran": oran,
        "recall": float(recall_score(yh, pred, zero_division=0)),
        "precision": float(precision_score(yh, pred, zero_division=0)),
        "f1": float(f1_score(yh, pred, zero_division=0)),
        "karar": _karar(transfer, ref, trivial, oran),
    }


def calistir_senaryo(anahtar, ad, kaynak_set, hedef_set, tip, veriler, seed):
    """Transfer with importance-based representatives (automatic shared-concept mapping)."""
    _, esleme = ortak_kavramlar(kaynak_set, veriler[kaynak_set], hedef_set, veriler[hedef_set])
    return _degerlendir(anahtar, ad, kaynak_set, hedef_set, tip, esleme, "onem", veriler, seed)


def calistir_semantik(anahtar, ad, kaynak_set, hedef_set, veriler, seed):
    """Transfer with semantic (hand-fixed) representatives — only the within-telecom pair."""
    esleme, atlanan = {}, []
    for label, c2c, iran, _ in SEMANTIK:
        if c2c not in veriler["cell2cell"].columns or iran not in veriler["iranian"].columns:
            atlanan.append(label)
            continue
        esleme[label] = (c2c, iran) if kaynak_set == "cell2cell" else (iran, c2c)
    r = _degerlendir(anahtar, ad, kaynak_set, hedef_set, "intra-sector", esleme, "semantik", veriler, seed)
    r["atlanan"] = atlanan
    return r


# ----------------------------- tables -----------------------------
def tablo_sonuc(sonuclar):
    K = S.KOLON6
    rows = []
    for r in sonuclar:
        rows.append({
            K["senaryo"]: r["anahtar"], K["yon"]: r["ad"],
            K["esleme_tipi"]: S.ESLEME_TIPI[r.get("esleme_tipi", "onem")],
            K["ortak_kavram"]: len(r["ortak"]),
            K["transfer"]: round(r["transfer"], 4), K["ref"]: round(r["ref"], 4),
            K["tam_ref"]: round(r["tam_ref"], 4), K["trivial"]: round(r["trivial"], 4),
            K["oran"]: round(r["oran"], 3),
            K["recall"]: round(r["recall"], 4), K["precision"]: round(r["precision"], 4),
            K["f1"]: round(r["f1"], 4), K["karar"]: S.TRANSFER_KARAR[r["karar"]],
        })
    df = pd.DataFrame(rows)
    df.to_csv(cfg.TABLES / "transfer_results.csv", index=False)
    return df


def tablo_feature_map(sonuclar):
    K = S.KOLON6
    rows = []
    for r in sonuclar:
        for k in r["ortak"]:
            ks, ht = r["esleme"][k]
            rows.append({K["senaryo"]: r["anahtar"], K["kavram"]: km.KAVRAM_AD[k],
                         K["kaynak_feature"]: ks, K["hedef_feature"]: ht})
    df = pd.DataFrame(rows)
    df.to_csv(cfg.TABLES / "transfer_feature_map.csv", index=False)
    return df


def tablo_feature_map_semantic():
    """Semantic (hand-fixed) mapping + selection rationales."""
    K = S.KOLON6
    rows = [{K["kavram"]: label, K["c2c_kolon"]: c2c, K["iran_kolon"]: iran, K["gerekce"]: ger}
            for label, c2c, iran, ger in SEMANTIK]
    df = pd.DataFrame(rows)
    df.to_csv(cfg.TABLES / "transfer_feature_map_semantic.csv", index=False)
    return df


# ----------------------------- figures -----------------------------
def _kaydet(fig, dosya):
    d = cfg.FIGURES / "_transfer"
    d.mkdir(parents=True, exist_ok=True)
    yol = d / dosya
    fig.savefig(yol)
    return yol


def figur_prauc(sonuclar):
    etk = [r["anahtar"] for r in sonuclar]
    x = np.arange(len(sonuclar))
    w = 0.26
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w, [r["transfer"] for r in sonuclar], w, label="Transfer", color=ps.CHURN_RENK[1])
    ax.bar(x, [r["ref"] for r in sonuclar], w, label="In-domain reference", color=ps.CHURN_RENK[0])
    ax.bar(x + w, [r["trivial"] for r in sonuclar], w, label="Trivial (prevalence)", color="#7F7F7F")
    for i, r in enumerate(sonuclar):
        ax.plot([i - 1.4 * w, i + 0.4 * w], [r["ref"] * ESIK_DAHIL] * 2, color="black",
                linestyle=":", linewidth=1.2)
    ax.plot([], [], color="black", linestyle=":", label=f"Threshold ({int(ESIK_DAHIL*100)}% ref)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['anahtar']}\n{r['ad']}" for r in sonuclar], fontsize=8)
    ax.set_ylabel(S.EKSEN6["prauc"])
    ax.set_title(S.FIG6_BASLIK["prauc"])
    ax.legend(fontsize=9)
    fig.tight_layout()
    return _kaydet(fig, S.FIG6_DOSYA["prauc"])


def figur_retention(sonuclar):
    etk = [r["anahtar"] for r in sonuclar]
    oran = [r["oran"] for r in sonuclar]
    renk = [ps.KOSUL_RENK["smote"] if r["tip"] == "intra-sector" else ps.KOSUL_RENK["adasyn"] for r in sonuclar]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(len(sonuclar)), oran, color=renk)
    ax.axhspan(0.70, 0.80, color="green", alpha=0.12, label="Threshold band (70–80%)")
    ax.axhline(1.0, color="#444444", linestyle="--", linewidth=1, label="Reference level (1.0)")
    ax.set_xticks(range(len(sonuclar)))
    ax.set_xticklabels([f"{r['anahtar']}\n{r['tip']}" for r in sonuclar], fontsize=8)
    ax.set_ylabel(S.EKSEN6["oran"])
    ax.set_title(S.FIG6_BASLIK["retention"])
    ax.legend(fontsize=9)
    fig.tight_layout()
    return _kaydet(fig, S.FIG6_DOSYA["retention"])


def figur_semantic_vs_importance(onem_sonuclar, semantik_sonuclar):
    """A1/A2: semantic vs importance-based retention ratio (threshold band + trivial/ref marker)."""
    anahtarlar = [r["anahtar"] for r in semantik_sonuclar]
    onem = {r["anahtar"]: r for r in onem_sonuclar}
    x = np.arange(len(anahtarlar))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7.5, 5))
    o_oran = [onem[a]["oran"] for a in anahtarlar]
    s_oran = [r["oran"] for r in semantik_sonuclar]
    ax.bar(x - w / 2, o_oran, w, label="Importance-based mapping", color="#7F7F7F")
    ax.bar(x + w / 2, s_oran, w, label="Semantic mapping", color=ps.CHURN_RENK[1])
    ax.axhspan(0.70, 0.80, color="green", alpha=0.12, label="Threshold band (70–80%)")
    for i, r in enumerate(semantik_sonuclar):
        tr_oran = r["trivial"] / r["ref"] if r["ref"] > 0 else 0.0
        ax.plot([i - 0.7 * w, i + 0.7 * w], [tr_oran, tr_oran], color="black",
                linestyle=":", linewidth=1.3)
    ax.plot([], [], color="black", linestyle=":", label="Trivial / reference")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{a}\n{onem[a]['ad']}" for a in anahtarlar], fontsize=8)
    ax.set_ylabel(S.EKSEN6["oran"])
    ax.set_title(S.FIG6_BASLIK["semantic"])
    ax.legend(fontsize=9)
    fig.tight_layout()
    return _kaydet(fig, S.FIG6_DOSYA["semantic"])
