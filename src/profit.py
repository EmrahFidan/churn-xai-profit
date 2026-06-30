"""ADIM 5 — RQ3: Maliyete duyarlı eşik, kâr/ROI ve EMP.

Açıklanan/operasyonel model: HAM/doğal-dağılım LightGBM (Adım 2 best_params),
kalibrasyon katmanı YOK (ham olasılık zaten iyi kalibre; kâr doğru olasılık ister).
Olasılıklar 5-kat out-of-fold üretilir (model gördüğü müşteriyi puanlamaz — sızıntı yok).
Maliyet/değer için sihirli sayı yok: parametrik (c, γ, CLV) + duyarlılık.

Kâr matrisi (müşteri başı, eşik t, p=ham olasılık):
  TP (tahmin churn & gerçek churn): γ·CLV − c
  FP (tahmin churn & gerçek kalır):        − c
  FN (tahmin kalır & gerçek churn):        − CLV
  TN:                                         0
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.stats import beta as beta_dist
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from . import config as cfg
from . import encode
from . import evaluate as ev
from . import plotstyle as ps
from . import strings_tr as S

PARAM = cfg.CFG["profit"]


def _lgbm_params(set_adi):
    bp = json.loads((cfg.TABLES / f"best_params_{set_adi}.json").read_text(encoding="utf-8")).get("lightgbm", {})
    return {k.replace("model__", ""): v for k, v in bp.items()}


# ----------------------------- CLV -----------------------------
def clv_hesapla(set_adi, df):
    """Müşteri-başı CLV (uniform değil) + temel açıklaması. Negatif/0 ele alınır."""
    u = PARAM["ufuk_ay"]
    if set_adi == "telco":
        clv = df["MonthlyCharges"].astype(float) * u
        temel = f"MonthlyCharges × {u} ay"
    elif set_adi == "cell2cell":
        clv = df["MonthlyRevenue"].astype(float) * u
        temel = f"MonthlyRevenue × {u} ay"
    elif set_adi == "bank":
        m = PARAM["banka_marj_yillik"]
        clv = df["Balance"].astype(float) * m * (u / 12.0)
        temel = f"Balance × {m} yıllık marj × {u/12:.1f} yıl"
    elif set_adi == "ecommerce":
        clv = df["CashbackAmount"].astype(float) * u
        temel = f"CashbackAmount (aylık değer proxy) × {u} ay"
    elif set_adi == "iranian":
        clv = df["Customer Value"].astype(float)
        temel = "Customer Value (doğrudan)"
    else:
        raise ValueError(set_adi)
    clv = clv.clip(lower=0).to_numpy()
    return clv, temel


def oof_olasilik(set_adi, df, seed):
    """5-kat out-of-fold ham olasılık (sızıntısız). Dönüş: (p_oof, y)."""
    X = df.drop(columns=["churn"])
    y = df["churn"].to_numpy()
    pre = encode.on_isleyici(set_adi, df, olcekle=False)[0]
    params = _lgbm_params(set_adi)
    p = np.zeros(len(y))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, va in skf.split(X, y):
        pipe = Pipeline([("pre", clone(pre)),
                         ("model", LGBMClassifier(random_state=seed, n_jobs=-1, verbose=-1, **params))])
        pipe.fit(X.iloc[tr], y[tr])
        p[va] = pipe.predict_proba(X.iloc[va])[:, 1]
    return p, y


# ----------------------------- kâr / eşik / ROI -----------------------------
def kar(p, y, clv, c, gamma, t):
    """Eşik t'de toplam kâr."""
    pred = p >= t
    tp = pred & (y == 1)
    fn = (~pred) & (y == 1)
    return float(gamma * clv[tp].sum() - c * pred.sum() - clv[fn].sum())


def kar_egrisi(p, y, clv, c, gamma, esikler):
    return np.array([kar(p, y, clv, c, gamma, t) for t in esikler])


def en_iyi_esik(p, y, clv, c, gamma, esikler):
    egri = kar_egrisi(p, y, clv, c, gamma, esikler)
    i = int(np.argmax(egri))
    return float(esikler[i]), float(egri[i])


def roi(p, y, clv, c, gamma, t):
    """ROI = net kâr / toplam müdahale maliyeti (hedeflenen × c)."""
    pred = p >= t
    maliyet = c * pred.sum()
    k = kar(p, y, clv, c, gamma, t)
    return (k / maliyet) if maliyet > 0 else 0.0


def naif(p, y, clv, c, gamma):
    """Referans: herkese müdahale (t=0) ve hiç müdahale (t>1)."""
    hepsi = kar(p, y, clv, c, gamma, 0.0)
    hic = kar(p, y, clv, c, gamma, 1.1)
    return hepsi, hic


# ----------------------------- EMP -----------------------------
def emp(p, y, clv, c_ref, seed=None):
    """Expected Maximum Profit for churn: γ ~ Beta(α,β) üstünde max kâr/kişi entegrali.

    Setler-arası kıyas için CLV ortalamaya normalize edilir (EMP = ortalama CLV'nin
    kesri olarak kişi-başı beklenen maksimum kâr). Her γ için kâr-maksimize eşikteki
    kişi-başı kâr, Beta yoğunluğuyla ağırlıklandırılır. Varsayım MSG5'te loglanır.
    """
    a, b = PARAM["emp_beta"]
    ort = clv.mean() or 1.0
    clv_n = clv / ort           # ortalama CLV = 1 birim
    c_n = c_ref / ort           # maliyet de aynı birimde
    esikler = np.linspace(0.0, 1.0, 101)
    gamalar = np.linspace(0.005, 0.995, 100)
    w = beta_dist.pdf(gamalar, a, b)
    w = w / w.sum()
    N = len(y)
    toplam = 0.0
    for g, wi in zip(gamalar, w):
        egri = kar_egrisi(p, y, clv_n, c_n, g, esikler)
        toplam += wi * (egri.max() / N)
    return float(toplam)


# ----------------------------- orkestrasyon -----------------------------
def calistir_set(set_adi, df, seed):
    """Bir set için OOF olasılık + CLV + (c,γ) taraması + EMP. Dönüş: dict."""
    p, y = oof_olasilik(set_adi, df, seed)
    clv, temel = clv_hesapla(set_adi, df)
    ort = float(np.mean(clv))
    esikler = np.linspace(0.0, 1.0, 101)
    satirlar = []
    for oran in PARAM["c_oranlari"]:
        c = ort * oran
        for g in PARAM["gamma_listesi"]:
            tb, kb = en_iyi_esik(p, y, clv, c, g, esikler)
            ka = kar(p, y, clv, c, g, 0.5)
            ra = roi(p, y, clv, c, g, 0.5)
            rb = roi(p, y, clv, c, g, tb)
            satirlar.append({
                "oran": oran, "c": c, "gamma": g,
                "esik_a": 0.5, "esik_b": tb, "kar_a": ka, "kar_b": kb,
                "roi_a": ra, "roi_b": rb,
            })
    e = emp(p, y, clv, ort * PARAM["emp_c_oran"], seed)
    return {"p": p, "y": y, "clv": clv, "temel": temel, "ort_clv": ort,
            "medyan_clv": float(np.median(clv)), "sifir": int((clv == 0).sum()),
            "satirlar": satirlar, "emp": e, "esikler": esikler}


# ----------------------------- tablolar -----------------------------
def tablo_ozet(tum):
    K = S.KOLON5
    rows = []
    for s, r in tum.items():
        for d in r["satirlar"]:
            ka, kb = d["kar_a"], d["kar_b"]
            rows.append({
                K["veri_seti"]: s, K["c_oran"]: d["oran"], K["gamma"]: d["gamma"],
                K["esik_a"]: round(d["esik_a"], 3), K["esik_b"]: round(d["esik_b"], 3),
                K["kar_a"]: round(ka, 1), K["kar_b"]: round(kb, 1),
                K["roi_a"]: round(d["roi_a"], 3), K["roi_b"]: round(d["roi_b"], 3),
                K["roi_artis"]: round(d["roi_b"] - d["roi_a"], 3),
                K["kar_artis_yuzde"]: round(100 * (kb - ka) / abs(ka), 1) if ka != 0 else np.nan,
                K["emp"]: round(r["emp"], 4),
            })
    df = pd.DataFrame(rows)
    df.to_csv(cfg.TABLES / "rq3_profit_summary.csv", index=False)
    return df


def tablo_clv(tum):
    K = S.KOLON5
    rows = [{K["veri_seti"]: s, K["clv_temeli"]: r["temel"],
             K["ort_clv"]: round(r["ort_clv"], 2), K["medyan_clv"]: round(r["medyan_clv"], 2),
             K["sifir_clv"]: r["sifir"]} for s, r in tum.items()]
    df = pd.DataFrame(rows)
    df.to_csv(cfg.TABLES / "rq3_clv_basis.csv", index=False)
    return df


# ----------------------------- figürler -----------------------------
def _temsili(r):
    """Temsili (c,γ): c=%5 ort_CLV, γ=0.3 (yoksa ilk)."""
    oran = 0.05 if 0.05 in PARAM["c_oranlari"] else PARAM["c_oranlari"][0]
    g = 0.3 if 0.3 in PARAM["gamma_listesi"] else PARAM["gamma_listesi"][0]
    return r["ort_clv"] * oran, g, oran


def figur_profit_threshold(set_adi, r):
    c, g, oran = _temsili(r)
    egri = kar_egrisi(r["p"], r["y"], r["clv"], c, g, r["esikler"])
    tb = float(r["esikler"][int(np.argmax(egri))])
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.plot(r["esikler"], egri, color=ps.CHURN_RENK[0], label="Kâr eğrisi")
    ax.axvline(tb, color=ps.CHURN_RENK[1], linestyle="-", label=f"Kâr-eşiği t*={tb:.2f}")
    ax.axvline(0.5, color="#7F7F7F", linestyle="--", label="Doğruluk-eşiği t=0.5")
    ax.set_xlabel(S.EKSEN5["esik"])
    ax.set_ylabel(S.EKSEN5["kar"])
    ax.set_title(S.FIG5_BASLIK["profit_threshold"].format(c=int(oran * 100), g=g, set=set_adi))
    ax.legend(fontsize=9)
    fig.tight_layout()
    return ev._kaydet(fig, set_adi, S.FIG5_DOSYA["profit_threshold"])


def figur_roi_sensitivity(set_adi, r):
    oranlar = PARAM["c_oranlari"]
    gamalar = PARAM["gamma_listesi"]
    M = np.zeros((len(oranlar), len(gamalar)))
    for i, oran in enumerate(oranlar):
        c = r["ort_clv"] * oran
        for j, g in enumerate(gamalar):
            tb, _ = en_iyi_esik(r["p"], r["y"], r["clv"], c, g, r["esikler"])
            M[i, j] = roi(r["p"], r["y"], r["clv"], c, g, tb)
    fig, ax = plt.subplots(figsize=(6, 4.8))
    im = ax.imshow(M, cmap="YlOrRd", aspect="auto", origin="lower")
    ax.set_xticks(range(len(gamalar))); ax.set_xticklabels(gamalar)
    ax.set_yticks(range(len(oranlar))); ax.set_yticklabels([f"%{int(o*100)}" for o in oranlar])
    for i in range(len(oranlar)):
        for j in range(len(gamalar)):
            ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=9,
                    color="black" if M[i, j] < M.max() * 0.6 else "white")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=S.EKSEN5["roi"])
    ax.set_xlabel(S.EKSEN5["gamma"]); ax.set_ylabel(S.EKSEN5["maliyet_oran"])
    ax.set_title(S.FIG5_BASLIK["roi_sensitivity"].format(set=set_adi))
    fig.tight_layout()
    return ev._kaydet(fig, set_adi, S.FIG5_DOSYA["roi_sensitivity"])


def figur_strategy(set_adi, r):
    c, g, oran = _temsili(r)
    tb, kb = en_iyi_esik(r["p"], r["y"], r["clv"], c, g, r["esikler"])
    ka = kar(r["p"], r["y"], r["clv"], c, g, 0.5)
    hepsi, hic = naif(r["p"], r["y"], r["clv"], c, g)
    etiket = [S.STRATEJI_AD["A"], S.STRATEJI_AD["B"], S.STRATEJI_AD["hepsi"], S.STRATEJI_AD["hic"]]
    deger = [ka, kb, hepsi, hic]
    renk = ["#7F7F7F", ps.CHURN_RENK[1], "#55A868", "#C44E52"]
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.bar(etiket, deger, color=renk)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel(S.EKSEN5["kar"])
    ax.set_title(S.FIG5_BASLIK["strategy"].format(c=int(oran * 100), g=g, set=set_adi))
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    return ev._kaydet(fig, set_adi, S.FIG5_DOSYA["strategy"])


def figur_emp(tum):
    setler = list(tum.keys())
    deger = [tum[s]["emp"] for s in setler]
    sira = np.argsort(deger)[::-1]
    setler = [setler[i] for i in sira]; deger = [deger[i] for i in sira]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(setler, deger, color=ps.CHURN_RENK[0])
    ax.set_ylabel(S.EKSEN5["kar_kisi"])
    ax.set_title(S.FIG5_BASLIK["emp"])
    for i, v in enumerate(deger):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    d = cfg.FIGURES / "_rq3"
    d.mkdir(parents=True, exist_ok=True)
    yol = d / S.FIG5_DOSYA["emp"]
    fig.savefig(yol)
    return yol


def figurler_set(set_adi, r):
    return {
        "profit_threshold": figur_profit_threshold(set_adi, r),
        "roi_sensitivity": figur_roi_sensitivity(set_adi, r),
        "strategy": figur_strategy(set_adi, r),
    }
