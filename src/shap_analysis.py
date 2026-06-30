"""ADIM 4 — RQ2: TreeSHAP açıklanabilirlik + setler-arası sürücü tutarlılığı.

Açıklanan model: HAM/doğal-dağılım LightGBM (dengeleme YOK, kalibrasyon YOK), Adım 2
best_params ile TÜM veri üstünde fit (seed=42). shap.TreeExplainer ile exact TreeSHAP.
Encoded kolonların önemi ham feature'a toplanır; kavram haritasıyla sektörler arası
karşılaştırılır (veri birleştirilmez).
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline

from . import concept_map as km
from . import config as cfg
from . import encode
from . import evaluate as ev
from . import plotstyle as ps
from . import strings_tr as S

ORNEKLEM_SINIR = 15000  # bu satırdan büyük setlerde SHAP için stratified örneklem


def _lgbm_params(set_adi):
    bp = json.loads((cfg.TABLES / f"best_params_{set_adi}.json").read_text(encoding="utf-8")).get("lightgbm", {})
    return {k.replace("model__", ""): v for k, v in bp.items()}


def _orijinal(name, nominal):
    """Encoded kolon adını ham feature'a indirger ('num__x'->'x', 'nom__c_kat'->'c')."""
    if name.startswith("num__"):
        return name[5:]
    if name.startswith("nom__"):
        rest = name[5:]
        for c in sorted(nominal, key=len, reverse=True):
            if rest.startswith(c + "_"):
                return c
        return rest
    return name


def _etiket(name, nominal):
    """Okunur gösterim etiketi (ham token korunur)."""
    if name.startswith("num__"):
        return name[5:]
    if name.startswith("nom__"):
        rest = name[5:]
        for c in sorted(nominal, key=len, reverse=True):
            if rest.startswith(c + "_"):
                return f"{c} = {rest[len(c)+1:]}"
        return rest
    return name


def hazirla(set_adi, df, seed):
    """Model + encode'u TÜM veride fit eder, SHAP için (örneklenmiş) matris döndürür.

    Dönüş: dict {model, Z, isimler, etiketler, orijinaller, nominal, proba, X_raw_ornek,
    n_ornek, N}.
    """
    X = df.drop(columns=["churn"])
    y = df["churn"].to_numpy()
    parts = encode.parcalar(set_adi, df, olcekle=False)

    on = []
    if parts["prep"] is not None:
        on.append(("hazirla", parts["prep"]))
    on.append(("ct", parts["ct"]))
    on_pipe = Pipeline(on)
    Z_full = on_pipe.fit_transform(X)
    isimler = list(on_pipe.named_steps["ct"].get_feature_names_out())

    model = LGBMClassifier(random_state=seed, n_jobs=-1, verbose=-1, **_lgbm_params(set_adi))
    model.fit(Z_full, y)

    N = len(y)
    if N > ORNEKLEM_SINIR:
        rng = np.random.RandomState(seed)
        # stratified örneklem
        idx = np.hstack([
            rng.choice(np.where(y == k)[0],
                       size=int(round(ORNEKLEM_SINIR * (y == k).mean())), replace=False)
            for k in (0, 1)
        ])
        rng.shuffle(idx)
    else:
        idx = np.arange(N)
    Z = Z_full[idx]
    proba = model.predict_proba(Z)[:, 1]

    nominal = parts["nominal"]
    etiketler = [_etiket(n, nominal) for n in isimler]
    orijinaller = [_orijinal(n, nominal) for n in isimler]
    return {"model": model, "Z": Z, "isimler": isimler, "etiketler": etiketler,
            "orijinaller": orijinaller, "nominal": nominal, "proba": proba,
            "X_raw_ornek": X.iloc[idx].reset_index(drop=True), "n_ornek": len(idx), "N": N}


def shap_hesap(h):
    """TreeSHAP değerleri + beklenen değer (pozitif sınıf)."""
    explainer = shap.TreeExplainer(h["model"])
    sv = explainer.shap_values(h["Z"])
    if isinstance(sv, list):
        sv = sv[1]
    elif getattr(sv, "ndim", 2) == 3:
        sv = sv[:, :, 1]
    ev0 = explainer.expected_value
    if np.ndim(ev0) > 0:
        ev = ev0[1] if len(np.atleast_1d(ev0)) > 1 else np.atleast_1d(ev0)[0]
    else:
        ev = float(ev0)
    return np.asarray(sv), float(ev)


# ----------------------------- önem (aggregate) -----------------------------
def onem_orijinal(sv, orijinaller):
    """Encoded kolon mean|SHAP|'larını ham feature'a toplar. Dönüş: Series(sıralı)."""
    mas = np.abs(sv).mean(axis=0)
    s = pd.Series(mas, index=orijinaller).groupby(level=0).sum().sort_values(ascending=False)
    return s


def tablo_global(set_adi, onem):
    K = S.KOLON4
    df = pd.DataFrame({K["feature"]: onem.index, K["mean_abs_shap"]: onem.values.round(6)})
    df[K["sira"]] = np.arange(1, len(df) + 1)
    df[K["kavram"]] = [km.KAVRAM_AD[km.kavram(f)] for f in onem.index]
    df.to_csv(cfg.TABLES / f"rq2_global_importance_{set_adi}.csv", index=False)
    return df


# ----------------------------- global figürler -----------------------------
def figur_beeswarm(set_adi, sv, h):
    plt.figure()
    shap.summary_plot(sv, h["Z"], feature_names=h["etiketler"], max_display=15, show=False)
    fig = plt.gcf()
    fig.set_size_inches(8, 6)
    ax = plt.gca()
    ax.set_xlabel(S.EKSEN4["shap_deger"])
    ax.set_title(S.FIG4_BASLIK["beeswarm"].format(set=set_adi))
    fig.tight_layout()
    yol = ev._kaydet(fig, set_adi, S.FIG4_DOSYA["beeswarm"])
    plt.close(fig)
    return yol


def figur_importance(set_adi, onem):
    top = onem.head(15)[::-1]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(list(top.index), top.values, color=ps.CHURN_RENK[1])
    ax.set_xlabel(S.EKSEN4["ortalama_etki"])
    ax.set_title(S.FIG4_BASLIK["importance"].format(set=set_adi))
    fig.tight_layout()
    return ev._kaydet(fig, set_adi, S.FIG4_DOSYA["importance"])


def figur_dependence(set_adi, sv, h, onem):
    # tek-kolonlu (sayısal) en güçlü ham feature'lar
    tekil = [f for f in onem.index if f"num__{f}" in h["isimler"]][:4]
    n = max(1, len(tekil))
    ncol = 2 if n > 1 else 1
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 4.6, nrow * 3.6), squeeze=False)
    for i, f in enumerate(tekil):
        ax = axes[i // ncol][i % ncol]
        idx = h["isimler"].index(f"num__{f}")
        shap.dependence_plot(idx, sv, h["Z"], feature_names=h["etiketler"],
                             interaction_index="auto", ax=ax, show=False)
        ax.set_title(f, fontsize=10)
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle(S.FIG4_BASLIK["dependence"].format(set=set_adi), y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return ev._kaydet(fig, set_adi, S.FIG4_DOSYA["dependence"])


# ----------------------------- tekil (local) -----------------------------
def figur_waterfall(set_adi, sv, ev_base, h, idx, anahtar):
    expl = shap.Explanation(values=sv[idx], base_values=ev_base,
                            data=h["Z"][idx], feature_names=h["etiketler"])
    plt.figure()
    shap.plots.waterfall(expl, max_display=12, show=False)
    fig = plt.gcf()
    fig.set_size_inches(8, 6)
    plt.title(S.FIG4_BASLIK[f"waterfall_{anahtar}"].format(set=set_adi))
    fig.tight_layout()
    yol = ev._kaydet(fig, set_adi, S.FIG4_DOSYA[f"waterfall_{anahtar}"])
    plt.close(fig)
    return yol


def tablo_local(set_adi, sv, h, hi, lo):
    """Seçilen 2 müşterinin profili (en etkili 6 feature) + olasılık."""
    K = S.KOLON4
    rows = []
    for durum, idx in [("high", hi), ("low", lo)]:
        kat = np.abs(sv[idx])
        top = np.argsort(kat)[::-1][:6]
        for j in top:
            orig = h["orijinaller"][j]
            ham = h["X_raw_ornek"].iloc[idx].get(orig, "")
            rows.append({
                K["durum"]: S.DURUM[durum], K["olasilik"]: round(float(h["proba"][idx]), 4),
                K["feature"]: h["etiketler"][j], K["ham_deger"]: ham,
                K["shap_katki"]: round(float(sv[idx][j]), 4),
            })
    df = pd.DataFrame(rows)
    df.to_csv(cfg.TABLES / f"rq2_local_cases_{set_adi}.csv", index=False)
    return df


# ----------------------------- setler-arası tutarlılık -----------------------------
def kavram_paylari(onem):
    """Ham feature önemini kavrama toplar, pay (0-1) döndürür."""
    pay = {}
    for f, v in onem.items():
        k = km.kavram(f)
        pay[k] = pay.get(k, 0.0) + v
    toplam = sum(pay.values()) or 1.0
    return {k: v / toplam for k, v in pay.items()}


def tutarlilik(tum_onem):
    """rq2_driver_consistency.csv + matris. tum_onem={set: Series}. Dönüş: (df, matris)."""
    setler = list(tum_onem.keys())
    paylar = {s: kavram_paylari(o) for s, o in tum_onem.items()}
    matris = pd.DataFrame(
        {S.KOLON4["kavram"]: [km.KAVRAM_AD[k] for k in km.KAVRAM_SIRA]}
    )
    for s in setler:
        matris[s] = [round(paylar[s].get(k, 0.0), 4) for k in km.KAVRAM_SIRA]
    # her sette top-3 kavram -> tutarlılık sayacı
    top3 = {s: set(pd.Series(paylar[s]).sort_values(ascending=False).head(3).index) for s in setler}
    matris[S.KOLON4["top3_say"]] = [sum(k in top3[s] for s in setler) for k in km.KAVRAM_SIRA]
    matris.to_csv(cfg.TABLES / "rq2_driver_consistency.csv", index=False)
    return matris, paylar


def figur_tutarlilik(matris, setler):
    deger = matris[setler].to_numpy()
    kavramlar = matris[S.KOLON4["kavram"]].tolist()
    fig, ax = plt.subplots(figsize=(1.4 * len(setler) + 3, 0.6 * len(kavramlar) + 2))
    im = ax.imshow(deger, cmap="YlOrRd", aspect="auto", vmin=0)
    ax.set_xticks(range(len(setler)))
    ax.set_xticklabels(setler, rotation=20)
    ax.set_yticks(range(len(kavramlar)))
    ax.set_yticklabels(kavramlar)
    for i in range(len(kavramlar)):
        for j in range(len(setler)):
            ax.text(j, i, f"{deger[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="black" if deger[i, j] < deger.max() * 0.6 else "white")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Önem payı")
    ax.set_title(S.FIG4_BASLIK["consistency"])
    ax.set_xlabel("Veri seti")
    ax.set_ylabel(S.EKSEN4["kavram"])
    fig.tight_layout()
    d = cfg.FIGURES / "_rq2"
    d.mkdir(parents=True, exist_ok=True)
    yol = d / S.FIG4_DOSYA["consistency"]
    fig.savefig(yol)
    return yol
