"""Keşifsel veri analizi (EDA) — set bazında figürler + genel bakış tablosu.

Her etiketli set için outputs/figures/<set>/ altına 5 figür üretir ve
outputs/tables/dataset_overview.csv yazar. churn=0/1 sabit renklerle gösterilir.
Tüm gösterim metinleri strings_tr'den gelir.
"""
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from . import config as cfg
from . import plotstyle as ps
from . import strings_tr as S

KATEGORIK_MAX = 12  # bu eşiğin üstündeki kardinalite kategorik figürde atlanır


def _sayisal_kolonlar(df):
    return [c for c in df.columns if c != "churn" and is_numeric_dtype(df[c])]


def _kategorik_kolonlar(df):
    return [c for c in df.columns if c != "churn" and not is_numeric_dtype(df[c])]


def _baslik(anahtar, set_adi):
    return S.FIG_BASLIK[anahtar].format(set=set_adi)


def churn_dengesi(df, set_adi):
    """churn_balance.png — churn 0/1 sayımları (sabit iki renk)."""
    say = df["churn"].value_counts().reindex([0, 1]).fillna(0).astype(int)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar([S.CHURN_ETIKET[0], S.CHURN_ETIKET[1]], [say[0], say[1]],
           color=[ps.CHURN_RENK[0], ps.CHURN_RENK[1]])
    toplam = int(say.sum())
    for i, v in enumerate([say[0], say[1]]):
        ax.text(i, v, f"{v}\n(%{100*v/toplam:.1f})", ha="center", va="bottom")
    ax.set_title(_baslik("churn_balance", set_adi))
    ax.set_xlabel(S.EKSEN["churn_durumu"])
    ax.set_ylabel(S.EKSEN["musteri_sayisi"])
    ax.margins(y=0.15)
    fig.tight_layout()
    return fig, ps.kaydet(fig, set_adi, "churn_balance")


def sayisal_dagilim(df, set_adi):
    """numeric_distributions.png — sayısal değişkenlerin churn'e göre histogramları."""
    kols = _sayisal_kolonlar(df)
    n = len(kols)
    ncol = min(4, n) if n else 1
    nrow = max(1, math.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 3.4, nrow * 2.6), squeeze=False)
    for i, c in enumerate(kols):
        ax = axes[i // ncol][i % ncol]
        for k in (0, 1):
            ax.hist(df.loc[df["churn"] == k, c].dropna(), bins=30, alpha=0.6,
                    color=ps.CHURN_RENK[k], label=S.CHURN_ETIKET[k])
        ax.set_title(c, fontsize=10)
        ax.set_xlabel(S.EKSEN["deger"])
        ax.set_ylabel(S.EKSEN["musteri_sayisi"])
    for j in range(n, nrow * ncol):  # boş eksenleri gizle
        axes[j // ncol][j % ncol].axis("off")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, title="Churn", loc="upper right")
    fig.suptitle(_baslik("numeric_distributions", set_adi), y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    return fig, ps.kaydet(fig, set_adi, "numeric_distributions")


def kategorik_churn(df, set_adi):
    """categorical_churn.png — düşük kardinaliteli kategoriklerde churn oranı.

    Dönüş: (fig, yol, atlanan_yuksek_kardinalite_listesi).
    """
    kategorik = _kategorik_kolonlar(df)
    kullan = [c for c in kategorik if df[c].nunique() <= KATEGORIK_MAX]
    atlanan = [c for c in kategorik if df[c].nunique() > KATEGORIK_MAX]
    if kullan:
        n = len(kullan)
        ncol = min(3, n)
        nrow = max(1, math.ceil(n / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 4.0, nrow * 3.0), squeeze=False)
        for i, c in enumerate(kullan):
            ax = axes[i // ncol][i % ncol]
            oran = df.groupby(c, observed=True)["churn"].mean().sort_values(ascending=False)
            ax.bar(oran.index.astype(str), oran.values, color=ps.CHURN_RENK[1])
            ax.axhline(df["churn"].mean(), color=ps.CHURN_RENK[0], linestyle="--",
                       label="Genel oran")
            ax.set_title(c, fontsize=10)
            ax.set_xlabel(S.EKSEN["kategori"])
            ax.set_ylabel(S.EKSEN["churn_orani"])
            ax.tick_params(axis="x", rotation=45)
            ax.legend()
        for j in range(n, nrow * ncol):
            axes[j // ncol][j % ncol].axis("off")
    else:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "Düşük kardinaliteli kategorik değişken yok",
                ha="center", va="center")
        ax.axis("off")
    fig.suptitle(_baslik("categorical_churn", set_adi), y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig, ps.kaydet(fig, set_adi, "categorical_churn"), atlanan


def korelasyon(df, set_adi):
    """correlation_heatmap.png — sayısal değişken + churn korelasyon ısı haritası."""
    kols = _sayisal_kolonlar(df) + ["churn"]
    corr = df[kols].corr(numeric_only=True)
    boyut = max(5, 0.45 * len(kols))
    fig, ax = plt.subplots(figsize=(boyut, boyut * 0.85))
    annot = len(kols) <= 16
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(kols)))
    ax.set_xticklabels(kols, rotation=90, fontsize=8)
    ax.set_yticks(range(len(kols)))
    ax.set_yticklabels(kols, fontsize=8)
    if annot:
        for i in range(len(kols)):
            for j in range(len(kols)):
                ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                        fontsize=7, color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(_baslik("correlation_heatmap", set_adi))
    fig.tight_layout()
    return fig, ps.kaydet(fig, set_adi, "correlation_heatmap")


def eksiklik(df_ham_std, set_adi):
    """missingness.png — temizlik öncesi kolon bazında eksik hücre sayısı."""
    eksik = df_ham_std.isna().sum()
    eksik = eksik[eksik > 0].sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, max(3, 0.35 * max(1, len(eksik)))))
    if len(eksik):
        ax.barh(eksik.index.astype(str)[::-1], eksik.values[::-1], color=ps.CHURN_RENK[0])
        ax.set_xlabel(S.EKSEN["eksik_sayisi"])
    else:
        ax.text(0.5, 0.5, "Eksik veri yok", ha="center", va="center")
        ax.axis("off")
    ax.set_title(_baslik("missingness", set_adi))
    fig.tight_layout()
    return fig, ps.kaydet(fig, set_adi, "missingness")


def figurler_set(set_adi, df_temiz, df_ham_std):
    """Bir set için 5 figürü üretir + kaydeder. Dönüş: {anahtar: yol} ve atlananlar."""
    yollar = {}
    _, yollar["churn_balance"] = churn_dengesi(df_temiz, set_adi)
    _, yollar["numeric_distributions"] = sayisal_dagilim(df_temiz, set_adi)
    _, yollar["categorical_churn"], atlanan = kategorik_churn(df_temiz, set_adi)
    _, yollar["correlation_heatmap"] = korelasyon(df_temiz, set_adi)
    _, yollar["missingness"] = eksiklik(df_ham_std, set_adi)
    return yollar, atlanan


def genel_bakis(etiketli, processed, holdout):
    """dataset_overview.csv — 5 set yan yana + cell2cell_test etiketsiz notu."""
    cfg.klasorleri_hazirla()
    satirlar = []
    for key in cfg.DATASETS:
        ham_std = etiketli[key]["df"]
        temiz = processed[key]
        satirlar.append({
            S.KOLON["veri_seti"]: key,
            S.KOLON["sektor"]: etiketli[key]["sector"],
            S.KOLON["satir"]: etiketli[key]["raw_shape"][0],
            S.KOLON["sutun"]: etiketli[key]["raw_shape"][1],
            S.KOLON["churn_yuzde"]: round(100 * temiz["churn"].mean(), 1),
            S.KOLON["eksik_once"]: int(ham_std.drop(columns=["churn"]).isna().sum().sum()),
            S.KOLON["eksik_sonra"]: int(temiz.drop(columns=["churn"]).isna().sum().sum()),
            S.KOLON["not"]: "",
        })
    # etiketsiz holdout notu
    satirlar.append({
        S.KOLON["veri_seti"]: "cell2cell_test",
        S.KOLON["sektor"]: cfg.DATASETS["cell2cell"]["sector"],
        S.KOLON["satir"]: holdout["raw_shape"][0],
        S.KOLON["sutun"]: holdout["raw_shape"][1],
        S.KOLON["churn_yuzde"]: np.nan,
        S.KOLON["eksik_once"]: int(holdout["df"].drop(columns=["churn"]).isna().sum().sum()),
        S.KOLON["eksik_sonra"]: int(processed["cell2cell_test"].drop(columns=["churn"]).isna().sum().sum()),
        S.KOLON["not"]: "Etiketsiz holdout — churn'e dayalı analizlere dahil edilmedi",
    })
    df = pd.DataFrame(satirlar)
    df.to_csv(cfg.TABLES / "dataset_overview.csv", index=False)
    return df
