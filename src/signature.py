"""STEP 9 — Signature finding: predictability (PR-AUC) ≠ profitability (EMP).

No NEW experiment. Uses Step 2 PR-AUC (model_performance.csv, LightGBM) + Step 5 EMP
(rq3_profit_summary.csv) + churner CLV distribution. Hypothesis: in the high-PR-AUC
set (iranian) the churner value is narrow/low → low EMP; in the low-PR-AUC set (telco)
it is wide/high → high EMP. n=5 is small → trend/illustration (no over-claiming).
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from . import config as cfg
from . import plotstyle as ps
from . import profit as pr
from . import strings as S


def gini(x):
    x = np.sort(np.asarray(x, float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    cum = np.cumsum(x)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def _prauc_lightgbm():
    """LightGBM PR-AUC (mean) from model_performance.csv — set -> float."""
    df = pd.read_csv(cfg.TABLES / "model_performance.csv")
    K = S.KOLON2
    alt = df[df[K["model"]] == "LightGBM"]
    return {r[K["veri_seti"]]: float(str(r[K["pr_auc"]]).split("±")[0]) for _, r in alt.iterrows()}


def _emp():
    """EMP from rq3_profit_summary.csv (constant per set) — set -> float."""
    df = pd.read_csv(cfg.TABLES / "rq3_profit_summary.csv")
    K = S.KOLON5
    return {s: float(g[K["emp"]].iloc[0]) for s, g in df.groupby(K["veri_seti"])}


def kanit_tablosu(veriler):
    """Produces signature_evidence.csv + returns a DataFrame."""
    prauc, emp = _prauc_lightgbm(), _emp()
    K = S.KOLON9
    rows = []
    churner_clv = {}
    for s in cfg.DATASETS:
        clv, _ = pr.clv_hesapla(s, veriler[s])
        y = veriler[s]["churn"].to_numpy()
        cc = clv[y == 1]
        churner_clv[s] = cc
        rows.append({
            K["veri_seti"]: s, K["pr_auc"]: round(prauc[s], 4), K["emp"]: round(emp[s], 4),
            K["churner_medyan"]: round(float(np.median(cc)), 1),
            K["churner_gini"]: round(gini(cc), 3),
            K["churner_cv"]: round(float(np.std(cc) / np.mean(cc)) if np.mean(cc) else 0.0, 3),
            K["churn_orani"]: round(float(y.mean()), 3),
        })
    df = pd.DataFrame(rows)
    df.to_csv(cfg.TABLES / "signature_evidence.csv", index=False)
    return df, churner_clv, prauc, emp


def korelasyon(prauc, emp):
    setler = list(cfg.DATASETS)
    x = np.array([prauc[s] for s in setler])
    y = np.array([emp[s] for s in setler])
    rho, pr_ = spearmanr(x, y)
    r, pp = pearsonr(x, y)
    return float(rho), float(pr_), float(r), float(pp)


def figur_scatter(prauc, emp):
    setler = list(cfg.DATASETS)
    x = np.array([prauc[s] for s in setler])
    y = np.array([emp[s] for s in setler])
    rho, pr_, r, pp = korelasyon(prauc, emp)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(x, y, s=90, color=ps.CHURN_RENK[1], zorder=5)
    for s, xi, yi in zip(setler, x, y):
        ax.annotate(s, (xi, yi), textcoords="offset points", xytext=(6, 6), fontsize=10)
    b, a = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 20)
    ax.plot(xs, b * xs + a, color=ps.CHURN_RENK[0], linestyle="--",
            label=f"trend (Spearman ρ={rho:.2f})")
    ax.set_xlabel(S.EKSEN9["prauc"])
    ax.set_ylabel(S.EKSEN9["emp"])
    ax.set_title(S.FIG9_BASLIK["scatter"])
    ax.legend(fontsize=9)
    fig.tight_layout()
    d = cfg.FIGURES / "_signature"
    d.mkdir(parents=True, exist_ok=True)
    yol = d / S.FIG9_DOSYA["scatter"]
    fig.savefig(yol)
    return yol


def figur_dagilim(churner_clv):
    setler = list(cfg.DATASETS)
    fig, axes = plt.subplots(1, len(setler), figsize=(3.2 * len(setler), 3.4), squeeze=False)
    for ax, s in zip(axes[0], setler):
        cc = churner_clv[s]
        ax.hist(cc, bins=30, color=ps.CHURN_RENK[1], alpha=0.85)
        ax.axvline(np.median(cc), color=ps.CHURN_RENK[0], linestyle="--",
                   label=f"median={np.median(cc):.0f}")
        ax.set_title(f"{s}\nGini={gini(cc):.2f}", fontsize=10)
        ax.set_xlabel(S.EKSEN9["clv"])
        ax.set_ylabel(S.EKSEN9["sayi"])
        ax.legend(fontsize=8)
    fig.suptitle(S.FIG9_BASLIK["dagilim"])
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    d = cfg.FIGURES / "_signature"
    d.mkdir(parents=True, exist_ok=True)
    yol = d / S.FIG9_DOSYA["dagilim"]
    fig.savefig(yol)
    return yol
