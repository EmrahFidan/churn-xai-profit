"""STEP 8 — Profit-oriented baseline: ProfLogit vs our approach.

Closes the objection 'if you are profit-oriented, why didn't you compare against a
profit-oriented trained model?'.
Two approaches (5 sets, stratified 5-fold, seed=42, same CLV/profit parameters, same encode+scale):
  (A) Ours     : raw LightGBM (Step 2 best_params) + profit-maximizing threshold t*.
  (B) ProfLogit: model that directly EMPC-maximizes the logistic coefficients.

ProfLogit method scientific citation: Stripling, vanden Broucke, Antonio, Baesens, Snoeck (2018),
"Profit maximizing logistic model for customer churn prediction using genetic algorithms."
EMPC objective function + real-coded genetic coefficient search (warm-start: logistic solution).
All fit/threshold/optimization is ONLY within-fold (no leakage).
"""
import json
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.stats import beta as beta_dist
from scipy.stats import wilcoxon
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import StratifiedKFold

from . import config as cfg
from . import encode
from . import evaluate as ev
from . import plotstyle as ps
from . import profit as pr
from . import strings as S

GAMMA = 0.30
C_ORAN = cfg.CFG["profit"]["emp_c_oran"]
ORNEKLEM = 15000            # ProfLogit speed subsample for cell2cell
POP, GEN, SABIR = 40, 80, 15   # genetic: population, generations, early-stopping patience


def _lgbm_params(set_adi):
    bp = json.loads((cfg.TABLES / f"best_params_{set_adi}.json").read_text(encoding="utf-8")).get("lightgbm", {})
    return {k.replace("model__", ""): v for k, v in bp.items()}


# ----------------------------- fast EMPC (GA fitness) -----------------------------
def _empc_hizli(scores, clv_y, c, gplus1, w):
    """EMPC by score order (const-free; for GA ranking). O(n log n)."""
    order = np.argsort(-scores)
    cc = np.cumsum(clv_y[order])            # top-m churner value
    k = np.arange(1, len(scores) + 1)
    base = gplus1[:, None] * cc[None, :] - c * k[None, :]   # (G, n)
    perg = np.maximum(base.max(axis=1), 0.0)
    return float((w * perg).sum())


class ProfLogit:
    """EMPC-maximizing logistic (real-coded GA, warm-start = MLE logistic)."""

    def __init__(self, c, seed, pop=POP, gen=GEN, sabir=SABIR):
        self.c, self.seed, self.pop, self.gen, self.sabir = c, seed, pop, gen, sabir

    def fit(self, Z, y, clv):
        rng = np.random.RandomState(self.seed)
        n, d = Z.shape
        Zb = np.hstack([np.ones((n, 1)), Z])          # bias column
        clv_y = clv * y
        a, b = cfg.CFG["profit"]["emp_beta"]
        gamalar = np.linspace(0.005, 0.995, 60)
        w = beta_dist.pdf(gamalar, a, b); w = w / w.sum()
        gplus1 = gamalar + 1.0

        # warm-start: MLE logistic coefficients (anchor); GA only nudges toward profit,
        # deviation from the warm-start is penalized (overfitting/direction distortion is prevented).
        lr = LogisticRegression(max_iter=1000, solver="liblinear").fit(Z, y)
        taban = np.concatenate([lr.intercept_, lr.coef_.ravel()])
        base = _empc_hizli(Zb @ taban, clv_y, self.c, gplus1, w)
        lam = 0.05 * abs(base)                         # deviation (anchor) penalty coefficient

        def uygunluk(theta):
            return _empc_hizli(Zb @ theta, clv_y, self.c, gplus1, w) - lam * np.sum((theta - taban) ** 2)

        pop = [taban] + [taban + rng.normal(0, 0.2, d + 1) for _ in range(self.pop - 1)]
        pop = np.array(pop)
        fit = np.array([uygunluk(t) for t in pop])
        en_iyi, en_iyi_fit, durgun = pop[fit.argmax()].copy(), fit.max(), 0

        for _ in range(self.gen):
            yeni = [en_iyi.copy()]                     # elitism
            while len(yeni) < self.pop:
                # tournament selection + convex blend (stay around the warm-start)
                i, j = rng.randint(self.pop), rng.randint(self.pop)
                p1 = pop[i] if fit[i] > fit[j] else pop[j]
                i, j = rng.randint(self.pop), rng.randint(self.pop)
                p2 = pop[i] if fit[i] > fit[j] else pop[j]
                alpha = rng.rand(d + 1)
                cocuk = alpha * p1 + (1 - alpha) * p2
                mask = rng.rand(d + 1) < 0.1              # mutation
                cocuk[mask] += rng.normal(0, 0.1, mask.sum())
                yeni.append(cocuk)
            pop = np.array(yeni)
            fit = np.array([uygunluk(t) for t in pop])
            if fit.max() > en_iyi_fit + 1e-9:
                en_iyi, en_iyi_fit, durgun = pop[fit.argmax()].copy(), fit.max(), 0
            else:
                durgun += 1
                if durgun >= self.sabir:
                    break
        self.theta_ = en_iyi
        return self

    def predict_proba(self, Z):
        z = np.hstack([np.ones((len(Z), 1)), Z]) @ self.theta_
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        return np.column_stack([1 - p, p])


# ----------------------------- protocol -----------------------------
def _ornekle(df, seed):
    if len(df) <= ORNEKLEM:
        return df, len(df), len(df)
    rng = np.random.RandomState(seed)
    y = df["churn"].to_numpy()
    idx = np.hstack([rng.choice(np.where(y == k)[0], int(round(ORNEKLEM * (y == k).mean())), replace=False)
                     for k in (0, 1)])
    return df.iloc[np.sort(idx)].reset_index(drop=True), len(idx), len(df)


def calistir_set(set_adi, df, seed):
    """5-fold: in each fold (A) LGBM+profit-threshold and (B) ProfLogit. Returns: per-fold metrics + n."""
    df_s, n, N = _ornekle(df, seed)
    X = df_s.drop(columns=["churn"])
    y = df_s["churn"].to_numpy()
    clv, _ = pr.clv_hesapla(set_adi, df_s)
    c = float(np.mean(clv)) * C_ORAN
    esikler = np.linspace(0.0, 1.0, 101)
    lp = _lgbm_params(set_adi)

    pre_ham = encode.on_isleyici(set_adi, df_s, olcekle=False)[0]
    pre_olc = encode.on_isleyici(set_adi, df_s, olcekle=True)[0]

    met = {y: {m: [] for m in ["EMPC", "kar", "roi", "PR-AUC", "recall", "precision", "F1"]}
           for y in ("ours", "proflogit")}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, va in skf.split(X, y):
        ytr, yva = y[tr], y[va]
        clv_va = clv[va]
        # (A) ours
        pipe = clone(pre_ham)
        Ztr = pipe.fit_transform(X.iloc[tr])
        Zva = pipe.transform(X.iloc[va])
        lgbm = LGBMClassifier(random_state=seed, n_jobs=-1, verbose=-1, **lp).fit(Ztr, ytr)
        p_a = lgbm.predict_proba(Zva)[:, 1]
        # (B) proflogit (scaled input)
        pipe2 = clone(pre_olc)
        Ztr2 = np.asarray(pipe2.fit_transform(X.iloc[tr]))
        Zva2 = np.asarray(pipe2.transform(X.iloc[va]))
        pl = ProfLogit(c, seed).fit(Ztr2, ytr, clv[tr])
        p_b = pl.predict_proba(Zva2)[:, 1]
        for adi, p in (("ours", p_a), ("proflogit", p_b)):
            t, _ = pr.en_iyi_esik(p, yva, clv_va, c, GAMMA, esikler)
            pred = (p >= t).astype(int)
            met[adi]["EMPC"].append(pr.emp(p, yva, clv_va, c))
            met[adi]["kar"].append(pr.kar(p, yva, clv_va, c, GAMMA, t))
            met[adi]["roi"].append(pr.roi(p, yva, clv_va, c, GAMMA, t))
            met[adi]["PR-AUC"].append(average_precision_score(yva, p))
            met[adi]["recall"].append(recall_score(yva, pred, zero_division=0))
            met[adi]["precision"].append(precision_score(yva, pred, zero_division=0))
            met[adi]["F1"].append(f1_score(yva, pred, zero_division=0))
    return met, n, N


# ----------------------------- table + figure -----------------------------
def tablo(tum):
    K = S.KOLON8
    rows = []
    for s, (met, _, _) in tum.items():
        pe = wilcoxon(met["ours"]["EMPC"], met["proflogit"]["EMPC"]).pvalue if \
            np.ptp(np.array(met["ours"]["EMPC"]) - np.array(met["proflogit"]["EMPC"])) > 0 else 1.0
        pk = wilcoxon(met["ours"]["kar"], met["proflogit"]["kar"]).pvalue if \
            np.ptp(np.array(met["ours"]["kar"]) - np.array(met["proflogit"]["kar"])) > 0 else 1.0
        for adi in ("ours", "proflogit"):
            m = met[adi]
            rows.append({
                K["veri_seti"]: s, K["yaklasim"]: S.YAKLASIM_AD[adi],
                K["empc"]: f"{np.mean(m['EMPC']):.4f} ± {np.std(m['EMPC']):.4f}",
                K["kar"]: f"{np.mean(m['kar']):.0f} ± {np.std(m['kar']):.0f}",
                K["roi"]: round(float(np.mean(m["roi"])), 3),
                K["pr_auc"]: f"{np.mean(m['PR-AUC']):.4f} ± {np.std(m['PR-AUC']):.4f}",
                K["recall"]: round(float(np.mean(m["recall"])), 3),
                K["precision"]: round(float(np.mean(m["precision"])), 3),
                K["f1"]: round(float(np.mean(m["F1"])), 3),
                K["p_empc"]: round(float(pe), 4), K["p_kar"]: round(float(pk), 4),
            })
    df = pd.DataFrame(rows)
    df.to_csv(cfg.TABLES / "profit_baseline_comparison.csv", index=False)
    return df


def _kaydet(fig, dosya):
    d = cfg.FIGURES / "_baseline"
    d.mkdir(parents=True, exist_ok=True)
    yol = d / dosya
    fig.savefig(yol)
    return yol


def _figur_metrik(tum, metrik, baslik, dosya):
    setler = list(tum.keys())
    x = np.arange(len(setler))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, adi in enumerate(("ours", "proflogit")):
        m = [np.mean(tum[s][0][adi][metrik]) for s in setler]
        e = [np.std(tum[s][0][adi][metrik]) for s in setler]
        ax.bar(x + (i - 0.5) * w, m, w, yerr=e, capsize=3,
               color=ps.MODEL_RENK["lightgbm"] if adi == "ours" else ps.MODEL_RENK["logreg"],
               label=S.YAKLASIM_AD[adi])
    ax.set_xticks(x); ax.set_xticklabels(setler)
    ax.set_ylabel(metrik)
    ax.set_title(baslik)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return _kaydet(fig, dosya)


def figurler(tum):
    return [_figur_metrik(tum, "EMPC", S.FIG8_BASLIK["empc"], S.FIG8_DOSYA["empc"]),
            _figur_metrik(tum, "kar", S.FIG8_BASLIK["profit"], S.FIG8_DOSYA["profit"])]
