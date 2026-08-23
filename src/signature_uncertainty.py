"""Uncertainty around the predictability-profitability ordering (R1.4 / R2.3 / R3.4).

With five datasets a rank correlation carries little statistical power, so the coefficient
is reported together with an exact permutation p-value and a bootstrap interval, and the
per-dataset EMP spread across seeds is written out so that the ordering can be judged
against its own noise.

Outputs: outputs/tables/signature_uncertainty.csv, outputs/tables/emp_by_dataset_ci.csv
"""
import itertools

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import config as cfg
from . import profit as pr

SEEDS = cfg.CFG["seeds"]


def permutasyon_testi(x, y):
    """Exact permutation test for Spearman rho (5! = 120 orderings). Return: (rho, p)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    gozlenen = spearmanr(x, y).statistic
    dagilim = [spearmanr(x, np.asarray(perm)).statistic
               for perm in itertools.permutations(y)]
    dagilim = np.asarray(dagilim, float)
    p = float(np.mean(np.abs(dagilim) >= abs(gozlenen) - 1e-12))
    return float(gozlenen), p


def bootstrap_araligi(x, y, n=10000, seed=None):
    """Bootstrap interval for rho over datasets. Wide by construction with five points."""
    rng = np.random.default_rng(cfg.SEED if seed is None else seed)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    k = len(x)
    ornekler = []
    for _ in range(n):
        idx = rng.integers(0, k, k)
        if len(np.unique(idx)) < 3:
            continue
        r = spearmanr(x[idx], y[idx]).statistic
        if np.isfinite(r):
            ornekler.append(r)
    ornekler = np.asarray(ornekler, float)
    return float(np.percentile(ornekler, 2.5)), float(np.percentile(ornekler, 97.5))


def emp_tohum_araligi(veriler):
    """Per-dataset EMP across seeds: mean, standard deviation and 95% interval."""
    satirlar = []
    for ad in cfg.DATASETS:
        df = veriler[ad]
        degerler = []
        for seed in SEEDS:
            p, y = pr.oof_olasilik(ad, df, seed)
            clv, _ = pr.clv_hesapla(ad, df)
            c_ref = float(np.mean(clv)) * pr.PARAM["emp_c_oran"]
            degerler.append(pr.emp(p, y, clv, c_ref, seed=seed))
        degerler = np.asarray(degerler, float)
        yari = 1.96 * degerler.std(ddof=1) / np.sqrt(len(degerler))
        satirlar.append({
            "Dataset": ad,
            "EMP mean": round(float(degerler.mean()), 4),
            "EMP sd": round(float(degerler.std(ddof=1)), 4),
            "CI low": round(float(degerler.mean() - yari), 4),
            "CI high": round(float(degerler.mean() + yari), 4),
            "Seeds": len(degerler),
        })
    df = pd.DataFrame(satirlar)
    df.to_csv(cfg.TABLES / "emp_by_dataset_ci.csv", index=False)
    return df


def calistir(veriler, prauc, emp):
    """Writes signature_uncertainty.csv and emp_by_dataset_ci.csv. Return: (df, ci_df)."""
    setler = list(cfg.DATASETS)
    x = [prauc[s] for s in setler]
    y = [emp[s] for s in setler]

    rho, p_perm = permutasyon_testi(x, y)
    alt, ust = bootstrap_araligi(x, y)
    df = pd.DataFrame([{
        "Datasets (n)": len(setler),
        "Spearman rho": round(rho, 3),
        "Asymptotic p": round(float(spearmanr(x, y).pvalue), 4),
        "Exact permutation p": round(p_perm, 4),
        "Bootstrap CI low": round(alt, 3),
        "Bootstrap CI high": round(ust, 3),
    }])
    df.to_csv(cfg.TABLES / "signature_uncertainty.csv", index=False)
    return df, emp_tohum_araligi(veriler)

