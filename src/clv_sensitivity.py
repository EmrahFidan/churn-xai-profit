"""CLV sensitivity analysis (reviewer request R1.3 / R2.2 / R3.10).

The customer lifetime value proxies differ across datasets: a monthly-charge horizon for
the telecom sets, an annual margin on the account balance for Bank, and a cashback proxy
for E-commerce. This module re-runs the profit evaluation over a grid of horizons and
bank margins and reports whether the reported ordering between predictability and
profitability survives those choices.

Output: outputs/tables/clv_sensitivity.csv
"""
import itertools

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import config as cfg
from . import profit as pr
from . import signature as sg

UFUKLAR = [12, 24, 36]          # CLV horizon in months
BANKA_MARJLARI = [0.01, 0.02, 0.03]   # annual margin proxy for the Bank dataset


def _emp_seti(veriler, seed):
    """Per-customer EMP for every dataset under the parameters currently in PARAM."""
    degerler = {}
    for ad in cfg.DATASETS:
        df = veriler[ad]
        p, y = pr.oof_olasilik(ad, df, seed)
        clv, _ = pr.clv_hesapla(ad, df)
        c_ref = float(np.mean(clv)) * pr.PARAM["emp_c_oran"]
        degerler[ad] = pr.emp(p, y, clv, c_ref, seed=seed)
    return degerler


def calistir(veriler, prauc, seed=None):
    """Sweeps the CLV grid and records the predictability-profitability correlation.

    prauc: {dataset: PR-AUC} from the main run, held fixed because the CLV assumptions
    do not affect ranking quality. Return: DataFrame written to clv_sensitivity.csv.
    """
    seed = cfg.SEED if seed is None else seed
    ufuk_ilk = pr.PARAM["ufuk_ay"]
    marj_ilk = pr.PARAM["banka_marj_yillik"]
    setler = list(cfg.DATASETS)
    satirlar = []

    try:
        for ufuk, marj in itertools.product(UFUKLAR, BANKA_MARJLARI):
            pr.PARAM["ufuk_ay"] = ufuk
            pr.PARAM["banka_marj_yillik"] = marj
            emp = _emp_seti(veriler, seed)
            x = [prauc[s] for s in setler]
            y = [emp[s] for s in setler]
            rho, p_rho = spearmanr(x, y)
            satir = {
                "Horizon (months)": ufuk,
                "Bank annual margin": marj,
                "Spearman rho": round(float(rho), 3),
                "p-value": round(float(p_rho), 4),
                "Sign preserved": "yes" if rho < 0 else "no",
            }
            for s in setler:
                satir[f"EMP {s}"] = round(float(emp[s]), 4)
            satirlar.append(satir)
    finally:
        pr.PARAM["ufuk_ay"] = ufuk_ilk
        pr.PARAM["banka_marj_yillik"] = marj_ilk

    df = pd.DataFrame(satirlar)
    df.to_csv(cfg.TABLES / "clv_sensitivity.csv", index=False)
    return df

