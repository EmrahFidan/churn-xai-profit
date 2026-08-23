"""Nested cross-validation check (R2.5 / R3.6).

In the main pipeline the hyperparameters are chosen by a randomized search over the whole
dataset and the selected configuration is then scored by cross-validation on the same
data. That design keeps every data-derived transformation inside the fold, but the choice
of configuration is still informed by all folds, which can make the reported scores
optimistic. This module repeats the evaluation with the search nested inside an outer
loop, so the outer test fold takes no part in selecting the configuration, and reports the
difference between the two protocols.

Output: outputs/tables/nested_cv_comparison.csv
"""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from . import config as cfg
from . import encode
from . import evaluate as ev
from . import models as models_mod

DIS_KAT = 5
IC_KAT = 5


def _fabrika(set_adi, df, madi, seed):
    """Returns (unfitted pipeline factory, search space, budget) for one model."""
    est, dist, n_iter, olcekle = models_mod.model_uzayi(madi, seed)
    pre = encode.on_isleyici(set_adi, df, olcekle)[0]
    return (lambda: Pipeline([("pre", clone(pre)), ("model", clone(est))]), dist, n_iter)


def nested_prauc(set_adi, df, madi, seed):
    """PR-AUC with the search nested inside the outer loop. Return: (mean, sd)."""
    X = df.drop(columns=["churn"])
    y = df["churn"].to_numpy()
    fab, dist, n_iter = _fabrika(set_adi, df, madi, seed)
    dis = StratifiedKFold(n_splits=DIS_KAT, shuffle=True, random_state=seed)
    skorlar = []
    for tr, te in dis.split(X, y):
        Xtr, ytr = X.iloc[tr], y[tr]
        bp, _ = models_mod.hpo(fab(), dist, n_iter, Xtr, ytr, seed)
        pipe = fab().set_params(**bp)
        pipe.fit(Xtr, ytr)
        p = pipe.predict_proba(X.iloc[te])[:, 1]
        skorlar.append(average_precision_score(y[te], p))
    skorlar = np.asarray(skorlar, float)
    return float(skorlar.mean()), float(skorlar.std(ddof=1))


def calistir(veriler, model_adlari=None, seed=None):
    """Compares the reported protocol with nested cross-validation on every dataset.

    Return: DataFrame written to nested_cv_comparison.csv.
    """
    seed = cfg.SEED if seed is None else seed
    model_adlari = model_adlari or models_mod.MODEL_ADLARI
    satirlar = []
    for set_adi, df in veriler.items():
        sonuc = ev.calistir_set(set_adi, df, seed, model_adlari=model_adlari,
                                kaydet_sema=False)
        for madi in model_adlari:
            rapor = ev.pr_ortalama(sonuc["sonuc"][madi])
            ic_dis, sd = nested_prauc(set_adi, df, madi, seed)
            satirlar.append({
                "Dataset": set_adi,
                "Model": madi,
                "PR-AUC (reported protocol)": round(rapor, 4),
                "PR-AUC (nested CV)": round(ic_dis, 4),
                "Nested sd": round(sd, 4),
                "Difference": round(rapor - ic_dis, 4),
            })
    df = pd.DataFrame(satirlar)
    df.to_csv(cfg.TABLES / "nested_cv_comparison.csv", index=False)
    return df

