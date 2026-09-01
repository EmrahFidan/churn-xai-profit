"""Multi-dataset reliability panel (Figure 2).

Reviewer 3 asked for reliability curves on more than one dataset, so that the claim about
raw LightGBM probabilities is supported across sectors rather than on a single case. This
module recomputes out-of-fold probabilities for the carrier model on every dataset, applies
Platt scaling and isotonic regression inside the fold, and draws one panel per dataset with
the expected calibration error reported in the legend.

Output: outputs/figures/_calibration/fig2_calibration_panel.png
"""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from . import config as cfg
from . import encode
from . import models as models_mod

TASIYICI = 'lightgbm'
KUTU = 10
YONTEMLER = ('raw', 'sigmoid', 'isotonic')
ETIKET = {'raw': 'Raw', 'sigmoid': 'Platt', 'isotonic': 'Isotonic'}
RENK = {'raw': '#8172B3', 'sigmoid': '#C44E52', 'isotonic': '#55A868'}


def _ece(p, y, kutu=KUTU):
    """Expected calibration error with equal-width bins."""
    kenar = np.linspace(0.0, 1.0, kutu + 1)
    toplam, n = 0.0, len(y)
    for i in range(kutu):
        ust = p <= kenar[i + 1] if i == kutu - 1 else p < kenar[i + 1]
        maske = (p >= kenar[i]) & ust
        if maske.sum() == 0:
            continue
        toplam += maske.sum() / n * abs(y[maske].mean() - p[maske].mean())
    return float(toplam)


def _egri(p, y, kutu=KUTU):
    """Bin centres and observed frequencies for the reliability curve."""
    kenar = np.linspace(0.0, 1.0, kutu + 1)
    xs, ys = [], []
    for i in range(kutu):
        ust = p <= kenar[i + 1] if i == kutu - 1 else p < kenar[i + 1]
        maske = (p >= kenar[i]) & ust
        if maske.sum() == 0:
            continue
        xs.append(p[maske].mean())
        ys.append(y[maske].mean())
    return np.asarray(xs), np.asarray(ys)


def _en_iyi_parametre(set_adi):
    """Reads the tuned parameters saved by the modelling step, when available."""
    yol = cfg.TABLES / ('best_params_' + set_adi + '.json')
    if not yol.exists():
        return {}
    kayit = json.loads(yol.read_text(encoding='utf-8'))
    p = kayit.get(TASIYICI, {}) or {}
    return p if isinstance(p, dict) else {}


def _oof(set_adi, df, seed):
    """Out-of-fold probabilities for the three calibration variants."""
    X = df.drop(columns=['churn'])
    y = df['churn'].to_numpy()
    est, _, _, olcekle = models_mod.model_uzayi(TASIYICI, seed)
    on = encode.on_isleyici(set_adi, df, olcekle)[0]
    parametre = _en_iyi_parametre(set_adi)

    cikti = {m: np.zeros(len(y), dtype=float) for m in YONTEMLER}
    kat = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, te in kat.split(X, y):
        temel = Pipeline([('pre', clone(on)), ('model', clone(est))])
        if parametre:
            try:
                temel.set_params(**parametre)
            except ValueError:
                pass
        ham = clone(temel).fit(X.iloc[tr], y[tr])
        cikti['raw'][te] = ham.predict_proba(X.iloc[te])[:, 1]
        for yontem in ('sigmoid', 'isotonic'):
            kal = CalibratedClassifierCV(clone(temel), method=yontem, cv=3)
            kal.fit(X.iloc[tr], y[tr])
            cikti[yontem][te] = kal.predict_proba(X.iloc[te])[:, 1]
    return cikti, y


def cizim(seed=None, kayit_yolu=None):
    """Draws one reliability panel per dataset. Return: saved path."""
    seed = cfg.SEED if seed is None else seed
    setler = list(cfg.DATASETS)
    sutun = 3
    satir = int(np.ceil(len(setler) / sutun))
    fig, eksenler = plt.subplots(satir, sutun, figsize=(4.6 * sutun, 4.3 * satir))
    eksenler = np.atleast_1d(eksenler).ravel()

    for ax, set_adi in zip(eksenler, setler):
        df = pd.read_csv(cfg.PROCESSED / (set_adi + '_clean.csv'))
        olasilik, y = _oof(set_adi, df, seed)
        ax.plot([0, 1], [0, 1], color='#999999', lw=1.0, ls='--', zorder=0, clip_on=False)
        for yontem in YONTEMLER:
            p = olasilik[yontem]
            xs, ys = _egri(p, y)
            ax.plot(xs, ys, marker='o', ms=4.5, lw=1.6, color=RENK[yontem],
                    clip_on=False, zorder=3,
                    label=ETIKET[yontem] + ' (ECE ' + format(_ece(p, y), '.3f') + ')')
        ax.set_title(cfg.DATASETS[set_adi]['name'], fontsize=11)
        ax.set_xlabel('Predicted probability')
        ax.set_ylabel('Observed frequency')
        # a small pad keeps markers that land on 0 or 1 fully visible
        ax.set_xlim(-0.035, 1.035)
        ax.set_ylim(-0.035, 1.035)
        ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.legend(fontsize=8, loc='upper left', frameon=False)

    for ax in eksenler[len(setler):]:
        ax.axis('off')

    fig.tight_layout()
    yol = kayit_yolu or (cfg.FIGURES / '_calibration' / 'fig2_calibration_panel.png')
    yol.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(yol, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return yol

