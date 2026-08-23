"""Framework figure (Figure 1) — study design and within-fold estimation boundary.

Draws the five datasets grouped by sector, the analysis pipeline, and the three research
questions. Steps whose parameters are learned from data are enclosed in a dashed frame to
show that they are fit on the training part of each cross-validation fold only. Dataset
sizes and churn rates are read from the processed CSV files, so the figure never carries
hard-coded numbers.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from . import config as cfg

NL = chr(10)

SEKTOR_RENK = {'telecom': '#3d6a9e', 'banking': '#3f8a55', 'ecommerce': '#94447c'}
SEKTOR_DOLGU = {'telecom': '#dce9f7', 'banking': '#e0f0e2', 'ecommerce': '#f3e4ef'}
SEKTOR_AD = {'telecom': 'Telecom', 'banking': 'Banking', 'ecommerce': 'E-commerce'}
FOLD_RENK = '#b07a2a'


def _set_ozeti():
    """Reads row count and churn rate per dataset. Return: {sector: [(label, n, rate)]}."""
    gruplar = {}
    for anahtar, bilgi in cfg.DATASETS.items():
        yol = cfg.PROCESSED / f'{anahtar}_clean.csv'
        df = pd.read_csv(yol)
        oran = float(df['churn'].mean())
        etiket = bilgi['name'].replace(' Customer Churn', '').replace(' Churn', '')
        gruplar.setdefault(bilgi['sector'], []).append((etiket, len(df), oran))
    return gruplar


def _kutu(ax, x, y, w, h, baslik, detay, dolgu, kenar, kalin=1.3):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.10',
                                fc=dolgu, ec=kenar, lw=kalin))
    ax.text(x + w / 2, y + h - 0.24, baslik, ha='center', va='center',
            fontsize=9.5, fontweight='bold')
    ax.text(x + w / 2, y + (h - 0.42) / 2, detay, ha='center', va='center', fontsize=7.8)


def cizim(kayit_yolu=None):
    """Builds the framework figure. Return: saved path."""
    gruplar = _set_ozeti()
    fig, ax = plt.subplots(figsize=(14.4, 7.0))
    ax.set_xlim(-0.2, 14.2)
    ax.set_ylim(0, 7.1)
    ax.axis('off')

    ax.text(7.0, 6.75, 'Five public datasets, analysed separately and never pooled',
            ha='center', fontsize=10.5, fontweight='bold', color='#333333')

    # ---- sector groups and dataset boxes ----
    gy, gh, dy, dh = 5.05, 1.28, 5.20, 0.95
    merkezler = []
    x = 0.55
    for sektor in ('telecom', 'banking', 'ecommerce'):
        setler = gruplar.get(sektor, [])
        genislik = len(setler) * 1.70 + 0.25
        ax.add_patch(FancyBboxPatch((x, gy), genislik, gh, boxstyle='round,pad=0.14',
                                    fc='none', ec=SEKTOR_RENK[sektor], lw=1.0,
                                    linestyle=(0, (4, 3))))
        ax.text(x + 0.22, gy + gh + 0.15, SEKTOR_AD[sektor], fontsize=9,
                fontweight='bold', color=SEKTOR_RENK[sektor])
        bx = x + 0.20
        for etiket, n, oran in setler:
            _kutu(ax, bx, dy, 1.50, dh, etiket,
                  f'n={n:,}' + NL + f'churn {oran*100:.1f}%',
                  SEKTOR_DOLGU[sektor], SEKTOR_RENK[sektor])
            bx += 1.70
        merkezler.append(x + genislik / 2)
        x += genislik + 0.55

    # ---- collector bus into the pipeline ----
    bus_y, y0 = 4.62, 2.55
    giris_x = xs_giris = 0.28 + 2.35 / 2
    for mx in merkezler:
        ax.plot([mx, mx], [gy, bus_y], color='#8a8a8a', lw=1.1)
    ax.plot([min(giris_x, *merkezler), max(merkezler)], [bus_y, bus_y],
            color='#8a8a8a', lw=1.1)
    ax.add_patch(FancyArrowPatch((giris_x, bus_y), (giris_x, y0 + 1.15),
                                 arrowstyle='-|>', mutation_scale=13, lw=1.3,
                                 color='#8a8a8a', shrinkA=0, shrinkB=2))

    # ---- within-fold frame, drawn behind the pipeline boxes ----
    ax.add_patch(FancyBboxPatch((2.82, y0 - 0.34), 8.30, 1.86,
                                boxstyle='round,pad=0.10', fc='#fdf7ec', ec=FOLD_RENK,
                                lw=1.6, linestyle=(0, (5, 3)), zorder=0))
    ax.text(6.97, y0 - 0.60, 'fit on the training part of each cross-validation fold',
            ha='center', fontsize=8.4, style='italic', color=FOLD_RENK)

    # ---- pipeline ----
    h, w = 1.15, 2.35
    xs = [0.28, 3.02, 5.76, 8.50, 11.24]
    basliklar = ['Cleaning + leakage audit', 'Imputation + encoding',
                 'Calibrated prediction', 'Profit decision', 'Evaluation']
    detaylar = [
        'deterministic fixes only' + NL + 'univariate AUC screen',
        'median / most-frequent' + NL + 'one-hot (+ resampling)',
        'LightGBM, raw probabilities' + NL + 'stratified 5-fold CV',
        'profit-maximizing' + NL + 'threshold; grid c, ' + chr(947),
        'profit, ROI, EMP' + NL + '5 seeds, 95% CI',
    ]
    fold_ici = [False, True, True, True, False]
    for i in range(5):
        kenar = FOLD_RENK if fold_ici[i] else '#4a6789'
        dolgu = '#ffffff' if fold_ici[i] else '#eef2f7'
        _kutu(ax, xs[i], y0, w, h, basliklar[i], detaylar[i], dolgu, kenar,
              kalin=1.6 if fold_ici[i] else 1.3)
    for i in range(4):
        ax.add_patch(FancyArrowPatch((xs[i] + w, y0 + h / 2), (xs[i + 1], y0 + h / 2),
                                     arrowstyle='-|>', mutation_scale=15, lw=1.4,
                                     color='#4a6789', shrinkA=0, shrinkB=0))

    # ---- research questions ----
    rq_y, rq_h, rq_w = 0.35, 0.98, 2.70
    rq_metin = [
        'RQ1: imbalance handling' + NL + '(resampling vs. threshold)',
        'RQ2: driver consistency' + NL + '+ transfer probe',
        'RQ3: profit vs. accuracy;' + NL + 'predictability vs. profitability',
    ]
    for rx, metin, hedef in zip([4.50, 7.35, 10.20], rq_metin,
                                [xs[1] + w / 2, xs[2] + w / 2, xs[3] + w / 2]):
        ax.add_patch(FancyBboxPatch((rx, rq_y), rq_w, rq_h, boxstyle='round,pad=0.10',
                                    fc='#f4f4f4', ec='#8a8a8a', lw=1.0))
        ax.text(rx + rq_w / 2, rq_y + rq_h / 2, metin, ha='center', va='center', fontsize=8.4)
        ax.add_patch(FancyArrowPatch((rx + rq_w / 2, rq_y + rq_h), (hedef, y0),
                                     arrowstyle='-|>', mutation_scale=12, lw=1.2,
                                     color='#999999', shrinkA=1, shrinkB=1))

    fig.tight_layout()
    yol = kayit_yolu or (cfg.FIGURES / '_framework' / 'fig1_framework.png')
    yol.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(yol, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return yol

