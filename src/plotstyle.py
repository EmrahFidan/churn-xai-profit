"""Ortak figür stili ve kaydetme yardımcıları (tek yerde tanımlı, tekrarsız).

Yayın kalitesi: seaborn whitegrid, dpi=300 kayıt, okunur fontlar, tutarlı palet.
churn=0 / churn=1 tüm figürlerde sabit iki renge bağlanır.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from . import config as cfg
from . import strings_tr as S

# churn için sabit iki renk (tüm figürlerde aynı)
CHURN_RENK = {0: "#4C72B0", 1: "#C44E52"}
PALET = [CHURN_RENK[0], CHURN_RENK[1]]


def uygula():
    """Ortak temayı uygular (idempotent)."""
    sns.set_theme(style="whitegrid", context="notebook")
    mpl.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "figure.facecolor": "white",
    })


def churn_legend(ax):
    """Sabit churn renk/etiketleriyle figüre legend ekler."""
    el = [mpl.patches.Patch(color=CHURN_RENK[k], label=S.CHURN_ETIKET[k]) for k in (0, 1)]
    ax.legend(handles=el, title="Churn")


def kaydet(fig, set_adi: str, anahtar: str):
    """Figürü outputs/figures/<set>/<dosya> olarak 300 dpi PNG kaydeder.

    anahtar, strings_tr.FIG_DOSYA içindeki anahtardır. Dönüş: kayıt yolu.
    """
    d = cfg.FIGURES / set_adi
    d.mkdir(parents=True, exist_ok=True)
    yol = d / S.FIG_DOSYA[anahtar]
    fig.savefig(yol)
    return yol
