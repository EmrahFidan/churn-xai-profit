"""Shared figure style and saving helpers (defined in one place, no repetition).

Publication quality: seaborn whitegrid, dpi=300 saving, readable fonts, consistent palette.
churn=0 / churn=1 are bound to two fixed colors across all figures.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from . import config as cfg
from . import strings as S

# two fixed colors for churn (same across all figures)
CHURN_RENK = {0: "#4C72B0", 1: "#C44E52"}

# fixed colors for models (curve/bar figures)
MODEL_RENK = {
    "logreg": "#4C72B0",
    "rf": "#55A868",
    "xgboost": "#C44E52",
    "lightgbm": "#8172B3",
}
# fixed line styles for calibration methods
YONTEM_STIL = {
    "ham": {"linestyle": ":", "color": "#7F7F7F"},
    "Platt": {"linestyle": "--", "color": "#4C72B0"},
    "Isotonic": {"linestyle": "-", "color": "#C44E52"},
}

# fixed colors for RQ1 resampling conditions
KOSUL_RENK = {
    "baseline": "#7F7F7F",
    "class_weight": "#4C72B0",
    "smote": "#55A868",
    "adasyn": "#C44E52",
    "threshold": "#8172B3",
}


def uygula():
    """Applies the shared theme (idempotent)."""
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



def kaydet(fig, set_adi: str, anahtar: str):
    """Saves the figure as a 300 dpi PNG at outputs/figures/<set>/<dosya>.

    anahtar is the key in strings_tr.FIG_DOSYA. Return: the saved path.
    """
    d = cfg.FIGURES / set_adi
    d.mkdir(parents=True, exist_ok=True)
    yol = d / S.FIG_DOSYA[anahtar]
    fig.savefig(yol)
    return yol
