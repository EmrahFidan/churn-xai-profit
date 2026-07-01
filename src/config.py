"""Configuration and path constants.

Reads config.yaml (random seed + dataset registry: file, sector, target) and
provides common paths to all modules. Datasets are processed independently, never
merged.
"""
from pathlib import Path
import yaml

# Repository root (this file is under src/)
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
HOLDOUT = ROOT / "data" / "holdout"
PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "outputs" / "figures"
TABLES = ROOT / "outputs" / "tables"
LOGS = ROOT / "outputs" / "logs"


def yukle():
    """Returns the contents of config.yaml as a dictionary."""
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


CFG = yukle()
SEED = CFG["random_seed"]
DATASETS = CFG["datasets"]  # key -> {name, file, sector, target}

# Reference profiles (validation to avoid blind trust): rows, columns(raw), churn%
REFERANS = {
    "telco": (7043, 21, 26.5),
    "cell2cell": (51047, 58, 28.8),
    "ecommerce": (3941, 11, 17.1),
    "iranian": (3150, 14, 15.7),
    "bank": (10000, 14, 20.4),
}
HOLDOUT_DOSYA = "cell2cell_test.csv"  # unlabeled, same schema as cell2cell


def klasorleri_hazirla():
    """Creates the output folders (idempotent)."""
    for d in (PROCESSED, FIGURES, TABLES, LOGS):
        d.mkdir(parents=True, exist_ok=True)
