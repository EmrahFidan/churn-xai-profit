"""Protocol comparison: results before and after moving imputation into the fold.

Reads the tables produced by the earlier run, kept under the revision folder, and the
tables produced after the change, then reports the difference for the headline metrics.
The point of the table is to show that relocating the imputation step does not move the
reported numbers, which is what the original manuscript anticipated.

Output: outputs/tables/protocol_comparison.csv
"""
from pathlib import Path

import pandas as pd

ONCE = Path('/Users/emrahfidan/Desktop/MAKALE-2-REVIZYON/06_YENI_SONUCLAR/outputs_ONCESI/tables')
SONRA = Path('/Users/emrahfidan/Desktop/MAKALE-2/outputs/tables')


def _oku(kok, ad):
    yol = kok / ad
    return pd.read_csv(yol) if yol.exists() else None


def _sayiya(seri):
    """Pulls the point estimate out of cells written as 'mean ± sd'."""
    return pd.to_numeric(
        seri.astype(str).str.split('±').str[0].str.strip(), errors='coerce')


def _anahtar_kolon(df, adaylar):
    for a in adaylar:
        if a in df.columns:
            return a
    return None


def karsilastir(dosya='model_performance.csv', metrik_adaylari=('PR-AUC', 'PR_AUC', 'prauc')):
    """Joins the two runs on dataset and model and reports the metric difference."""
    once, sonra = _oku(ONCE, dosya), _oku(SONRA, dosya)
    if once is None or sonra is None:
        return None

    set_k = _anahtar_kolon(once, ('Dataset', 'dataset', 'Set', 'set'))
    mod_k = _anahtar_kolon(once, ('Model', 'model'))
    met_k = _anahtar_kolon(once, metrik_adaylari)
    if not all((set_k, mod_k, met_k)):
        return None

    a = once[[set_k, mod_k, met_k]].rename(columns={met_k: 'before'})
    b = sonra[[set_k, mod_k, met_k]].rename(columns={met_k: 'after'})
    df = a.merge(b, on=[set_k, mod_k], how='inner')
    for c in ('before', 'after'):
        df[c] = _sayiya(df[c])
    df['difference'] = (df['after'] - df['before']).round(4)
    df['before'] = df['before'].round(4)
    df['after'] = df['after'].round(4)
    df = df.rename(columns={set_k: 'Dataset', mod_k: 'Model'})
    df.to_csv(SONRA / 'protocol_comparison.csv', index=False)
    return df


def ozet(df):
    """One-line summary for the response letter."""
    if df is None or df.empty:
        return 'comparison not available'
    en_buyuk = df['difference'].abs().max()
    return (f'{len(df)} dataset-model pairs compared; '
            f'largest absolute change in PR-AUC: {en_buyuk:.4f}')

