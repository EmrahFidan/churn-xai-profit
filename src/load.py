"""Veri seti yükleme ve standardizasyon.

Her seti data/raw/'dan (etiketsiz holdout data/holdout/'tan) yükler, hedefi tek
isme ('churn', 0/1) çeker, kimlik kolonlarını düşürür (loglar). Tek seferde tek
set; setler asla birleştirilmez. Bu adımda ENCODING YAPILMAZ — kategorikler ham
bırakılır.
"""
import pandas as pd
from pandas.api.types import is_numeric_dtype

from . import config as cfg

# Set bazında düşürülecek kimlik kolonları
ID_KOLON = {
    "telco": ["customerID"],
    "cell2cell": ["CustomerID"],
    "bank": ["RowNumber", "CustomerId", "Surname"],
    "ecommerce": [],
    "iranian": [],
}
_POZITIF = {"yes", "true", "1", "churn", "churned"}
_NEGATIF = {"no", "false", "0"}


def _ikili_churn(s: pd.Series) -> pd.Series:
    """Hedef seriyi 0/1'e çevirir. Çevrilemeyen (etiketsiz) değerler NaN kalır.

    Sayısal 0/1 ya da metinsel Yes/No / True/False kabul eder. Etiketsiz holdout
    için tüm değerler boştur ve NaN olarak korunur (nullable Int64).
    """
    if is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").astype("Int64")
    m = s.astype("string").str.strip().str.lower()
    out = m.map(lambda v: 1 if v in _POZITIF else (0 if v in _NEGATIF else pd.NA))
    return out.astype("Int64")


def standardize(df: pd.DataFrame, key: str, hedef_kolon: str):
    """Kimlik kolonlarını düşürür, hedefi 'churn'e çevirir.

    Dönüş: (yeni_df, dusurulen_kimlik_kolonlari). 'churn' kolonu en sona eklenir;
    feature kolon adlarına dokunulmaz.
    """
    df = df.copy()
    dusurulen = [c for c in ID_KOLON.get(key, []) if c in df.columns]
    if dusurulen:
        df = df.drop(columns=dusurulen)
    churn = _ikili_churn(df[hedef_kolon])
    df = df.drop(columns=[hedef_kolon])
    df["churn"] = churn
    return df, dusurulen


def yukle_etiketli():
    """5 etiketli seti yükler ve standardize eder.

    Dönüş: key -> {df, sector, dropped, raw_shape, raw}. 'df' standardize edilmiş
    (churn int), 'raw' ham yüklenmiş kopyadır (eksiklik figürü/genel bakış için).
    """
    out = {}
    for key, d in cfg.DATASETS.items():
        raw = pd.read_csv(cfg.RAW / d["file"], low_memory=False)
        std, dusurulen = standardize(raw, key, d["target"])
        assert std["churn"].notna().all(), f"{key}: etiketli sette boş churn var"
        std["churn"] = std["churn"].astype(int)
        out[key] = {
            "df": std,
            "sector": d["sector"],
            "dropped": dusurulen,
            "raw_shape": raw.shape,
            "raw": raw,
        }
    return out


def yukle_holdout():
    """Etiketsiz cell2cell test holdout'unu yükler (cell2cell şemasıyla aynı).

    churn kolonu tamamen NaN'dır; EDA/leakage'ın churn'e dayalı kısımlarına dahil
    edilmez, yalnızca temizliği yapılıp ayrı saklanır.
    """
    raw = pd.read_csv(cfg.HOLDOUT / cfg.HOLDOUT_DOSYA, low_memory=False)
    std, dusurulen = standardize(raw, "cell2cell", cfg.DATASETS["cell2cell"]["target"])
    return {"df": std, "dropped": dusurulen, "raw_shape": raw.shape, "raw": raw}
