"""Encoding / önişleme — SIZINTI YOK (encoder'lar yalnız train-fold'da fit edilir).

sklearn Pipeline + ColumnTransformer döndürür; gerçek fit CV katı içinde olur
(global encode edilmiş CSV kaydedilmez). Target/mean encoding kullanılmaz.

cell2cell'e özel (kilitli kararlar):
- HandsetPrice: string -> sayısal, 'Unknown'/parse edilemez -> NaN -> train medyanı;
  ayrıca handsetprice_unknown 0/1 bayrağı eklenir.
- ServiceArea: en sık 15 kategori + 'Other', sonra one-hot.
"""
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config as cfg
from . import strings_tr as S


class HamHazirla(BaseEstimator, TransformerMixin):
    """cell2cell yüksek-kardinalite önişlemi (fold içinde fit edilir, sızıntısız).

    HandsetPrice -> sayısal + handsetprice_unknown bayrağı (medyan train'den);
    ServiceArea -> en sık `topn` kategori + 'Other'.
    """

    def __init__(self, topn: int = 15):
        self.topn = topn

    def fit(self, X, y=None):
        hp = pd.to_numeric(X["HandsetPrice"], errors="coerce")
        self.hp_medyan_ = float(hp.median())
        self.top_ = X["ServiceArea"].value_counts().head(self.topn).index.tolist()
        return self

    def transform(self, X):
        X = X.copy()
        hp = pd.to_numeric(X["HandsetPrice"], errors="coerce")
        X["handsetprice_unknown"] = hp.isna().astype(int)
        X["HandsetPrice"] = hp.fillna(self.hp_medyan_)
        X["ServiceArea"] = X["ServiceArea"].where(X["ServiceArea"].isin(self.top_), "Other")
        return X


def _ornek_kolonlar(set_adi, feats):
    """Önişlem sonrası kolon adlarını/rolleri belirler (yapısal metadata, sızıntı değil)."""
    if set_adi == "cell2cell":
        ornek = HamHazirla().fit_transform(feats)
    else:
        ornek = feats
    num = [c for c in ornek.columns if is_numeric_dtype(ornek[c])]
    nom = [c for c in ornek.columns if not is_numeric_dtype(ornek[c])]
    return num, nom


def on_isleyici(set_adi: str, df: pd.DataFrame, olcekle: bool):
    """Set + model tipine göre önişleyici Pipeline döndürür (fit edilmemiş).

    olcekle=True (LogReg) -> sayısallara StandardScaler; ağaç modelleri için False.
    Dönüş: (pipeline, sema) — sema={sayisal, nominal, ozel}.
    """
    feats = df.drop(columns=["churn"])
    ozel = set_adi == "cell2cell"
    num, nom = _ornek_kolonlar(set_adi, feats)

    num_tf = StandardScaler() if olcekle else "passthrough"
    transformers = [("num", num_tf, num)]
    if nom:
        transformers.append(("nom", OneHotEncoder(handle_unknown="ignore", sparse_output=False), nom))
    ct = ColumnTransformer(transformers, remainder="drop")

    steps = []
    if ozel:
        steps.append(("hazirla", HamHazirla()))
    steps.append(("ct", ct))
    return Pipeline(steps), {"sayisal": num, "nominal": nom, "ozel": ozel}


def parcalar(set_adi: str, df: pd.DataFrame, olcekle: bool = False):
    """Önişlem parçalarını AYRI döndürür (resampler'ı prep ile encode arasına koymak için).

    Dönüş: {prep, ct, sayisal, nominal, kolonlar, kat_idx}. `prep` cell2cell için
    HamHazirla, diğerlerinde None. `kat_idx` resampler (SMOTENC) için kategorik kolon
    indeksleri (önişlem sonrası kolon sırasına göre). `ct` yalnız encode (prep hariç).
    """
    feats = df.drop(columns=["churn"])
    ozel = set_adi == "cell2cell"
    prep = HamHazirla() if ozel else None
    ornek = HamHazirla().fit_transform(feats) if ozel else feats
    num = [c for c in ornek.columns if is_numeric_dtype(ornek[c])]
    nom = [c for c in ornek.columns if not is_numeric_dtype(ornek[c])]

    num_tf = StandardScaler() if olcekle else "passthrough"
    transformers = [("num", num_tf, num)]
    if nom:
        transformers.append(("nom", OneHotEncoder(handle_unknown="ignore", sparse_output=False), nom))
    ct = ColumnTransformer(transformers, remainder="drop")

    kolonlar = list(ornek.columns)
    kat_idx = [kolonlar.index(c) for c in nom]
    return {"prep": prep, "ct": ct, "sayisal": num, "nominal": nom,
            "kolonlar": kolonlar, "kat_idx": kat_idx}


def sema_yaz(set_adi: str, sema: dict):
    """encoding_schema_<set>.csv yazar (kolon, rol, not). Dönüş: yol."""
    cfg.klasorleri_hazirla()
    satir = []
    ozel_kolon = {"HandsetPrice", "handsetprice_unknown", "ServiceArea"} if sema["ozel"] else set()
    for c in sema["sayisal"]:
        if c == "HandsetPrice":
            rol, notu = S.ROL["sayisal"], S.ROL["ozel_hp"]
        elif c == "handsetprice_unknown":
            rol, notu = S.ROL["sayisal"], S.ROL["ozel_hp_bayrak"]
        else:
            rol, notu = S.ROL["sayisal"], ""
        satir.append((c, rol, notu))
    for c in sema["nominal"]:
        if c == "ServiceArea":
            rol, notu = S.ROL["nominal"], S.ROL["ozel_sa"]
        else:
            rol, notu = S.ROL["nominal"], ""
        satir.append((c, rol, notu))
    df = pd.DataFrame(satir, columns=[S.KOLON2["kolon"], S.KOLON2["rol"], S.KOLON2["not"]])
    yol = cfg.TABLES / f"encoding_schema_{set_adi}.csv"
    df.to_csv(yol, index=False)
    return yol
