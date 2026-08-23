"""Encoding / preprocessing — NO LEAKAGE (encoders are fit only on the train fold).

Returns an sklearn Pipeline + ColumnTransformer; the actual fit happens inside the
CV fold (no globally encoded CSV is saved). Target/mean encoding is not used.

cell2cell-specific (locked decisions):
- HandsetPrice: string -> numeric, 'Unknown'/unparseable -> NaN -> train median;
  additionally a handsetprice_unknown 0/1 flag is added.
- ServiceArea: most frequent 15 categories + 'Other', then one-hot.

Missing values are imputed inside the fold by MissingValueImputer, which is placed
before any resampling step so that the resampler never sees missing cells.
"""
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config as cfg
from . import strings as S


class MissingValueImputer(BaseEstimator, TransformerMixin):
    """Median / most-frequent imputation fit on the within-fold training data.

    Numeric columns are filled with the training median, categorical columns with the
    training mode ('Unknown' when a column is empty). Column names and order are kept
    unchanged so that the categorical indices used by SMOTENC stay valid.
    """

    def fit(self, X, y=None):
        self.istatistikler_ = {}
        for c in X.columns:
            if is_numeric_dtype(X[c]):
                self.istatistikler_[c] = float(X[c].median())
            else:
                mod = X[c].mode(dropna=True)
                self.istatistikler_[c] = mod.iloc[0] if len(mod) else 'Unknown'
        return self

    def transform(self, X):
        X = X.copy()
        for c, deger in self.istatistikler_.items():
            if c in X.columns:
                X[c] = X[c].fillna(deger)
        return X


class HamHazirla(BaseEstimator, TransformerMixin):
    """cell2cell high-cardinality preprocessing (fit within the fold, leakage-free).

    HandsetPrice -> numeric + handsetprice_unknown flag (median from train);
    ServiceArea -> most frequent `topn` categories + 'Other'.
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


def on_isleyici(set_adi: str, df: pd.DataFrame, olcekle: bool):
    """Preprocessor pipeline for a dataset/model type (thin wrapper over parcalar).

    olcekle=True (LogReg) -> StandardScaler on numerics; False for tree models.
    Returns: (pipeline, sema) where sema={sayisal, nominal, ozel}.
    """
    parts = parcalar(set_adi, df, olcekle)
    steps = ([("hazirla", parts["prep"])] if parts["prep"] is not None else [])
    steps.append(("doldur", parts["impute"]))
    steps.append(("ct", parts["ct"]))
    return Pipeline(steps), {"sayisal": parts["sayisal"], "nominal": parts["nominal"],
                             "ozel": parts["prep"] is not None}


def parcalar(set_adi: str, df: pd.DataFrame, olcekle: bool = False):
    """Returns the preprocessing pieces SEPARATELY (to place the resampler between prep and encode).

    Returns: {prep, impute, ct, sayisal, nominal, kolonlar, kat_idx}. `prep` is HamHazirla
    for cell2cell, None otherwise. `kat_idx` are the categorical column indices for
    the resampler (SMOTENC), following the post-preprocessing column order. `ct` is
    encode only (excluding prep).
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
    return {"prep": prep, "impute": MissingValueImputer(), "ct": ct,
            "sayisal": num, "nominal": nom, "kolonlar": kolonlar, "kat_idx": kat_idx}


def sema_yaz(set_adi: str, sema: dict):
    """Writes encoding_schema_<set>.csv (column, role, note). Returns: path."""
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
