"""Probability calibration — Platt (sigmoid) and Isotonic.

The calibrator is learned within CV, only from the training data (no test leakage):
the pipeline wrapped with CalibratedClassifierCV is fit on the training fold.
"""
from sklearn.calibration import CalibratedClassifierCV

# method list compatible with strings_tr keys
YONTEMLER = [("sigmoid", "Platt"), ("isotonic", "Isotonic")]


def kalibre(fab, best_params: dict, method: str, cv: int = 3):
    """Returns a calibrator built from the given pipeline factory + parameters (not fit).

    `fab` produces a fresh Pipeline; `method` is 'sigmoid' (Platt) or 'isotonic'.
    The inner cv splits only the training data (calibration from the training data).
    """
    taban = fab()
    taban.set_params(**best_params)
    return CalibratedClassifierCV(estimator=taban, method=method, cv=cv)
