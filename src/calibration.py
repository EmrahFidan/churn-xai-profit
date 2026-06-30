"""Olasılık kalibrasyonu — Platt (sigmoid) ve Isotonic.

Kalibratör CV içinde, yalnız eğitim verisinden öğrenilir (test sızdırmaz):
CalibratedClassifierCV ile sarılmış pipeline, eğitim katında fit edilir.
"""
from sklearn.calibration import CalibratedClassifierCV

# strings_tr anahtarlarıyla uyumlu yöntem listesi
YONTEMLER = [("sigmoid", "Platt"), ("isotonic", "Isotonic")]


def kalibre(fab, best_params: dict, method: str, cv: int = 3):
    """Verilen pipeline fabrikası + parametrelerle kalibre edici döndürür (fit edilmemiş).

    `fab` taze bir Pipeline üretir; `method` 'sigmoid' (Platt) veya 'isotonic'.
    İçteki cv yalnız eğitim verisini böler (kalibrasyon eğitim verisinden).
    """
    taban = fab()
    taban.set_params(**best_params)
    return CalibratedClassifierCV(estimator=taban, method=method, cv=cv)
