"""Model definitions + lightweight hyperparameter optimization (no GPU).

From simple to strong: Logistic Regression, Random Forest, XGBoost, LightGBM.
HPO: RandomizedSearchCV, 5-fold, scoring=average_precision (PR-AUC; imbalanced data).
No excessive tuning. Models n_jobs=1; parallelism at the search layer.
"""
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier

MODEL_ADLARI = ["logreg", "rf", "xgboost", "lightgbm"]


def model_uzayi(model_adi: str, seed: int):
    """Returns a model instance + search space.

    Return: (estimator, param_dist, n_iter, olcekle). Keys are prefixed with 'model__'
    (the model step in the full pipeline). olcekle=True only for LogReg (StandardScaler).
    """
    if model_adi == "logreg":
        est = LogisticRegression(max_iter=2000, solver="liblinear", random_state=seed)
        dist = {"model__C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]}
        return est, dist, 7, True

    if model_adi == "rf":
        est = RandomForestClassifier(random_state=seed, n_jobs=1)
        dist = {
            "model__n_estimators": [300, 500],
            "model__max_depth": [None, 8, 16, 24],
            "model__max_features": ["sqrt", 0.3, 0.5],
            "model__min_samples_leaf": [1, 5, 20],
        }
        return est, dist, 15, False

    if model_adi == "xgboost":
        est = XGBClassifier(
            tree_method="hist", n_jobs=1, random_state=seed,
            eval_metric="logloss", n_estimators=300,
        )
        dist = {
            "model__n_estimators": [200, 300, 400, 600],
            "model__max_depth": [3, 4, 5, 6],
            "model__learning_rate": [0.02, 0.05, 0.1],
            "model__subsample": [0.7, 0.85, 1.0],
            "model__colsample_bytree": [0.7, 0.85, 1.0],
            "model__min_child_weight": [1, 5, 10],
            "model__reg_lambda": [1.0, 5.0, 10.0],
        }
        return est, dist, 40, False

    if model_adi == "lightgbm":
        est = LGBMClassifier(n_jobs=1, random_state=seed, verbose=-1)
        dist = {
            "model__n_estimators": [200, 300, 400, 600],
            "model__num_leaves": [31, 63, 127],
            "model__learning_rate": [0.02, 0.05, 0.1],
            "model__subsample": [0.7, 0.85, 1.0],
            "model__subsample_freq": [1],
            "model__colsample_bytree": [0.7, 0.85, 1.0],
            "model__min_child_samples": [20, 50, 100],
            "model__reg_lambda": [0.0, 1.0, 5.0],
        }
        return est, dist, 40, False

    raise ValueError(f"unknown model: {model_adi}")


def hpo(pipeline_full, param_dist, n_iter, X, y, seed):
    """Finds the best parameters with RandomizedSearchCV (5-fold, PR-AUC).

    Since the encoders are inside the pipeline, in each fold they are fit only on the
    training data (no leakage). Return: (best_params, best_score).
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    rs = RandomizedSearchCV(
        pipeline_full, param_dist, n_iter=n_iter, scoring="average_precision",
        cv=cv, n_jobs=-1, random_state=seed, refit=False, error_score="raise",
    )
    rs.fit(X, y)
    return rs.best_params_, float(rs.best_score_)
