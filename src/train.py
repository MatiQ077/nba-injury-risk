import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from .config import SEED

def train_lgbm(X_train, y_train):
    pos = y_train.sum()
    neg = len(y_train) - pos
    scale_pos_weight = (neg / pos) if pos > 0 else 1.0

    base = LGBMClassifier(
        boosting_type="gbdt",
        objective="binary",
        n_jobs=-1,
        random_state=SEED,
        scale_pos_weight=scale_pos_weight,
        verbose=-1,
        force_col_wise=True
    )

    param_distributions = {
        "n_estimators": [200, 300, 400],
        "learning_rate": [0.01, 0.02, 0.03],
        "num_leaves": [15, 31, 63],
        "max_depth": [4, 6, 8],
        "subsample": [0.7, 0.8],
        "colsample_bytree": [0.7, 0.8],
        "min_child_samples": [10, 20, 30],
        "reg_alpha": [0.1, 0.5, 1.0],
        "reg_lambda": [1.0, 2.0, 3.0],
    }

    cv = TimeSeriesSplit(n_splits=3)

    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=param_distributions,
        n_iter=12,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        random_state=SEED,
    )

    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    return best_model, search.best_params_, search.best_score_