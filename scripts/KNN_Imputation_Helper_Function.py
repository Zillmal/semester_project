import warnings
import pandas as pd
import numpy as np

from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler


def fit_transform_train_test_methylation(
    X_meth,
    train_ids,
    test_ids,
    n_neighbors=10,
    weights="distance",
    scale=True
):
    """
    Fold-safe preprocessing helpers for survival-model cross-validation.

    Important:
    - Fit KNN imputation and scaling on training patients only.
    - Apply the fitted objects to the test patients.
    - Do not use meth_pam50_knn_imputed.csv for cross-validated model evaluation.
    That file is for exploratory analysis and visualizations only.
    """

    X_train = X_meth.loc[train_ids].copy()
    X_test = X_meth.loc[test_ids].copy()

    retained_cpgs = X_train.columns[~X_train.isna().all(axis=0)]

    X_train = X_train[retained_cpgs]
    X_test = X_test[retained_cpgs]

    imputer = KNNImputer(
        n_neighbors=n_neighbors,
        weights=weights
    )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            module="sklearn.utils.extmath"
        )

        X_train_imputed = imputer.fit_transform(X_train)
        X_test_imputed = imputer.transform(X_test)

    X_train_imputed = pd.DataFrame(
        X_train_imputed,
        index=X_train.index,
        columns=retained_cpgs
    )

    X_test_imputed = pd.DataFrame(
        X_test_imputed,
        index=X_test.index,
        columns=retained_cpgs
    )

    scaler = None

    if scale:
        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(X_train_imputed)
        X_test_scaled = scaler.transform(X_test_imputed)

        X_train_processed = pd.DataFrame(
            X_train_scaled,
            index=X_train.index,
            columns=retained_cpgs
        )

        X_test_processed = pd.DataFrame(
            X_test_scaled,
            index=X_test.index,
            columns=retained_cpgs
        )
    else:
        X_train_processed = X_train_imputed
        X_test_processed = X_test_imputed

    fitted_objects = {
        "imputer": imputer,
        "scaler": scaler,
        "retained_cpgs": list(retained_cpgs),
        "n_neighbors": n_neighbors,
        "weights": weights,
        "scale": scale
    }

    return X_train_processed, X_test_processed, fitted_objects