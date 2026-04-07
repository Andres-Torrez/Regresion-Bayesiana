import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump

from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.pipeline import Pipeline

from src.preprocessing.preprocessing import (
    load_data,
    clean_data,
    prepare_features,
    build_preprocessing_pipeline,
)


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mean_bias_error(y_true, y_pred) -> float:
    return float(np.mean(y_pred - y_true))


def chronological_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    df_ref: pd.DataFrame,
    test_size: float = 0.2,
):
    """
    Split cronológico basado en el orden temporal ya existente en df_ref.
    Asume que df_ref, X e y están alineados por índice/orden.
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    n_samples = len(X)
    split_idx = int(n_samples * (1 - test_size))

    if split_idx <= 0 or split_idx >= n_samples:
        raise ValueError("Invalid split index generated from test_size.")

    X_train = X.iloc[:split_idx].copy()
    X_test = X.iloc[split_idx:].copy()
    y_train = y.iloc[:split_idx].copy()
    y_test = y.iloc[split_idx:].copy()

    df_train = df_ref.iloc[:split_idx].copy()
    df_test = df_ref.iloc[split_idx:].copy()

    return X_train, X_test, y_train, y_test, df_train, df_test


def evaluate_split(model, X_train, y_train, X_test, y_test) -> dict:
    model.fit(X_train, y_train)

    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    train_rmse = rmse(y_train, pred_train)
    test_rmse = rmse(y_test, pred_test)

    metrics = {
        "train": {
            "mae": float(mean_absolute_error(y_train, pred_train)),
            "rmse": train_rmse,
            "r2": float(r2_score(y_train, pred_train)),
            "mbe": mean_bias_error(y_train, pred_train),
        },
        "test": {
            "mae": float(mean_absolute_error(y_test, pred_test)),
            "rmse": test_rmse,
            "r2": float(r2_score(y_test, pred_test)),
            "mbe": mean_bias_error(y_test, pred_test),
        },
        "overfitting": {
            "rmse_gap_ratio": float((test_rmse - train_rmse) / train_rmse) if train_rmse > 0 else None,
            "mae_gap_ratio": float(
                (
                    mean_absolute_error(y_test, pred_test)
                    - mean_absolute_error(y_train, pred_train)
                ) / mean_absolute_error(y_train, pred_train)
            ) if mean_absolute_error(y_train, pred_train) > 0 else None,
        }
    }

    return metrics


def build_model(X_train: pd.DataFrame) -> Pipeline:
    preprocessor = build_preprocessing_pipeline(X_train)

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", BayesianRidge()),
        ]
    )
    return model


def main(
    data_path: str = "data/raw/amazon_sales_dataset.csv",
    target: str = "quantity_sold",
    out_dir: str = "models/baseline",
    report_dir: str = "reports/metrics",
    test_size: float = 0.2,
    n_splits_cv: int = 5,
):
    out_dir = Path(out_dir)
    report_dir = Path(report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load + clean
    df = load_data(data_path)
    df = clean_data(df)

    if "order_date" not in df.columns:
        raise ValueError("The dataset must contain 'order_date' for time-aware validation.")

    # 2) Sort chronologically BEFORE feature prep
    df = df.sort_values("order_date").reset_index(drop=True)

    # 3) Prepare X, y, reference df
    X, y, df_prepared = prepare_features(df, target=target)

    # Garantiza alineación y orden
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    df_prepared = df_prepared.reset_index(drop=True)

    # 4) Chronological holdout split
    X_train, X_test, y_train, y_test, df_train, df_test = chronological_train_test_split(
        X=X,
        y=y,
        df_ref=df_prepared,
        test_size=test_size,
    )

    # 5) Build model
    model = build_model(X_train)

    # 6) Time-aware cross-validation on train only
    # Ojo: TSCV exige orden cronológico; como ya está ordenado, vale.
    tscv = TimeSeriesSplit(n_splits=n_splits_cv)

    scoring = {
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2",
    }

    cv_res = cross_validate(
        estimator=model,
        X=X_train,
        y=y_train,
        cv=tscv,
        scoring=scoring,
        return_train_score=True,
        n_jobs=None,
    )

    # 7) Holdout evaluation
    holdout_metrics = evaluate_split(model, X_train, y_train, X_test, y_test)

    # 8) Fit final model on full train and save
    model.fit(X_train, y_train)
    model_path = out_dir / "bayesian_ridge_quantity_sold.joblib"
    dump(model, model_path)

    # 9) Metadata útil para trazabilidad
    train_start = str(df_train["order_date"].min())
    train_end = str(df_train["order_date"].max())
    test_start = str(df_test["order_date"].min())
    test_end = str(df_test["order_date"].max())

    feature_columns = X_train.columns.tolist()

    report = {
        "problem_type": "regression",
        "model_name": "BayesianRidge",
        "target": target,
        "validation_strategy": {
            "holdout": "chronological",
            "cross_validation": "TimeSeriesSplit",
            "n_splits_cv": n_splits_cv,
            "test_size": test_size,
        },
        "date_ranges": {
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
        },
        "n_samples": {
            "full": int(len(X)),
            "train": int(len(X_train)),
            "test": int(len(X_test)),
        },
        "features": feature_columns,
        "cv_metrics": {
            "train_mae_mean": float((-cv_res["train_mae"]).mean()),
            "train_mae_std": float((-cv_res["train_mae"]).std()),
            "valid_mae_mean": float((-cv_res["test_mae"]).mean()),
            "valid_mae_std": float((-cv_res["test_mae"]).std()),
            "train_rmse_mean": float((-cv_res["train_rmse"]).mean()),
            "train_rmse_std": float((-cv_res["train_rmse"]).std()),
            "valid_rmse_mean": float((-cv_res["test_rmse"]).mean()),
            "valid_rmse_std": float((-cv_res["test_rmse"]).std()),
            "train_r2_mean": float(cv_res["train_r2"].mean()),
            "train_r2_std": float(cv_res["train_r2"].std()),
            "valid_r2_mean": float(cv_res["test_r2"].mean()),
            "valid_r2_std": float(cv_res["test_r2"].std()),
        },
        "holdout_metrics": holdout_metrics,
        "artifact_path": str(model_path),
    }

    report_path = report_dir / "metrics_baseline_quantity_sold.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Model saved to: {model_path}")
    print(f"Metrics report saved to: {report_path}")
    print(
        "Holdout metrics | "
        f"MAE={holdout_metrics['test']['mae']:.4f} | "
        f"RMSE={holdout_metrics['test']['rmse']:.4f} | "
        f"R2={holdout_metrics['test']['r2']:.4f}"
    )


if __name__ == "__main__":
    main()