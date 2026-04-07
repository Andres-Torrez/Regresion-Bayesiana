import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
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


def evaluate_holdout(model, X_train, y_train, X_test, y_test) -> dict:
    model.fit(X_train, y_train)

    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    train_mae = mean_absolute_error(y_train, pred_train)
    test_mae = mean_absolute_error(y_test, pred_test)
    train_rmse = rmse(y_train, pred_train)
    test_rmse = rmse(y_test, pred_test)

    return {
        "train": {
            "mae": float(train_mae),
            "rmse": float(train_rmse),
            "r2": float(r2_score(y_train, pred_train)),
            "mbe": mean_bias_error(y_train, pred_train),
        },
        "test": {
            "mae": float(test_mae),
            "rmse": float(test_rmse),
            "r2": float(r2_score(y_test, pred_test)),
            "mbe": mean_bias_error(y_test, pred_test),
        },
        "overfitting": {
            "mae_gap_ratio": float((test_mae - train_mae) / train_mae) if train_mae > 0 else None,
            "rmse_gap_ratio": float((test_rmse - train_rmse) / train_rmse) if train_rmse > 0 else None,
        },
    }


def build_models(X_train: pd.DataFrame) -> dict:
    preprocessor = build_preprocessing_pipeline(X_train)

    models = {
        "dummy_mean": Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", DummyRegressor(strategy="mean")),
            ]
        ),
        "bayesian_ridge": Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", BayesianRidge()),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", RandomForestRegressor(
                    n_estimators=300,
                    max_depth=None,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    random_state=42,
                    n_jobs=-1,
                )),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", HistGradientBoostingRegressor(
                    learning_rate=0.05,
                    max_iter=300,
                    max_depth=8,
                    min_samples_leaf=20,
                    l2_regularization=0.1,
                    random_state=42,
                )),
            ]
        ),
    }

    return models


def summarize_cv(cv_res: dict) -> dict:
    return {
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
    }


def main(
    data_path: str = "data/raw/amazon_sales_dataset.csv",
    target: str = "quantity_sold",
    report_dir: str = "reports/benchmark",
    test_size: float = 0.2,
    n_splits_cv: int = 5,
):
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    # Load and prepare data
    df = load_data(data_path)
    df = clean_data(df)

    if "order_date" not in df.columns:
        raise ValueError("The dataset must contain 'order_date' for time-aware validation.")

    df = df.sort_values("order_date").reset_index(drop=True)

    X, y, df_prepared = prepare_features(df, target=target)

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    df_prepared = df_prepared.reset_index(drop=True)

    X_train, X_test, y_train, y_test, df_train, df_test = chronological_train_test_split(
        X=X,
        y=y,
        df_ref=df_prepared,
        test_size=test_size,
    )

    tscv = TimeSeriesSplit(n_splits=n_splits_cv)
    scoring = {
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2",
    }

    models = build_models(X_train)

    results = {}
    summary_rows = []

    for model_name, model in models.items():
        print(f"Evaluating: {model_name}")

        cv_res = cross_validate(
            estimator=model,
            X=X_train,
            y=y_train,
            cv=tscv,
            scoring=scoring,
            return_train_score=True,
            n_jobs=None,
        )

        holdout_metrics = evaluate_holdout(model, X_train, y_train, X_test, y_test)
        cv_summary = summarize_cv(cv_res)

        results[model_name] = {
            "cv_metrics": cv_summary,
            "holdout_metrics": holdout_metrics,
        }

        summary_rows.append({
            "model": model_name,
            "cv_valid_mae_mean": cv_summary["valid_mae_mean"],
            "cv_valid_rmse_mean": cv_summary["valid_rmse_mean"],
            "cv_valid_r2_mean": cv_summary["valid_r2_mean"],
            "holdout_test_mae": holdout_metrics["test"]["mae"],
            "holdout_test_rmse": holdout_metrics["test"]["rmse"],
            "holdout_test_r2": holdout_metrics["test"]["r2"],
            "holdout_rmse_gap_ratio": holdout_metrics["overfitting"]["rmse_gap_ratio"],
        })

    # Save detailed results
    with open(report_dir / "benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Save summary table
    summary_df = pd.DataFrame(summary_rows).sort_values(
        by=["holdout_test_rmse", "holdout_test_mae"],
        ascending=True,
    )
    summary_df.to_csv(report_dir / "benchmark_summary.csv", index=False)

    print(f"Detailed benchmark results saved to: {report_dir / 'benchmark_results.json'}")
    print(f"Benchmark summary saved to: {report_dir / 'benchmark_summary.csv'}")
    print("\nBenchmark summary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()