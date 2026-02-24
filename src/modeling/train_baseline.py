import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump

from sklearn.pipeline import Pipeline
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.preprocessing.preprocessing import load_data, clean_data, prepare_features, build_preprocessing_pipeline


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate_holdout(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)

    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    metrics = {
        "train": {
            "rmse": rmse(y_train, pred_train),
            "mae": float(mean_absolute_error(y_train, pred_train)),
            "r2": float(r2_score(y_train, pred_train)),
        },
        "test": {
            "rmse": rmse(y_test, pred_test),
            "mae": float(mean_absolute_error(y_test, pred_test)),
            "r2": float(r2_score(y_test, pred_test)),
        }
    }

    # overfitting gap on RMSE
    tr = metrics["train"]["rmse"]
    te = metrics["test"]["rmse"]
    metrics["rmse_gap_ratio"] = float((te - tr) / tr) if tr > 0 else None
    return metrics


def main(
    data_path: str = "data/raw/amazon_sales_dataset.csv",
    target: str = "discounted_price",
    use_log_target: bool = False,
    out_dir: str = "models",
    report_dir: str = "reports",
    seed: int = 42,
):
    out_dir = Path(out_dir)
    report_dir = Path(report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(data_path)
    df = clean_data(df)

    X, y = prepare_features(df, target=target)

    # Optional: log-transform the target (recommended if target is skewed)
    if use_log_target:
        y = y.clip(lower=1e-9)
        y = np.log(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    preprocessor = build_preprocessing_pipeline(X)

    model = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", BayesianRidge())
    ])

    # Cross-validation (KFold)
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    scoring = {
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
        "r2": "r2",
    }
    cv_res = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring, return_train_score=True)

    # Holdout evaluation
    holdout_metrics = evaluate_holdout(model, X_train, y_train, X_test, y_test)

    # Save model
    model_name = "bayesridge_logtarget.joblib" if use_log_target else "bayesridge.joblib"
    dump(model, out_dir / model_name)

    # Save metrics report
    report = {
        "target": target,
        "use_log_target": use_log_target,
        "cv": {
            "train_rmse_mean": float((-cv_res["train_rmse"]).mean()),
            "train_rmse_std": float((-cv_res["train_rmse"]).std()),
            "test_rmse_mean": float((-cv_res["test_rmse"]).mean()),
            "test_rmse_std": float((-cv_res["test_rmse"]).std()),
            "train_mae_mean": float((-cv_res["train_mae"]).mean()),
            "test_mae_mean": float((-cv_res["test_mae"]).mean()),
            "train_r2_mean": float((cv_res["train_r2"]).mean()),
            "test_r2_mean": float((cv_res["test_r2"]).mean()),
        },
        "holdout": holdout_metrics,
        "artifact": str(out_dir / model_name),
    }

    with open(report_dir / "metrics_baseline.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("Saved:", out_dir / model_name)
    print("Report:", report_dir / "metrics_baseline.json")
    print("Holdout RMSE gap ratio:", report["holdout"]["rmse_gap_ratio"])


if __name__ == "__main__":
    main(use_log_target=True)