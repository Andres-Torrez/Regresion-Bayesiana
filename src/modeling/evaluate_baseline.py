import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import load

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.preprocessing.preprocessing import (
    load_data,
    clean_data,
    prepare_features,
)


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


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

    X_train = X.iloc[:split_idx].copy()
    X_test = X.iloc[split_idx:].copy()
    y_train = y.iloc[:split_idx].copy()
    y_test = y.iloc[split_idx:].copy()
    df_train = df_ref.iloc[:split_idx].copy()
    df_test = df_ref.iloc[split_idx:].copy()

    return X_train, X_test, y_train, y_test, df_train, df_test


def save_plot(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(
    data_path: str = "data/raw/amazon_sales_dataset.csv",
    model_path: str = "models/baseline/bayesian_ridge_quantity_sold.joblib",
    report_dir: str = "reports/evaluation",
    test_size: float = 0.2,
):
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load data
    df = load_data(data_path)
    df = clean_data(df)

    if "order_date" not in df.columns:
        raise ValueError("The dataset must contain 'order_date'.")

    df = df.sort_values("order_date").reset_index(drop=True)

    # 2) Prepare features/target
    X, y, df_prepared = prepare_features(df, target="quantity_sold")

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    df_prepared = df_prepared.reset_index(drop=True)

    # 3) Rebuild same temporal split
    X_train, X_test, y_train, y_test, df_train, df_test = chronological_train_test_split(
        X=X,
        y=y,
        df_ref=df_prepared,
        test_size=test_size,
    )

    # 4) Load trained model
    model = load(model_path)

    # 5) Predict
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    # 6) Residuals
    residuals_train = y_train - pred_train
    residuals_test = y_test - pred_test
    abs_errors_test = np.abs(residuals_test)

    # 7) Metrics
    metrics = {
        "train": {
            "mae": float(mean_absolute_error(y_train, pred_train)),
            "rmse": rmse(y_train, pred_train),
            "r2": float(r2_score(y_train, pred_train)),
            "residual_mean": float(np.mean(residuals_train)),
            "residual_std": float(np.std(residuals_train)),
        },
        "test": {
            "mae": float(mean_absolute_error(y_test, pred_test)),
            "rmse": rmse(y_test, pred_test),
            "r2": float(r2_score(y_test, pred_test)),
            "residual_mean": float(np.mean(residuals_test)),
            "residual_std": float(np.std(residuals_test)),
            "p90_abs_error": float(np.percentile(abs_errors_test, 90)),
            "p95_abs_error": float(np.percentile(abs_errors_test, 95)),
            "max_abs_error": float(np.max(abs_errors_test)),
        },
    }

    with open(report_dir / "baseline_residual_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # 8) Plot: actual vs predicted
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    ax1.scatter(y_test, pred_test, alpha=0.25)
    min_val = min(y_test.min(), pred_test.min())
    max_val = max(y_test.max(), pred_test.max())
    ax1.plot([min_val, max_val], [min_val, max_val], linestyle="--")
    ax1.set_title("Actual vs Predicted (Test)")
    ax1.set_xlabel("Actual quantity_sold")
    ax1.set_ylabel("Predicted quantity_sold")
    save_plot(fig1, report_dir / "actual_vs_predicted_test.png")

    # 9) Plot: residuals vs predictions
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.scatter(pred_test, residuals_test, alpha=0.25)
    ax2.axhline(0, linestyle="--")
    ax2.set_title("Residuals vs Predicted (Test)")
    ax2.set_xlabel("Predicted quantity_sold")
    ax2.set_ylabel("Residuals")
    save_plot(fig2, report_dir / "residuals_vs_predicted_test.png")

    # 10) Plot: residual distribution
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    ax3.hist(residuals_test, bins=40)
    ax3.axvline(0, linestyle="--")
    ax3.set_title("Residual Distribution (Test)")
    ax3.set_xlabel("Residual")
    ax3.set_ylabel("Frequency")
    save_plot(fig3, report_dir / "residual_distribution_test.png")

    # 11) Top worst predictions
    error_df = df_test.copy()
    error_df["actual_quantity_sold"] = y_test.values
    error_df["predicted_quantity_sold"] = pred_test
    error_df["residual"] = residuals_test
    error_df["abs_error"] = abs_errors_test
    error_df = error_df.sort_values("abs_error", ascending=False)

    error_df.head(50).to_csv(report_dir / "top_50_worst_predictions.csv", index=False)

    print(f"Residual metrics saved to: {report_dir / 'baseline_residual_metrics.json'}")
    print(f"Plots saved under: {report_dir}")
    print(f"Top error cases saved to: {report_dir / 'top_50_worst_predictions.csv'}")


if __name__ == "__main__":
    main()