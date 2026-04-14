import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.preprocessing.preprocessing import load_data, clean_data


def main(
    data_path: str = "data/raw/amazon_sales_dataset.csv",
    report_dir: str = "reports/eda",
):
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load
    df = load_data(data_path)
    df = clean_data(df)

    print("\n===== BASIC INFO =====")
    print(df.shape)
    print(df.dtypes)

    # =========================
    # TARGET ANALYSIS
    # =========================
    target = "quantity_sold"

    print("\n===== TARGET DISTRIBUTION =====")
    print(df[target].describe())

    print("\nUnique values:", df[target].nunique())
    print("\nValue counts:")
    print(df[target].value_counts().sort_index())

    # Histogram
    plt.figure()
    df[target].hist(bins=30)
    plt.title("Distribution of quantity_sold")
    plt.savefig(report_dir / "target_distribution.png")
    plt.close()

    # =========================
    # NUMERIC RELATIONSHIPS
    # =========================
    numeric_cols = ["price", "discount_percent", "rating", "review_count"]

    print("\n===== CORRELATIONS =====")
    corr = df[numeric_cols + [target]].corr()
    print(corr[target].sort_values(ascending=False))

    # Scatter plots
    for col in numeric_cols:
        plt.figure()
        plt.scatter(df[col], df[target], alpha=0.2)
        plt.xlabel(col)
        plt.ylabel(target)
        plt.title(f"{col} vs {target}")
        plt.savefig(report_dir / f"{col}_vs_target.png")
        plt.close()

    # =========================
    # CATEGORICAL ANALYSIS
    # =========================
    categorical_cols = [
        "product_category",
        "customer_region",
        "payment_method",
    ]

    for col in categorical_cols:
        print(f"\n===== TARGET MEAN BY {col} =====")
        grouped = df.groupby(col)[target].mean().sort_values(ascending=False)
        print(grouped)

        # Plot
        plt.figure()
        grouped.plot(kind="bar")
        plt.title(f"Mean {target} by {col}")
        plt.savefig(report_dir / f"{col}_mean_target.png")
        plt.close()

    # =========================
    # TIME ANALYSIS
    # =========================
    if "order_date" in df.columns:
        df["order_date"] = pd.to_datetime(df["order_date"])

        df["year_month"] = df["order_date"].dt.to_period("M")

        monthly = df.groupby("year_month")[target].mean()

        print("\n===== TIME TREND =====")
        print(monthly.tail(10))

        plt.figure()
        monthly.plot()
        plt.title("Monthly average quantity_sold")
        plt.savefig(report_dir / "time_trend.png")
        plt.close()

    # =========================
    # SIMPLE BASELINES
    # =========================
    print("\n===== GROUP BASELINES =====")

    baseline_category = df.groupby("product_category")[target].mean()
    df["cat_pred"] = df["product_category"].map(baseline_category)

    mae_cat = np.mean(np.abs(df[target] - df["cat_pred"]))
    print(f"MAE using product_category mean: {mae_cat:.4f}")

    baseline_region = df.groupby("customer_region")[target].mean()
    df["region_pred"] = df["customer_region"].map(baseline_region)

    mae_region = np.mean(np.abs(df[target] - df["region_pred"]))
    print(f"MAE using customer_region mean: {mae_region:.4f}")

    print("\nEDA diagnosis completed. Check reports/eda folder.")


if __name__ == "__main__":
    main()