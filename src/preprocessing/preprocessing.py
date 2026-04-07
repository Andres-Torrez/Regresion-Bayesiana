import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- Convert types safely ---
    # Dates
    if "order_date" in df.columns:
        df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")

    # Numeric columns (some datasets come as strings)
    numeric_cols = [
        "price",
        "discount_percent",
        "quantity_sold",
        "rating",
        "review_count",
        "discounted_price",
        "total_revenue",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop rows where target is missing
    df = df.dropna(subset=["discounted_price"])

    return df


def prepare_features(df: pd.DataFrame, target: str = "discounted_price"):
    # Select only columns that exist in your dataset
    feature_cols = [
        "price",
        "discount_percent",
        "quantity_sold",
        "rating",
        "review_count",
        "product_category",
        "customer_region",
        "payment_method",
    ]

    missing = [c for c in feature_cols + [target] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in df: {missing}")

    X = df[feature_cols].copy()
    y = df[target].copy()

    # Optional: add time-based features if order_date exists
    if "order_date" in df.columns:
        X["order_month"] = df["order_date"].dt.month
        X["order_dayofweek"] = df["order_date"].dt.dayofweek

    return X, y


def build_preprocessing_pipeline(X: pd.DataFrame):
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = X.select_dtypes(include=["object", "category"]).columns

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor


def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)