import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_DEFAULT = "quantity_sold"

FORBIDDEN_FEATURES = {
    "quantity_sold",
    "discounted_price",
    "total_revenue",
    "order_id",
}


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "order_date" in df.columns:
        df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")

    numeric_cols = [
        "price",
        "discount_percent",
        "quantity_sold",
        "rating",
        "review_count",
        "discounted_price",
        "total_revenue",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def drop_missing_target(df: pd.DataFrame, target: str = TARGET_DEFAULT) -> pd.DataFrame:
    df = df.copy()
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataframe.")
    return df.dropna(subset=[target])


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "order_date" in df.columns:
        df["order_year"] = df["order_date"].dt.year
        df["order_month"] = df["order_date"].dt.month
        df["order_day"] = df["order_date"].dt.day
        df["order_dayofweek"] = df["order_date"].dt.dayofweek
        df["order_quarter"] = df["order_date"].dt.quarter
        df["is_weekend"] = (df["order_date"].dt.dayofweek >= 5).astype(int)

    return df


def get_feature_columns(df: pd.DataFrame, target: str = TARGET_DEFAULT) -> list[str]:
    candidate_features = [
        "price",
        "discount_percent",
        "rating",
        "review_count",
        "product_category",
        "customer_region",
        "payment_method",
        "order_year",
        "order_month",
        "order_day",
        "order_dayofweek",
        "order_quarter",
        "is_weekend",
    ]

    feature_cols = [
        col for col in candidate_features
        if col in df.columns and col not in FORBIDDEN_FEATURES and col != target
    ]

    if not feature_cols:
        raise ValueError("No valid feature columns found after filtering.")

    return feature_cols


def prepare_features(df: pd.DataFrame, target: str = TARGET_DEFAULT):
    df = df.copy()

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataframe.")

    df = drop_missing_target(df, target=target)
    df = add_time_features(df)

    feature_cols = get_feature_columns(df, target=target)

    X = df[feature_cols].copy()
    y = df[target].copy()

    return X, y, df


def build_preprocessing_pipeline(X: pd.DataFrame):
    numeric_features = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

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
        ],
        remainder="drop"
    )

    return preprocessor