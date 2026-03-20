"""
preprocessing.py
----------------
Handles data loading, cleaning, encoding, and train/test splitting for
the Indian housing dataset.
Produces a scikit-learn ColumnTransformer that can be embedded in a Pipeline.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


# ── Feature Definitions ────────────────────────────────────────────────────────
TARGET = "price"

CATEGORICAL_FEATURES = ["city", "locality_tier", "furnishing"]

NUMERIC_FEATURES = [
    "area_sqft",
    "bhk",
    "bathrooms",
    "floor",
    "total_floors",
    "parking",
    "lift",
    "east_facing",
    "property_age",
]


def load_data(data_path: str | Path = None) -> pd.DataFrame:
    """Load the Indian housing CSV. Defaults to data/housing.csv."""
    if data_path is None:
        data_path = Path(__file__).parent.parent / "data" / "housing.csv"
    df = pd.read_csv(data_path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: remove outlier prices, fix dtypes."""
    df = df.copy()

    # Drop rows with missing target
    df.dropna(subset=[TARGET], inplace=True)

    # Remove extreme price outliers (4-sigma)
    mu, sigma = df[TARGET].mean(), df[TARGET].std()
    df = df[df[TARGET].between(mu - 4 * sigma, mu + 4 * sigma)]

    # Clip area
    df["area_sqft"] = df["area_sqft"].clip(100, 10_000)

    # Ensure floor <= total_floors
    df["floor"] = df.apply(
        lambda r: min(r["floor"], r["total_floors"]), axis=1
    )

    # Non-negative age
    df["property_age"] = df["property_age"].clip(0, 60)

    return df.reset_index(drop=True)


def split_data(df: pd.DataFrame, test_size: float = 0.20, random_state: int = 42):
    """Split into train / test sets."""
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def build_preprocessor() -> ColumnTransformer:
    """Return a ColumnTransformer that imputes + scales numerics and OHE categoricals."""
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe",     OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline,     NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ], remainder="drop")

    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Extract feature names after fitting the ColumnTransformer."""
    num_names = NUMERIC_FEATURES
    cat_names = list(
        preprocessor.named_transformers_["cat"]["ohe"].get_feature_names_out(CATEGORICAL_FEATURES)
    )
    return num_names + cat_names
