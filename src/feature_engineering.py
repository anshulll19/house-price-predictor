"""
feature_engineering.py
-----------------------
India-specific derived features built on top of the raw housing DataFrame.
Call `engineer_features(df)` before preprocessing.
"""

import pandas as pd
import numpy as np


# City → Tier mapping (used as an ordinal feature)
CITY_TIER_MAP = {
    "Mumbai":    1,
    "Delhi":     1,
    "Bangalore": 1,
    "Hyderabad": 2,
    "Chennai":   2,
    "Pune":      2,
    "Kolkata":   2,
    "Ahmedabad": 3,
    "Noida":     2,
    "Jaipur":    3,
}


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate India-specific derived features from raw housing data.

    New columns added
    -----------------
    city_tier          – 1 = Metro, 2 = Tier-2, 3 = Tier-3
    area_per_bhk       – area_sqft / bhk  (space efficiency)
    floor_ratio        – floor / total_floors  (view premium proxy)
    is_top_floor       – 1 if on the topmost floor
    is_ground_floor    – 1 if ground floor (0)
    amenity_score      – parking + lift + east_facing (0–3)
    furnishing_score   – Unfurnished=0, Semi=1, Fully=2 (ordinal)
    age_bucket         – binned property age (0–4)
    area_x_bhk         – interaction term
    bath_per_bhk       – bathrooms / bhk
    is_new_property    – property_age <= 3
    locality_score     – Premium=2, Mid=1, Budget=0 (ordinal)
    """
    df = df.copy()

    # City tier
    df["city_tier"] = df["city"].map(CITY_TIER_MAP).fillna(2).astype(int)

    # Space efficiency
    df["area_per_bhk"] = (df["area_sqft"] / df["bhk"].replace(0, 1)).round(1)

    # Floor features
    safe_total = df["total_floors"].replace(0, 1)
    df["floor_ratio"]       = (df["floor"] / safe_total).round(3)
    df["is_top_floor"]      = (df["floor"] == df["total_floors"]).astype(int)
    df["is_ground_floor"]   = (df["floor"] == 0).astype(int)

    # Amenity composite
    df["amenity_score"] = df["parking"] + df["lift"] + df["east_facing"]

    # Furnishing ordinal
    furn_map = {"Unfurnished": 0, "Semi-Furnished": 1, "Fully Furnished": 2}
    df["furnishing_score"] = df["furnishing"].map(furn_map).fillna(0).astype(int)

    # Age bucket
    df["age_bucket"] = pd.cut(
        df["property_age"],
        bins=[-1, 3, 10, 20, 35, 100],
        labels=[0, 1, 2, 3, 4],
    ).astype(float)

    # Interaction
    df["area_x_bhk"]  = df["area_sqft"] * df["bhk"]
    df["bath_per_bhk"] = (df["bathrooms"] / df["bhk"].replace(0, 1)).round(2)

    # New property flag
    df["is_new_property"] = (df["property_age"] <= 3).astype(int)

    # Locality tier ordinal
    loc_map = {"Premium": 2, "Mid": 1, "Budget": 0}
    df["locality_score"] = df["locality_tier"].map(loc_map).fillna(1).astype(int)

    return df


# ── Extended numeric feature list (includes engineered columns) ────────────────
ENGINEERED_NUMERIC_FEATURES = [
    # raw
    "area_sqft", "bhk", "bathrooms", "floor", "total_floors",
    "parking", "lift", "east_facing", "property_age",
    # engineered
    "city_tier", "area_per_bhk", "floor_ratio", "is_top_floor",
    "is_ground_floor", "amenity_score", "furnishing_score",
    "age_bucket", "area_x_bhk", "bath_per_bhk",
    "is_new_property", "locality_score",
]
