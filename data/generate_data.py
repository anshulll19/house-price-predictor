"""
generate_data.py
----------------
Generates a realistic synthetic Indian housing dataset and saves it as
data/housing.csv.  Features mirror real-world Indian property listings
(BHK, INR pricing, city, locality tier, amenities, etc.).
"""

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

N = 8_000  # number of samples

# ── Indian Cities & Price Multipliers ─────────────────────────────────────────
# Base price per sqft in INR for a "Suburbs / Mid" locality
CITY_BASE_PRICE_PER_SQFT = {
    "Mumbai":    18_000,
    "Delhi":     12_000,
    "Bangalore": 9_500,
    "Hyderabad": 7_500,
    "Chennai":   7_000,
    "Pune":      7_200,
    "Kolkata":   5_500,
    "Ahmedabad": 5_000,
    "Noida":     6_000,
    "Jaipur":    4_500,
}

CITY_WEIGHTS = [0.18, 0.16, 0.15, 0.12, 0.08, 0.10, 0.07, 0.05, 0.06, 0.03]

LOCALITY_TIER_MULTIPLIER = {
    "Premium": 1.45,
    "Mid":     1.00,
    "Budget":  0.65,
}

LOCALITY_TIER_WEIGHTS = [0.20, 0.50, 0.30]  # Premium, Mid, Budget

# ── Sample cities & locality tiers ───────────────────────────────────────────
cities        = np.random.choice(list(CITY_BASE_PRICE_PER_SQFT.keys()), size=N, p=CITY_WEIGHTS)
locality_tier = np.random.choice(list(LOCALITY_TIER_MULTIPLIER.keys()), size=N, p=LOCALITY_TIER_WEIGHTS)

# ── Physical Attributes ───────────────────────────────────────────────────────
bhk       = np.random.choice([1, 2, 3, 4, 5], size=N, p=[0.10, 0.35, 0.35, 0.15, 0.05])
bathrooms = np.clip(bhk - 1 + np.random.randint(0, 2, size=N), 1, 5)

# Area: 300–700 sqft per BHK, with random variance
area_sqft = (bhk * 500 + np.random.normal(0, 120, N)).clip(250, 5000).astype(int)

# Floor info
total_floors = np.clip(np.random.geometric(p=0.12, size=N) + 1, 2, 24)
floor = np.array([np.random.randint(0, t + 1) for t in total_floors])

# ── Amenities ─────────────────────────────────────────────────────────────────
parking     = np.random.choice([0, 1], size=N, p=[0.30, 0.70])
lift        = np.where(total_floors >= 4,
                       np.random.choice([0, 1], size=N, p=[0.15, 0.85]),
                       np.random.choice([0, 1], size=N, p=[0.70, 0.30]))
east_facing = np.random.choice([0, 1], size=N, p=[0.60, 0.40])

furnishing  = np.random.choice(
    ["Unfurnished", "Semi-Furnished", "Fully Furnished"],
    size=N,
    p=[0.35, 0.40, 0.25],
)

FURNISHING_MULTIPLIER = {"Unfurnished": 1.00, "Semi-Furnished": 1.08, "Fully Furnished": 1.18}

# ── Property Age ──────────────────────────────────────────────────────────────
property_age = np.clip(np.random.poisson(lam=8, size=N), 0, 40).astype(int)

# ── Price Calculation ─────────────────────────────────────────────────────────
base_price_per_sqft = np.array([CITY_BASE_PRICE_PER_SQFT[c] for c in cities])
locality_mult        = np.array([LOCALITY_TIER_MULTIPLIER[t] for t in locality_tier])
furnishing_mult      = np.array([FURNISHING_MULTIPLIER[f]   for f in furnishing])

effective_price_per_sqft = base_price_per_sqft * locality_mult

price = (
    effective_price_per_sqft * area_sqft
    + parking     * base_price_per_sqft * 30          # parking adds ~30 sqft value
    + lift        * 50_000
    + east_facing * base_price_per_sqft * 15
    - property_age * base_price_per_sqft * 0.8        # depreciation per year
    + (floor / np.maximum(total_floors, 1)) * base_price_per_sqft * 40  # higher floor premium
) * furnishing_mult

# Add market noise
noise = np.random.normal(0, effective_price_per_sqft * 25, N)
price = (price + noise).clip(5_00_000, 15_00_00_000).astype(int)  # ₹5L – ₹15Cr

# ── Assemble DataFrame ────────────────────────────────────────────────────────
df = pd.DataFrame({
    "price":         price,
    "city":          cities,
    "locality_tier": locality_tier,
    "area_sqft":     area_sqft,
    "bhk":           bhk,
    "bathrooms":     bathrooms,
    "floor":         floor,
    "total_floors":  total_floors,
    "parking":       parking,
    "lift":          lift,
    "east_facing":   east_facing,
    "furnishing":    furnishing,
    "property_age":  property_age,
})

output_path = Path(__file__).parent / "housing.csv"
df.to_csv(output_path, index=False)
print(f"✅  Indian housing dataset generated: {output_path}  ({len(df):,} rows × {df.shape[1]} columns)")
print(df.describe(include="all").round(2))
