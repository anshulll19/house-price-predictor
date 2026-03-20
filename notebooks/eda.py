"""
eda.py
------
Exploratory Data Analysis on the housing dataset.
Generates and saves plots to outputs/eda/.

Usage:
    python notebooks/eda.py
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.preprocessing import load_data, clean_data, TARGET

# ── Setup ─────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
OUT_DIR = Path(__file__).parent.parent / "outputs" / "eda"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEP = "─" * 55


def save(fig, name):
    path = OUT_DIR / f"{name}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {path.name}")


# ── Load ──────────────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print("  🔍  House Price — Exploratory Data Analysis")
print(f"{'='*55}\n")

df_raw = load_data()
df     = clean_data(df_raw)

print(f"Shape (raw)    : {df_raw.shape}")
print(f"Shape (cleaned): {df.shape}")
print(f"\n{SEP}")
print("Dtypes & nulls:")
print(f"{SEP}")
print(df.dtypes.to_string())
print("\nNull counts:")
print(df.isnull().sum()[df.isnull().sum() > 0].to_string() or "  None — dataset is complete ✓")
print(f"\n{SEP}")
print("Descriptive statistics (numeric):")
print(f"{SEP}")
print(df.describe().round(2).to_string())


# ── 1. Price distribution ─────────────────────────────────────────────────────
print(f"\n{SEP}\nGenerating plots …\n{SEP}")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].hist(df[TARGET], bins=60, color="#4C72B0", edgecolor="white")
axes[0].set_xlabel("Price ($)")
axes[0].set_ylabel("Count")
axes[0].set_title("Price Distribution")

axes[1].hist(np.log1p(df[TARGET]), bins=60, color="#DD8452", edgecolor="white")
axes[1].set_xlabel("log(Price + 1)")
axes[1].set_ylabel("Count")
axes[1].set_title("log(Price) Distribution")

fig.suptitle("Target Variable — House Sale Price", fontsize=14, fontweight="bold")
plt.tight_layout()
save(fig, "01_price_distribution")


# ── 2. Price by neighborhood ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
order = df.groupby("neighborhood")[TARGET].median().sort_values(ascending=False).index
sns.boxplot(data=df, x="neighborhood", y=TARGET, order=order, palette="Set2", ax=ax)
ax.set_title("Price by Neighborhood", fontsize=13)
ax.set_xlabel("Neighborhood")
ax.set_ylabel("Price ($)")
plt.tight_layout()
save(fig, "02_price_by_neighborhood")


# ── 3. Correlation heatmap ────────────────────────────────────────────────────
NUM_COLS = [
    "price", "sqft_living", "grade", "condition", "school_rating",
    "bedrooms", "bathrooms", "age", "crime_rate", "dist_city_center",
    "luxury_score" if "luxury_score" in df.columns else "garage_spaces",
    "location_score" if "location_score" in df.columns else "has_pool",
]
NUM_COLS = [c for c in NUM_COLS if c in df.columns]

corr = df[NUM_COLS].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))

fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(
    corr, mask=mask, annot=True, fmt=".2f",
    cmap="coolwarm", center=0,
    linewidths=0.4, ax=ax,
)
ax.set_title("Feature Correlation Heatmap", fontsize=13)
plt.tight_layout()
save(fig, "03_correlation_heatmap")


# ── 4. Scatter: sqft_living vs price ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
sc = ax.scatter(
    df["sqft_living"], df[TARGET],
    c=df["grade"], cmap="viridis",
    alpha=0.3, s=12, edgecolors="none",
)
plt.colorbar(sc, ax=ax, label="Grade")
ax.set_xlabel("Living Area (sqft)")
ax.set_ylabel("Price ($)")
ax.set_title("Living Area vs Price (coloured by Grade)")
plt.tight_layout()
save(fig, "04_sqft_vs_price")


# ── 5. Price by bedrooms ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
sns.violinplot(data=df, x="bedrooms", y=TARGET, palette="pastel", ax=ax)
ax.set_title("Price Distribution by Bedroom Count")
ax.set_xlabel("Bedrooms")
ax.set_ylabel("Price ($)")
plt.tight_layout()
save(fig, "05_price_by_bedrooms")


# ── 6. Age vs price ───────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
ax.scatter(df["age"], df[TARGET], alpha=0.2, s=12, color="#55A868", edgecolors="none")
ax.set_xlabel("House Age (years)")
ax.set_ylabel("Price ($)")
ax.set_title("House Age vs Price")
plt.tight_layout()
save(fig, "06_age_vs_price")


# ── 7. Feature distributions ──────────────────────────────────────────────────
FD_COLS = ["sqft_living", "sqft_lot", "age", "school_rating", "crime_rate", "grade", "condition", "dist_city_center"]
FD_COLS = [c for c in FD_COLS if c in df.columns]

fig, axes = plt.subplots(2, 4, figsize=(18, 8))
for ax, col in zip(axes.ravel(), FD_COLS):
    ax.hist(df[col].dropna(), bins=40, color="#4C72B0", edgecolor="white", alpha=0.85)
    ax.set_title(col)
    ax.set_ylabel("Count")

fig.suptitle("Feature Distributions", fontsize=14, fontweight="bold")
plt.tight_layout()
save(fig, "07_feature_distributions")


# ── 8. Pool & basement impact ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, feat, label in zip(
    axes,
    ["has_pool", "has_basement"],
    ["Pool", "Basement"],
):
    df[feat + "_label"] = df[feat].map({0: f"No {label}", 1: f"Has {label}"})
    sns.boxplot(data=df, x=feat + "_label", y=TARGET, palette="Set2", ax=ax)
    ax.set_title(f"Price by {label} Presence")
    ax.set_xlabel("")
    ax.set_ylabel("Price ($)")

plt.tight_layout()
save(fig, "08_amenity_impact")


print(f"\n✅  All EDA plots saved to {OUT_DIR}\n")
