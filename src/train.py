"""
train.py
--------
Trains Linear Regression (Ridge), Random Forest, and XGBoost models on the
Indian housing dataset.  Evaluates each model, saves the best one, and
writes per-city price statistics for the Streamlit app's insights panel.

Usage:
    python src/train.py
"""

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import (
    load_data, clean_data, split_data, build_preprocessor,
    NUMERIC_FEATURES, CATEGORICAL_FEATURES,
)
from src.feature_engineering import engineer_features, ENGINEERED_NUMERIC_FEATURES

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent.parent
MODEL_DIR = ROOT / "models"
OUT_DIR   = ROOT / "outputs"
MODEL_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)


# ── INR formatting helpers ────────────────────────────────────────────────────
def inr_format(value: float) -> str:
    """Format a price in INR using Indian number system (Lakhs / Crores)."""
    if value >= 1_00_00_000:
        return f"₹{value/1_00_00_000:.2f} Cr"
    elif value >= 1_00_000:
        return f"₹{value/1_00_000:.2f} L"
    else:
        return f"₹{value:,.0f}"


# ── Metric helpers ────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred) -> dict:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    return {"RMSE": round(rmse, 2), "MAE": round(mae, 2), "R2": round(r2, 4)}


def print_metrics(name: str, metrics: dict):
    print(f"\n{'─'*50}")
    print(f"  {name}")
    print(f"{'─'*50}")
    print(f"  RMSE : {inr_format(metrics['RMSE']):>20}")
    print(f"  MAE  : {inr_format(metrics['MAE']):>20}")
    print(f"  R²   : {metrics['R2']:>20.4f}")


# ── Plot helpers ──────────────────────────────────────────────────────────────
def plot_actual_vs_predicted(y_test, y_pred, name: str, out_dir: Path):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_test, y_pred, alpha=0.35, edgecolors="none", color="#4C72B0", s=18)
    lo = min(y_test.min(), y_pred.min())
    hi = max(y_test.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="Perfect fit")
    ax.set_xlabel("Actual Price (₹)", fontsize=12)
    ax.set_ylabel("Predicted Price (₹)", fontsize=12)
    ax.set_title(f"{name} — Actual vs Predicted", fontsize=14)
    ax.legend()
    plt.tight_layout()
    safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    fig.savefig(out_dir / f"{safe_name}_actual_vs_pred.png", dpi=120)
    plt.close(fig)


def plot_residuals(y_test, y_pred, name: str, out_dir: Path):
    residuals = y_test - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(y_pred, residuals, alpha=0.35, color="#DD8452", edgecolors="none", s=18)
    axes[0].axhline(0, color="k", lw=1, ls="--")
    axes[0].set_xlabel("Predicted Price (₹)")
    axes[0].set_ylabel("Residual (₹)")
    axes[0].set_title("Residuals vs Predicted")

    axes[1].hist(residuals, bins=60, color="#55A868", edgecolor="white")
    axes[1].axvline(0, color="k", lw=1, ls="--")
    axes[1].set_xlabel("Residual (₹)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Residual Distribution")

    fig.suptitle(f"{name} — Residual Analysis", fontsize=14)
    plt.tight_layout()
    safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    fig.savefig(out_dir / f"{safe_name}_residuals.png", dpi=120)
    plt.close(fig)


def plot_feature_importance(model_pipeline, feature_names: list[str], name: str, out_dir: Path, top_n: int = 20):
    try:
        estimator = model_pipeline.named_steps["model"]
        importances = estimator.feature_importances_
    except AttributeError:
        return  # Ridge – skip
    fi = pd.Series(importances, index=feature_names[:len(importances)]).nlargest(top_n)
    fig, ax = plt.subplots(figsize=(9, 6))
    fi.sort_values().plot.barh(ax=ax, color="#4C72B0")
    ax.set_title(f"{name} — Top {top_n} Feature Importances", fontsize=13)
    ax.set_xlabel("Importance")
    plt.tight_layout()
    safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    fig.savefig(out_dir / f"{safe_name}_feature_importance.png", dpi=120)
    plt.close(fig)


def plot_model_comparison(all_metrics: dict, out_dir: Path):
    names   = list(all_metrics.keys())
    metrics = ["RMSE", "MAE", "R2"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    for i, metric in enumerate(metrics):
        vals = [all_metrics[n][metric] for n in names]
        bars = axes[i].bar(names, vals, color=colors)
        axes[i].set_title(metric, fontsize=13)
        axes[i].set_xticklabels(names, rotation=15, ha="right")
        for bar, val in zip(bars, vals):
            label = inr_format(val) if metric != "R2" else f"{val:.4f}"
            axes[i].text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() * 1.01, label,
                         ha="center", va="bottom", fontsize=8)
    fig.suptitle("Model Comparison", fontsize=15, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "model_comparison.png", dpi=120)
    plt.close(fig)
    print(f"\n📊  Model comparison chart saved → {out_dir / 'model_comparison.png'}")


def compute_city_stats(df_full: pd.DataFrame, out_dir: Path):
    """Compute and save per-city median price/sqft for the app insights panel."""
    df_full = df_full.copy()
    df_full["price_per_sqft"] = df_full["price"] / df_full["area_sqft"]
    stats = (
        df_full.groupby("city")["price_per_sqft"]
        .agg(median_price_per_sqft="median", count="count")
        .round(0)
        .to_dict(orient="index")
    )
    overall_median = float(df_full["price_per_sqft"].median())
    stats["_overall_median"] = {"median_price_per_sqft": overall_median, "count": len(df_full)}
    out_path = out_dir / "city_stats.json"
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"      ✅  City stats saved → {out_path}")


# ── Main Training Routine ─────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  🏠  India House Price Prediction — Model Training")
    print("=" * 60)

    # 1. Load & clean
    print("\n[1/5] Loading and cleaning data …")
    df = load_data()
    df = clean_data(df)
    print(f"      Dataset: {df.shape[0]:,} rows × {df.shape[1]} cols")

    # 2. Feature engineering
    print("[2/5] Engineering features …")
    df_fe = engineer_features(df)

    # Update numeric feature list to include engineered features
    import src.preprocessing as prep_mod
    prep_mod.NUMERIC_FEATURES = ENGINEERED_NUMERIC_FEATURES

    # 3. Split
    print("[3/5] Splitting train / test (80/20) …")
    X_train, X_test, y_train, y_test = split_data(df_fe)
    print(f"      Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # 4. Define models
    preprocessor = build_preprocessor()

    models = {
        "Linear Regression (Ridge)": Pipeline([
            ("prep",  preprocessor),
            ("model", Ridge(alpha=10.0)),
        ]),
        "Random Forest": Pipeline([
            ("prep",  build_preprocessor()),
            ("model", RandomForestRegressor(
                n_estimators=300, max_depth=14,
                min_samples_leaf=3, n_jobs=-1, random_state=42,
            )),
        ]),
        "XGBoost": Pipeline([
            ("prep",  build_preprocessor()),
            ("model", XGBRegressor(
                n_estimators=500, learning_rate=0.05,
                max_depth=6, subsample=0.8,
                colsample_bytree=0.8, reg_alpha=0.1,
                reg_lambda=1.0, random_state=42,
                verbosity=0, n_jobs=-1,
            )),
        ]),
    }

    # 5. Train, evaluate, plot
    print("[4/5] Training models …\n")
    all_metrics: dict[str, dict] = {}
    best_r2       = -np.inf
    best_name     = ""
    best_pipeline = None

    for name, pipeline in models.items():
        t0 = time.time()
        print(f"  ▶  {name} …", end=" ", flush=True)
        pipeline.fit(X_train, y_train)
        elapsed = time.time() - t0
        print(f"done ({elapsed:.1f}s)")

        y_pred  = pipeline.predict(X_test)
        metrics = compute_metrics(y_test, y_pred)
        all_metrics[name] = metrics
        print_metrics(name, metrics)

        # Plots
        plot_actual_vs_predicted(y_test.values, y_pred, name, OUT_DIR)
        plot_residuals(y_test.values, y_pred, name, OUT_DIR)

        preprocessor_fit = pipeline.named_steps["prep"]
        from src.preprocessing import get_feature_names
        feat_names = get_feature_names(preprocessor_fit)
        plot_feature_importance(pipeline, feat_names, name, OUT_DIR)

        if metrics["R2"] > best_r2:
            best_r2       = metrics["R2"]
            best_name     = name
            best_pipeline = pipeline

    plot_model_comparison(all_metrics, OUT_DIR)

    # Save city stats
    print("\n[5/5] Saving best model & city statistics …")
    compute_city_stats(df, OUT_DIR)

    model_path = MODEL_DIR / "best_model.joblib"
    joblib.dump(best_pipeline, model_path)
    print(f"      ✅  Best model ({best_name}, R²={best_r2:.4f}) saved → {model_path}")

    metrics_path = OUT_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({"best_model": best_name, "metrics": all_metrics}, f, indent=2)
    print(f"      ✅  Metrics saved → {metrics_path}")

    print("\n" + "=" * 60)
    print("  🎉  Training complete!  Best model:", best_name)
    print("=" * 60)


if __name__ == "__main__":
    main()
