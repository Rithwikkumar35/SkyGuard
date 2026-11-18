"""
src/eda.py
Comprehensive Exploratory Data Analysis (EDA) module for SkyGuard.
Generates statistical summaries, multi-panel visualizations, and data-quality artifacts.
"""

import sys
from pathlib import Path

# Add project root to sys.path to support both direct and module execution
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import json
from typing import Dict, Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data_loader import (
    load_raw_dataset,
    audit_physical_consistency,
    generate_data_quality_report,
)
from src.feature_engineering import compute_engineered_features
from src.utils import get_results_dir, set_academic_plot_style, setup_logger

logger = setup_logger("EDA")


def run_full_eda() -> Dict[str, Any]:
    """
    Execute full exploratory data analysis pipeline and export publication-ready charts.
    """
    set_academic_plot_style()
    results_dir = get_results_dir()

    logger.info("Starting comprehensive EDA...")
    raw_df = load_raw_dataset(parse_measurements=True)
    df_engineered = compute_engineered_features(raw_df)

    # 1. Data Quality & Missing Value Audit
    quality_report = generate_data_quality_report(raw_df)
    quality_json_path = results_dir / "data_quality_report.json"
    with open(quality_json_path, "w") as f:
        json.dump(quality_report, f, indent=4)
    logger.info(f"Saved data quality report to {quality_json_path}")

    # Plot missing value analysis across all 15 raw attributes
    plt.figure(figsize=(10, 5))
    missing_series = pd.Series(quality_report["missing_percentages"]).sort_values(ascending=False)
    colors = ["#D9534F" if pct > 50 else "#F0AD4E" if pct > 5 else "#5CB85C" for pct in missing_series.values]
    bars = plt.bar(missing_series.index, missing_series.values, color=colors, edgecolor="#333333", linewidth=0.7)
    plt.axhline(50, color="red", linestyle="--", alpha=0.6, label="Severe Missing Threshold (50%)")
    plt.title("Missing Data Percentage by Column (Raw Dataset)", fontsize=13, fontweight="bold")
    plt.ylabel("Missing Observations (%)")
    plt.xlabel("Dataset Columns")
    plt.xticks(rotation=45, ha="right")
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            plt.text(bar.get_x() + bar.get_width() / 2.0, height + 1.2, f"{height:.1f}%",
                     ha="center", va="bottom", fontsize=8.5)
    plt.ylim(0, 110)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(results_dir / "missing_values_analysis.png")
    plt.close()

    # 2. Descriptive Statistics Export
    features = ["TMIN", "TAVG", "TMAX", "PRCP", "WSPD", "TEMP_RANGE", "TEMP_DEVIATION", "PRCP_LOG"]
    desc_df = df_engineered[features].describe().T
    desc_df["iqr"] = desc_df["75%"] - desc_df["25%"]
    desc_df["skewness"] = df_engineered[features].skew()
    desc_df.to_csv(results_dir / "descriptive_statistics.csv")
    logger.info("Saved descriptive statistics to descriptive_statistics.csv")

    # 3. Feature Distributions (Multi-panel)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    dist_features = [
        ("TAVG", "Average Daily Temperature (°C)", "#2b5c8f"),
        ("TMAX", "Maximum Daily Temperature (°C)", "#d95f02"),
        ("TMIN", "Minimum Daily Temperature (°C)", "#7570b3"),
        ("PRCP", "Daily Precipitation (mm)", "#1b9e77"),
        ("WSPD", "Sustained Wind Speed (km/h)", "#e7298a"),
        ("TEMP_RANGE", "Diurnal Temp Range (°C)", "#66a61e"),
    ]

    for idx, (col, label, col_color) in enumerate(dist_features):
        ax = axes[idx]
        sns.histplot(df_engineered[col].dropna(), kde=True, ax=ax, color=col_color, bins=35, edgecolor="black", linewidth=0.5)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")

    plt.suptitle("SkyGuard: Feature Distributions with Kernel Density Estimates", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(results_dir / "feature_distributions.png")
    plt.close()

    # 4. Correlation Heatmap (Including Engineered Features)
    corr_features = ["TMIN", "TAVG", "TMAX", "PRCP", "WSPD", "TEMP_RANGE", "TEMP_DEVIATION", "PRCP_LOG"]
    corr_matrix = df_engineered[corr_features].corr()

    plt.figure(figsize=(9, 7))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Pearson Correlation Coefficient"},
    )
    plt.title("Weather Feature Correlation Matrix (Triangular View)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(results_dir / "correlation_matrix.png")
    plt.close()

    # 5. Temperature Analysis & Thermodynamic Consistency
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Scatter: TMIN vs TMAX with identity line
    ax1.scatter(df_engineered["TMIN"], df_engineered["TMAX"], alpha=0.35, s=15, color="#1f77b4", label="Valid Observations")
    min_temp = min(df_engineered["TMIN"].min(), df_engineered["TMAX"].min())
    max_temp = max(df_engineered["TMIN"].max(), df_engineered["TMAX"].max())
    ax1.plot([min_temp, max_temp], [min_temp, max_temp], color="red", linestyle="--", label="TMIN = TMAX Boundary")
    ax1.set_title("TMIN vs TMAX Thermodynamic Check", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Minimum Temperature TMIN (°C)")
    ax1.set_ylabel("Maximum Temperature TMAX (°C)")
    ax1.legend()

    # Highlight physical anomaly cases: TAVG outside [TMIN, TMAX]
    df_audited, audit_summary = audit_physical_consistency(df_engineered)
    inconsistent_mask = df_audited["is_physically_inconsistent"]
    
    ax2.hist(df_engineered["TEMP_RANGE"].dropna(), bins=35, color="#e6ab02", edgecolor="black", linewidth=0.5)
    ax2.axvline(0, color="red", linestyle="--", label="Zero Range (Stuck Sensor)")
    ax2.set_title(f"Diurnal Temperature Range (DTR) | {audit_summary['total_inconsistent_records']} Physical Inconsistencies", fontsize=11, fontweight="bold")
    ax2.set_xlabel("DTR = TMAX - TMIN (°C)")
    ax2.set_ylabel("Observation Count")
    ax2.legend()

    plt.suptitle("SkyGuard: Temperature Analysis & Consistency Diagnostics", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(results_dir / "temperature_analysis.png")
    plt.close()

    # 6. Rainfall Distribution (Raw vs Log-Transformed)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    
    prcp_clean = df_engineered["PRCP"].dropna()
    dry_days_pct = (prcp_clean == 0).mean() * 100
    
    ax1.hist(prcp_clean[prcp_clean > 0], bins=40, color="#2ca02c", edgecolor="black", linewidth=0.5)
    ax1.set_title(f"Wet Days Precipitation Distribution (Dry Days: {dry_days_pct:.1f}%)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Precipitation (mm)")
    ax1.set_ylabel("Frequency")
    
    ax2.hist(df_engineered["PRCP_LOG"].dropna(), bins=40, color="#17becf", edgecolor="black", linewidth=0.5)
    ax2.set_title("Log-Transformed Precipitation log1p(PRCP)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("log(1 + PRCP)")
    ax2.set_ylabel("Frequency")

    plt.suptitle("SkyGuard: Rainfall Distribution & Zero-Inflation Handling", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(results_dir / "rainfall_distribution.png")
    plt.close()

    # 7. Wind-speed Analysis
    plt.figure(figsize=(10, 4.5))
    sns.histplot(df_engineered["WSPD"].dropna(), kde=True, color="#9467bd", bins=35, edgecolor="black", linewidth=0.5)
    p99 = df_engineered["WSPD"].quantile(0.99)
    plt.axvline(p99, color="crimson", linestyle="--", label=f"99th Percentile ({p99:.1f} km/h)")
    plt.title("Sustained Wind Speed (WSPD) Distribution & Extreme Cutoff", fontsize=12, fontweight="bold")
    plt.xlabel("Sustained Wind Speed (km/h)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "wind_speed_analysis.png")
    plt.close()

    # 8. Outliers Visualization (Boxplots)
    plt.figure(figsize=(12, 6))
    features_to_plot = ["TMIN", "TAVG", "TMAX", "PRCP", "WSPD", "TEMP_RANGE"]
    sns.boxplot(data=df_engineered[features_to_plot], orient="h", palette="Set2")
    plt.title("Weather Feature Boxplots: Outliers & Spread (Original Units)", fontsize=13, fontweight="bold")
    plt.xlabel("Measurement Value")
    plt.tight_layout()
    plt.savefig(results_dir / "outliers_boxplots.png")
    plt.close()

    # 9. City-Wise Weather Comparisons
    city_group = df_engineered.groupby("City").agg({
        "TAVG": "mean",
        "PRCP": "sum",
        "WSPD": "mean",
        "TEMP_RANGE": "mean"
    }).reset_index()

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 5 Coldest and 5 Warmest
    top_cold = city_group.nsmallest(7, "TAVG")
    top_warm = city_group.nlargest(7, "TAVG")
    temp_extremes = pd.concat([top_cold, top_warm]).sort_values("TAVG")
    axes[0, 0].barh(temp_extremes["City"], temp_extremes["TAVG"], color=["#3182bd" if t < 10 else "#de2d26" for t in temp_extremes["TAVG"]])
    axes[0, 0].set_title("Extreme Mean Temperature Cities (°C)", fontsize=11, fontweight="bold")
    axes[0, 0].set_xlabel("Mean TAVG (°C)")

    # 10 Wettest Cities (Cumulative Precipitation)
    top_wet = city_group.nlargest(10, "PRCP").sort_values("PRCP")
    axes[0, 1].barh(top_wet["City"], top_wet["PRCP"], color="#2ca02c")
    axes[0, 1].set_title("Top 10 Wettest Cities (Cumulative Jan-Feb Rain mm)", fontsize=11, fontweight="bold")
    axes[0, 1].set_xlabel("Total PRCP (mm)")

    # 10 Windiest Cities
    top_wind = city_group.nlargest(10, "WSPD").sort_values("WSPD")
    axes[1, 0].barh(top_wind["City"], top_wind["WSPD"], color="#9467bd")
    axes[1, 0].set_title("Top 10 Windiest Cities (Mean Wind km/h)", fontsize=11, fontweight="bold")
    axes[1, 0].set_xlabel("Mean WSPD (km/h)")

    # 10 Highest Diurnal Temperature Range
    top_range = city_group.nlargest(10, "TEMP_RANGE").sort_values("TEMP_RANGE")
    axes[1, 1].barh(top_range["City"], top_range["TEMP_RANGE"], color="#e6ab02")
    axes[1, 1].set_title("Top 10 Highest Diurnal Temp Range (TMAX - TMIN)", fontsize=11, fontweight="bold")
    axes[1, 1].set_xlabel("Mean Diurnal Range (°C)")

    plt.suptitle("SkyGuard: Worldwide City-Wise Meteorological Comparison (Jan-Feb 2025)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(results_dir / "city_weather_comparisons.png")
    plt.close()

    logger.info("Comprehensive EDA complete. All figures saved to results/ directory.")
    return {
        "status": "success",
        "records_analyzed": len(df_engineered),
        "figures_generated": [
            "missing_values_analysis.png",
            "feature_distributions.png",
            "correlation_matrix.png",
            "temperature_analysis.png",
            "rainfall_distribution.png",
            "wind_speed_analysis.png",
            "outliers_boxplots.png",
            "city_weather_comparisons.png",
        ]
    }


if __name__ == "__main__":
    run_full_eda()
