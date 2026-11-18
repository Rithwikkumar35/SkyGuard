"""
src/evaluate.py
Unsupervised evaluation, comparative benchmarking (IF vs LOF vs DBSCAN),
contamination sensitivity, and qualitative investigation.
"""

import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import time
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.neighbors import LocalOutlierFactor

from src.data_loader import load_raw_dataset, audit_physical_consistency
from src.preprocessing import WeatherPreprocessor
from src.train import load_trained_model
from src.utils import get_results_dir, set_academic_plot_style, setup_logger

logger = setup_logger("Evaluation")


def evaluate_contamination_sensitivity(
    df: pd.DataFrame,
    contamination_levels: List[float] = [0.01, 0.03, 0.05, 0.10],
    n_estimators: int = 150,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Perform sensitivity analysis across multiple contamination settings.
    Evaluates stability and score statistics without relying on ground-truth labels.
    """
    logger.info("Evaluating contamination sensitivity...")
    preprocessor = WeatherPreprocessor(feature_mode="baseline")
    X = preprocessor.fit_transform(df)

    records = []
    for cont in contamination_levels:
        t0 = time.perf_counter()
        model = IsolationForest(
            n_estimators=n_estimators,
            contamination=cont,
            random_state=random_state,
            n_jobs=-1,
        )
        preds = model.fit_predict(X)
        scores = model.decision_function(X)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        flags = preds == -1
        count = int(flags.sum())
        pct = round((count / len(df)) * 100, 2)

        records.append({
            "contamination_setting": cont,
            "records_analyzed": len(df),
            "flagged_records": count,
            "flagged_percentage": pct,
            "mean_anomaly_score": round(float(scores.mean()), 4),
            "std_anomaly_score": round(float(scores.std()), 4),
            "min_anomaly_score": round(float(scores.min()), 4),
            "runtime_ms": round(elapsed_ms, 2),
        })

    comp_df = pd.DataFrame(records)
    results_dir = get_results_dir()
    comp_df.to_csv(results_dir / "contamination_comparison.csv", index=False)

    # Plot contamination comparison
    set_academic_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    x_labels = [f"{c:.0%}" for c in comp_df["contamination_setting"]]
    bars = ax1.bar(x_labels, comp_df["flagged_records"], color="#2b5c8f", edgecolor="black", linewidth=0.7)
    ax1.set_title("Flagged Observations by Contamination Setting", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Contamination Parameter (Expected %)")
    ax1.set_ylabel("Number of Flagged Observations")
    for bar in bars:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2.0, h + 8, f"{h:,}", ha="center", va="bottom", fontsize=9.5)

    ax2.plot(x_labels, comp_df["mean_anomaly_score"], marker="o", color="#d95f02", linewidth=2, markersize=7)
    ax2.set_title("Mean Decision-Function Score vs Contamination", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Contamination Parameter")
    ax2.set_ylabel("Mean Decision Function Score")

    plt.suptitle("SkyGuard: Isolation Forest Contamination Sensitivity Analysis", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(results_dir / "contamination_comparison.png")
    plt.savefig(results_dir / "contamination_sensitivity.png")
    plt.close()

    logger.info("Contamination sensitivity analysis complete.")
    return comp_df


def compare_unsupervised_algorithms(
    df: pd.DataFrame,
    contamination: float = 0.03,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Fair comparative benchmarking of three unsupervised anomaly detection paradigms:
    1. Isolation Forest (Randomized Tree Partitioning)
    2. Local Outlier Factor (LOF - Density-based Local K-NN deviation)
    3. DBSCAN (Density-based Spatial Clustering with Noise identification)

    Evaluates:
    - Flagged anomaly count & percentage
    - Execution latency (runtime)
    - Jaccard similarity & anomaly overlap
    - Silhouette score between normal cluster mass and detected anomalies
    """
    logger.info("Executing comparative benchmarking: IF vs LOF vs DBSCAN...")
    preprocessor = WeatherPreprocessor(feature_mode="baseline")
    X = preprocessor.fit_transform(df)

    algorithms = {}

    # 1. Isolation Forest
    t0 = time.perf_counter()
    if_model = IsolationForest(
        n_estimators=150,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    if_preds = if_model.fit_predict(X)
    if_time = (time.perf_counter() - t0) * 1000
    if_anomalies = if_preds == -1
    algorithms["Isolation Forest"] = {
        "flags": if_anomalies,
        "runtime_ms": if_time,
        "paradigm": "Tree-based Partitioning",
    }

    # 2. Local Outlier Factor
    t0 = time.perf_counter()
    lof_model = LocalOutlierFactor(
        n_neighbors=20,
        contamination=contamination,
        n_jobs=-1,
    )
    lof_preds = lof_model.fit_predict(X)
    lof_time = (time.perf_counter() - t0) * 1000
    lof_anomalies = lof_preds == -1
    algorithms["Local Outlier Factor"] = {
        "flags": lof_anomalies,
        "runtime_ms": lof_time,
        "paradigm": "K-NN Density Deviation",
    }

    # 3. DBSCAN (eps=1.8, min_samples=10, tuned for 5-D standardized weather data)
    t0 = time.perf_counter()
    dbscan = DBSCAN(eps=1.8, min_samples=10, n_jobs=-1)
    db_labels = dbscan.fit_predict(X)
    db_time = (time.perf_counter() - t0) * 1000
    # DBSCAN label -1 indicates noise/outliers
    db_anomalies = db_labels == -1
    algorithms["DBSCAN"] = {
        "flags": db_anomalies,
        "runtime_ms": db_time,
        "paradigm": "Density Clustering (Noise)",
    }

    # Compute comparative metrics
    comparison_records = []
    for name, data in algorithms.items():
        flags = data["flags"]
        count = int(flags.sum())
        pct = round((count / len(df)) * 100, 2)

        # Silhouette score: measures separation between the normal partition and anomalous partition
        # Only valid if at least 2 distinct clusters exist with size > 1
        binary_labels = flags.astype(int)
        if len(np.unique(binary_labels)) > 1 and count > 1 and count < len(df) - 1:
            try:
                # Sample up to 3000 points if needed for speed
                sample_idx = np.random.RandomState(42).choice(len(X), size=min(3000, len(X)), replace=False)
                sil_score = round(float(silhouette_score(X[sample_idx], binary_labels[sample_idx])), 4)
            except Exception:
                sil_score = np.nan
        else:
            sil_score = np.nan

        comparison_records.append({
            "algorithm": name,
            "paradigm": data["paradigm"],
            "flagged_observations": count,
            "flagged_percentage": pct,
            "silhouette_score": sil_score,
            "runtime_ms": round(data["runtime_ms"], 2),
        })

    comp_df = pd.DataFrame(comparison_records)
    results_dir = get_results_dir()
    comp_df.to_csv(results_dir / "model_comparison.csv", index=False)

    # Calculate Jaccard similarity between detected anomaly sets
    if_set = set(np.where(if_anomalies)[0])
    lof_set = set(np.where(lof_anomalies)[0])
    db_set = set(np.where(db_anomalies)[0])

    def jaccard(s1, s2):
        union = len(s1.union(s2))
        return round(len(s1.intersection(s2)) / union, 4) if union > 0 else 0.0

    overlap_info = {
        "if_count": len(if_set),
        "lof_count": len(lof_set),
        "dbscan_count": len(db_set),
        "if_and_lof_overlap": len(if_set.intersection(lof_set)),
        "if_and_dbscan_overlap": len(if_set.intersection(db_set)),
        "lof_and_dbscan_overlap": len(lof_set.intersection(db_set)),
        "three_way_agreement": len(if_set.intersection(lof_set).intersection(db_set)),
        "jaccard_if_lof": jaccard(if_set, lof_set),
        "jaccard_if_dbscan": jaccard(if_set, db_set),
        "jaccard_lof_dbscan": jaccard(lof_set, db_set),
    }

    # Plot Model Comparison & Overlap
    set_academic_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart of flagged observations and runtimes
    alg_names = comp_df["algorithm"]
    colors = ["#2b5c8f", "#7570b3", "#1b9e77"]
    bars = ax1.bar(alg_names, comp_df["flagged_observations"], color=colors, edgecolor="black", linewidth=0.7)
    ax1.set_title("Flagged Observations by Algorithm", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Number of Flagged Observations")
    for bar in bars:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2.0, h + 5, f"{h}", ha="center", va="bottom", fontsize=10)

    # Runtime comparison
    bars2 = ax2.bar(alg_names, comp_df["runtime_ms"], color=colors, edgecolor="black", linewidth=0.7)
    ax2.set_title("Computational Latency / Runtime (ms)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Execution Time (milliseconds)")
    for bar in bars2:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, h + 2, f"{h:.1f} ms", ha="center", va="bottom", fontsize=10)

    plt.suptitle("SkyGuard: Unsupervised Anomaly Detection Benchmark (IF vs LOF vs DBSCAN)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(results_dir / "model_comparison.png")
    plt.close()

    logger.info("Comparative benchmark complete.")
    return comp_df, overlap_info


def plot_anomaly_score_distribution(
    scores: np.ndarray,
    flags: np.ndarray,
    threshold: float = 0.0,
) -> Path:
    """
    Plot the empirical anomaly score distribution for normal vs flagged observations.
    """
    set_academic_plot_style()
    results_dir = get_results_dir()

    plt.figure(figsize=(10, 5))
    sns.histplot(scores[~flags], color="#2b5c8f", kde=True, label="Normal Observations", bins=45, stat="density", alpha=0.6)
    sns.histplot(scores[flags], color="#d9534f", kde=True, label="Flagged Observations (Anomalies)", bins=25, stat="density", alpha=0.7)
    plt.axvline(threshold, color="black", linestyle="--", linewidth=1.5, label=f"Decision Boundary ({threshold:.3f})")

    plt.title("Isolation Forest: Decision-Function Anomaly Score Distribution", fontsize=12, fontweight="bold")
    plt.xlabel("Anomaly Score (Lower values denote greater anomaly severity)")
    plt.ylabel("Probability Density")
    plt.legend(loc="upper left")
    plt.tight_layout()

    out_path = results_dir / "anomaly_score_distribution.png"
    plt.savefig(out_path)
    plt.close()
    return out_path


def investigate_qualitative_anomalies(
    df: pd.DataFrame,
    results_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Conduct an in-depth qualitative investigation into the most anomalous observations:
    - Identifies dominant driving factors (polar vortex, severe storms, physical inconsistencies)
    - Cross-references with the 7 physically inconsistent records
    """
    logger.info("Investigating qualitative anomaly characteristics...")
    results_dir = get_results_dir()

    # Sort by anomaly score ascending (most anomalous first)
    top_flagged = results_df[results_df["anomaly"]].sort_values("anomaly_score").head(25).copy()

    # Cross-reference with physical inconsistencies
    audited, _ = audit_physical_consistency(df)
    inconsistent_indices = set(audited[audited["is_physically_inconsistent"]].index)

    reasons = []
    for idx, row in top_flagged.iterrows():
        driver = []
        if idx in inconsistent_indices:
            driver.append(f"PHYSICAL INCONSISTENCY: {audited.loc[idx, 'inconsistency_reason']}")
        if row["TMIN"] <= -20.0:
            driver.append(f"Extreme sub-zero cold ({row['TMIN']}°C)")
        if row["PRCP"] >= 20.0:
            driver.append(f"Heavy precipitation ({row['PRCP']} mm)")
        if row["WSPD"] >= 30.0:
            driver.append(f"High gale/storm wind ({row['WSPD']} km/h)")
        if not driver:
            driver.append("Multivariate outlier combination")
        reasons.append(" | ".join(driver))

    top_flagged["anomaly_driver"] = reasons
    cols_to_save = [c for c in ["City", "Country", "DATE", "TMIN", "TAVG", "TMAX", "PRCP", "WSPD", "anomaly_score", "severity", "anomaly_driver"] if c in top_flagged.columns]
    qual_df = top_flagged[cols_to_save]
    qual_df.to_csv(results_dir / "qualitative_anomaly_analysis.csv", index=False)

    logger.info(f"Top 25 anomalies analyzed and saved to qualitative_anomaly_analysis.csv.")
    return qual_df


def run_full_evaluation() -> Dict[str, Any]:
    """
    Run complete unsupervised model evaluation and generate all reports.
    """
    df = load_raw_dataset(parse_measurements=True)
    results_path = get_results_dir() / "anomaly_results_with_locations.csv"
    if results_path.exists():
        results_df = pd.read_csv(results_path)
    else:
        from src.train import train_isolation_forest
        _, _, results_df, _ = train_isolation_forest(df)

    # 1. Contamination Sensitivity
    sensitivity_df = evaluate_contamination_sensitivity(df)

    # 2. Comparative Benchmark (IF vs LOF vs DBSCAN)
    comp_df, overlap_info = compare_unsupervised_algorithms(df)

    # 3. Anomaly Score Distribution Plot
    plot_anomaly_score_distribution(
        results_df["anomaly_score"].to_numpy(),
        results_df["anomaly"].to_numpy(),
        threshold=0.0,
    )

    # 4. Qualitative Investigation
    qual_df = investigate_qualitative_anomalies(df, results_df)

    return {
        "status": "success",
        "sensitivity": sensitivity_df.to_dict(orient="records"),
        "comparison": comp_df.to_dict(orient="records"),
        "overlap": overlap_info,
        "top_anomalies_count": len(qual_df),
    }


if __name__ == "__main__":
    run_full_evaluation()
