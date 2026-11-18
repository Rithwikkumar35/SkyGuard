"""
src/train.py
Model training, inference, metadata tracking, and artifact serialization for SkyGuard.
"""

import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from datetime import datetime
import json
from typing import Dict, Optional, Tuple, Any
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.data_loader import load_raw_dataset
from src.preprocessing import WeatherPreprocessor
from src.utils import get_models_dir, get_results_dir, setup_logger

logger = setup_logger("ModelTraining")

DEFAULT_N_ESTIMATORS = 150
DEFAULT_RANDOM_STATE = 42
DEFAULT_CONTAMINATION = 0.03


def train_isolation_forest(
    df: Optional[pd.DataFrame] = None,
    contamination: float = DEFAULT_CONTAMINATION,
    n_estimators: int = DEFAULT_N_ESTIMATORS,
    random_state: int = DEFAULT_RANDOM_STATE,
    feature_mode: str = "baseline",
    save_artifacts: bool = True,
) -> Tuple[IsolationForest, WeatherPreprocessor, pd.DataFrame, Dict[str, Any]]:
    """
    Train Isolation Forest model on weather observations, compute anomaly scores,
    and optionally serialize the fitted model, scaler, and metadata.

    Parameters:
    -----------
    df: Weather DataFrame. If None, loaded from raw dataset.
    contamination: Expected proportion of anomalies (default 0.03).
    n_estimators: Number of isolation trees (default 150).
    random_state: Seed for reproducibility (default 42).
    feature_mode: 'baseline' (5 features) or 'engineered' (9 features).
    save_artifacts: Whether to persist artifacts to models/ and results/.

    Returns:
    --------
    model: Fitted IsolationForest instance.
    preprocessor: Fitted WeatherPreprocessor instance.
    results_df: Aligned DataFrame with features, anomaly scores, and flags.
    metadata: Comprehensive metadata dictionary.
    """
    if df is None:
        df = load_raw_dataset(parse_measurements=True)

    logger.info(
        f"Training Isolation Forest [n_estimators={n_estimators}, "
        f"contamination={contamination:.2%}, random_state={random_state}, mode={feature_mode}]..."
    )

    # 1. Fit Preprocessor
    preprocessor = WeatherPreprocessor(
        feature_mode=feature_mode,
        scaler_type="standard",
        imputation_strategy="median",
    )
    X_scaled = preprocessor.fit_transform(df)

    # 2. Fit Isolation Forest
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    predictions = model.fit_predict(X_scaled)
    # decision_function: lower values = more abnormal
    scores = model.decision_function(X_scaled)
    anomaly_flags = predictions == -1

    # 3. Construct Aligned Results DataFrame
    metadata_cols = [c for c in ["City", "Country", "DATE", "DATE_PARSED", "latitude", "longitude"] if c in df.columns]
    results_df = df[metadata_cols].copy()

    # Add numeric features used
    feature_names = preprocessor.get_feature_names()
    for col in feature_names:
        if col in df.columns:
            results_df[col] = df[col]

    results_df["anomaly_score"] = scores
    results_df["anomaly"] = anomaly_flags
    results_df["anomaly_label"] = np.where(anomaly_flags, "Flagged", "Normal")

    # Severity scoring: rank anomalies into Moderate, High, Severe based on score percentiles
    flagged_mask = results_df["anomaly"]
    results_df["severity"] = "Normal"
    if flagged_mask.sum() > 0:
        flagged_scores = results_df.loc[flagged_mask, "anomaly_score"]
        p33 = flagged_scores.quantile(0.33)
        p66 = flagged_scores.quantile(0.66)

        results_df.loc[flagged_mask & (results_df["anomaly_score"] <= p33), "severity"] = "Severe"
        results_df.loc[flagged_mask & (results_df["anomaly_score"] > p33) & (results_df["anomaly_score"] <= p66), "severity"] = "High"
        results_df.loc[flagged_mask & (results_df["anomaly_score"] > p66), "severity"] = "Moderate"

    total_records = len(results_df)
    flagged_count = int(anomaly_flags.sum())
    flagged_pct = float(round((flagged_count / total_records) * 100, 2))

    # 4. Construct Metadata
    metadata = {
        "project": "SkyGuard Weather Anomaly Detection",
        "algorithm": "Isolation Forest",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_records_trained": total_records,
        "n_estimators": n_estimators,
        "random_state": random_state,
        "contamination_setting": contamination,
        "feature_mode": feature_mode,
        "features_used": feature_names,
        "flagged_observations": flagged_count,
        "flagged_percentage": flagged_pct,
        "score_statistics": {
            "mean": float(round(scores.mean(), 5)),
            "std": float(round(scores.std(), 5)),
            "min": float(round(scores.min(), 5)),
            "max": float(round(scores.max(), 5)),
            "threshold_offset": float(round(model.offset_, 5)),
            "quantiles": {
                "1%": float(round(np.percentile(scores, 1), 5)),
                "3%": float(round(np.percentile(scores, 3), 5)),
                "5%": float(round(np.percentile(scores, 5), 5)),
                "10%": float(round(np.percentile(scores, 10), 5)),
                "50%": float(round(np.percentile(scores, 50), 5)),
            }
        }
    }

    # 5. Persist Artifacts
    if save_artifacts:
        models_dir = get_models_dir()
        results_dir = get_results_dir()

        # Save model and preprocessor/scaler
        model_path = models_dir / "isolation_forest.joblib"
        scaler_path = models_dir / "scaler.joblib"
        metadata_path = models_dir / "model_metadata.json"

        joblib.dump(model, model_path)
        joblib.dump(preprocessor.scaler_, scaler_path)
        preprocessor.save(models_dir / "preprocessor.joblib")

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

        # Save CSV outputs
        results_df.sort_values("anomaly_score").to_csv(
            results_dir / "anomaly_results_with_locations.csv",
            index=False,
        )
        results_df[feature_names + ["anomaly_score", "anomaly"]].to_csv(
            results_dir / "anomaly_results.csv",
            index=False,
        )

        logger.info(f"Model and artifacts serialized to {models_dir} and {results_dir}.")

    logger.info(
        f"Training complete. Flagged {flagged_count:,} of {total_records:,} "
        f"observations ({flagged_pct:.2f}%)."
    )
    return model, preprocessor, results_df, metadata


def load_trained_model(
    model_path: Optional[Path] = None,
    metadata_path: Optional[Path] = None,
) -> Tuple[IsolationForest, Dict[str, Any]]:
    """
    Load the serialized Isolation Forest model and its accompanying metadata.
    """
    if model_path is None:
        model_path = get_models_dir() / "isolation_forest.joblib"
    if metadata_path is None:
        metadata_path = get_models_dir() / "model_metadata.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    model = joblib.load(model_path)
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

    logger.info(f"Loaded trained IsolationForest from {model_path}")
    return model, metadata


if __name__ == "__main__":
    train_isolation_forest()
