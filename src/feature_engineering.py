"""
src/feature_engineering.py
Domain-specific meteorological feature engineering for weather anomaly detection.
"""

import sys
from pathlib import Path

# Add project root to sys.path to support both direct and module execution
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from typing import List, Optional
import numpy as np
import pandas as pd

from src.utils import setup_logger

logger = setup_logger("FeatureEngineering")

BASELINE_FEATURES: List[str] = ["TMIN", "TAVG", "TMAX", "PRCP", "WSPD"]

ENGINEERED_FEATURES: List[str] = [
    "TMIN", "TAVG", "TMAX", "PRCP", "WSPD",
    "TEMP_RANGE", "TEMP_DEVIATION", "PRCP_LOG", "TEMP_DEV_CITY"
]


def compute_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute domain-justified meteorological features:
    1. TEMP_RANGE: Diurnal Temperature Range (TMAX - TMIN).
       Detects stagnant sensor values (near 0) or extreme radiative swings.
    2. TEMP_DEVIATION: Deviation of daily mean from the diurnal midpoint:
       TAVG - (TMAX + TMIN) / 2. Detects sudden frontal passages or sensor skew.
    3. PRCP_LOG: Natural log transform of precipitation log1p(PRCP).
       Reduces right-skewness and dampens the scale of torrential rain extremes.
    4. TEMP_DEV_CITY: Localized temperature anomaly relative to the station's
       period baseline (TAVG - City_Mean_TAVG). Distinguishes genuine local
       anomalies from natural climatic latitudinal gradients.
    """
    df = df.copy()

    # 1. Diurnal Temperature Range (DTR)
    if "TMAX" in df.columns and "TMIN" in df.columns:
        df["TEMP_RANGE"] = df["TMAX"] - df["TMIN"]
    else:
        df["TEMP_RANGE"] = np.nan

    # 2. Temperature Diurnal Midpoint Deviation
    if {"TAVG", "TMAX", "TMIN"}.issubset(df.columns):
        midpoint = (df["TMAX"] + df["TMIN"]) / 2.0
        df["TEMP_DEVIATION"] = df["TAVG"] - midpoint
    else:
        df["TEMP_DEVIATION"] = np.nan

    # 3. Log-transformed precipitation
    if "PRCP" in df.columns:
        # Non-negative clipping for safety
        prcp_clean = df["PRCP"].clip(lower=0)
        df["PRCP_LOG"] = np.log1p(prcp_clean)
    else:
        df["PRCP_LOG"] = np.nan

    # 4. City-relative temperature anomaly
    if "City" in df.columns and "TAVG" in df.columns:
        city_means = df.groupby("City")["TAVG"].transform("mean")
        df["TEMP_DEV_CITY"] = df["TAVG"] - city_means
    elif "TAVG" in df.columns:
        # Fallback if City is not present
        df["TEMP_DEV_CITY"] = df["TAVG"] - df["TAVG"].mean()
    else:
        df["TEMP_DEV_CITY"] = np.nan

    logger.info("Engineered features computed: TEMP_RANGE, TEMP_DEVIATION, PRCP_LOG, TEMP_DEV_CITY.")
    return df


def get_feature_list(mode: str = "baseline") -> List[str]:
    """
    Return the feature column list according to feature engineering mode:
    - 'baseline': ["TMIN", "TAVG", "TMAX", "PRCP", "WSPD"]
    - 'engineered': ["TMIN", "TAVG", "TMAX", "PRCP", "WSPD", "TEMP_RANGE", "TEMP_DEVIATION", "PRCP_LOG", "TEMP_DEV_CITY"]
    """
    if mode == "engineered":
        return list(ENGINEERED_FEATURES)
    return list(BASELINE_FEATURES)
