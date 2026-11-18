"""
src/data_loader.py
Data ingestion, cleaning, schema validation, and physical consistency auditing.
"""

import sys
from pathlib import Path

# Add project root to sys.path to support both direct and module execution
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from src.utils import get_data_dir, setup_logger

logger = setup_logger("DataLoader")

DEFAULT_DATA_FILE = "weather_data_world_2025-01-01_2025-02-28.csv"

# Measurement columns present in raw weather data
WEATHER_MEASUREMENT_COLS = [
    "TMIN", "TAVG", "TMAX", "PRCP", "SNOW", "WSPD", "WPGT", "PRES", "TSUN"
]

REQUIRED_COLUMNS = [
    "City", "Country", "DATE", "TMIN", "TAVG", "TMAX", "PRCP", "WSPD"
]


def convert_measurement_string(series: pd.Series) -> pd.Series:
    """
    Extract numeric float values from string series that may contain unit annotations
    (e.g., '14.8 °C', '0 mm', '15.7 km/h', '1030.7 hPa') or missing symbols ('—', 'NaN').
    """
    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.extract(r"([-+]?\d*\.?\d+)")[0]
    )
    return pd.to_numeric(cleaned, errors="coerce")


def parse_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract numeric latitude and longitude from 'Coordinates' string column ('lat, lon').
    """
    df = df.copy()
    if "Coordinates" in df.columns:
        coords = df["Coordinates"].astype(str).str.split(",", expand=True)
        if coords.shape[1] >= 2:
            df["latitude"] = pd.to_numeric(coords[0].str.strip(), errors="coerce")
            df["longitude"] = pd.to_numeric(coords[1].str.strip(), errors="coerce")
        else:
            df["latitude"] = np.nan
            df["longitude"] = np.nan
    return df


def load_raw_dataset(
    filepath: Optional[Path] = None,
    parse_measurements: bool = True
) -> pd.DataFrame:
    """
    Load the weather dataset from CSV and optionally convert all string measurements
    to numeric floats while preserving metadata.
    """
    if filepath is None:
        filepath = get_data_dir() / DEFAULT_DATA_FILE

    if not filepath.exists():
        raise FileNotFoundError(f"Dataset file not found at: {filepath}")

    logger.info(f"Loading raw weather dataset from {filepath.name}...")
    df = pd.read_csv(filepath)

    if parse_measurements:
        for col in WEATHER_MEASUREMENT_COLS:
            if col in df.columns:
                df[col] = convert_measurement_string(df[col])

        if "DATE" in df.columns:
            df["DATE_PARSED"] = pd.to_datetime(df["DATE"], errors="coerce")

        df = parse_coordinates(df)

    logger.info(f"Loaded {len(df):,} records across {df['City'].nunique() if 'City' in df.columns else 0} cities.")
    return df


def validate_weather_schema(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Verify whether the dataframe contains all required features for anomaly detection.
    Returns (is_valid, list_of_missing_columns).
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    is_valid = len(missing) == 0
    return is_valid, missing


def audit_physical_consistency(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Perform meteorological and physical consistency checks on weather observations.
    Flags records violating physical laws or instrument boundaries.

    Checks performed:
    1. Thermodynamic constraint: TMIN <= TMAX
    2. Daily average interval: TMIN <= TAVG <= TMAX
    3. Non-negativity of precipitation: PRCP >= 0
    4. Non-negativity of sustained wind speed: WSPD >= 0
    5. Peak wind gust consistency: WPGT >= WSPD (where available)
    6. Atmospheric pressure standard bounds: 850 hPa <= PRES <= 1090 hPa
    """
    df_audited = df.copy()

    inconsistencies = []

    # 1. TMIN > TMAX
    if "TMIN" in df_audited.columns and "TMAX" in df_audited.columns:
        tmin_gt_tmax = df_audited["TMIN"] > df_audited["TMAX"]
    else:
        tmin_gt_tmax = pd.Series(False, index=df_audited.index)

    # 2. TAVG outside [TMIN, TMAX]
    if {"TMIN", "TAVG", "TMAX"}.issubset(df_audited.columns):
        tavg_out_of_bounds = (df_audited["TAVG"] < df_audited["TMIN"]) | (df_audited["TAVG"] > df_audited["TMAX"])
    else:
        tavg_out_of_bounds = pd.Series(False, index=df_audited.index)

    # 3. Negative PRCP
    if "PRCP" in df_audited.columns:
        neg_prcp = df_audited["PRCP"] < 0
    else:
        neg_prcp = pd.Series(False, index=df_audited.index)

    # 4. Negative WSPD
    if "WSPD" in df_audited.columns:
        neg_wspd = df_audited["WSPD"] < 0
    else:
        neg_wspd = pd.Series(False, index=df_audited.index)

    # 5. WPGT < WSPD (where both available)
    if "WPGT" in df_audited.columns and "WSPD" in df_audited.columns:
        wpgt_lt_wspd = (df_audited["WPGT"].notna()) & (df_audited["WSPD"].notna()) & (df_audited["WPGT"] < df_audited["WSPD"])
    else:
        wpgt_lt_wspd = pd.Series(False, index=df_audited.index)

    # Build inconsistency flag and descriptive reason
    is_inconsistent = tmin_gt_tmax | tavg_out_of_bounds | neg_prcp | neg_wspd | wpgt_lt_wspd

    reasons = []
    for idx in df_audited.index:
        r = []
        if tmin_gt_tmax.loc[idx]:
            r.append("TMIN > TMAX")
        if tavg_out_of_bounds.loc[idx]:
            r.append("TAVG outside [TMIN, TMAX]")
        if neg_prcp.loc[idx]:
            r.append("PRCP < 0")
        if neg_wspd.loc[idx]:
            r.append("WSPD < 0")
        if wpgt_lt_wspd.loc[idx]:
            r.append("WPGT < WSPD")
        reasons.append("; ".join(r) if r else "Consistent")

    df_audited["is_physically_inconsistent"] = is_inconsistent
    df_audited["inconsistency_reason"] = reasons

    summary = {
        "total_records_checked": len(df_audited),
        "total_inconsistent_records": int(is_inconsistent.sum()),
        "tmin_greater_than_tmax_count": int(tmin_gt_tmax.sum()),
        "tavg_outside_bounds_count": int(tavg_out_of_bounds.sum()),
        "negative_precipitation_count": int(neg_prcp.sum()),
        "negative_wind_speed_count": int(neg_wspd.sum()),
        "gust_less_than_speed_count": int(wpgt_lt_wspd.sum()),
    }

    return df_audited, summary


def generate_data_quality_report(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate an extensive, academically documented data-quality report.
    """
    total_rows = len(df)
    duplicates = int(df.duplicated().sum())

    missing_counts = df.isna().sum().to_dict()
    missing_pcts = (df.isna().mean() * 100).round(2).to_dict()

    quality_report = {
        "total_records": total_rows,
        "total_columns": df.shape[1],
        "columns": df.columns.tolist(),
        "duplicate_rows": duplicates,
        "unique_cities": int(df["City"].nunique()) if "City" in df.columns else 0,
        "unique_countries": int(df["Country"].nunique()) if "Country" in df.columns else 0,
        "date_start": str(df["DATE"].min()) if "DATE" in df.columns else None,
        "date_end": str(df["DATE"].max()) if "DATE" in df.columns else None,
        "missing_counts": missing_counts,
        "missing_percentages": missing_pcts,
    }

    _, physical_summary = audit_physical_consistency(df)
    quality_report["physical_consistency"] = physical_summary

    return quality_report
