"""
tests/test_data.py
Unit tests for data ingestion, string parsing, schema validation, and physical consistency.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data_loader import (
    load_raw_dataset,
    convert_measurement_string,
    parse_coordinates,
    validate_weather_schema,
    audit_physical_consistency,
    generate_data_quality_report,
)


def test_measurement_string_conversion():
    """Verify regex extraction from measurement strings with units and missing symbols."""
    series = pd.Series(["-15.4 °C", "0 mm", "35.2 km/h", "1024.5 hPa", "—", "NaN", None])
    converted = convert_measurement_string(series)

    assert converted.iloc[0] == -15.4
    assert converted.iloc[1] == 0.0
    assert converted.iloc[2] == 35.2
    assert converted.iloc[3] == 1024.5
    assert pd.isna(converted.iloc[4])
    assert pd.isna(converted.iloc[5])
    assert pd.isna(converted.iloc[6])


def test_parse_coordinates():
    """Verify latitude and longitude extraction from Coordinates column."""
    df = pd.DataFrame({"Coordinates": ["41.3888, 2.159", "-33.8688, 151.2093", "invalid"]})
    parsed = parse_coordinates(df)

    assert "latitude" in parsed.columns
    assert "longitude" in parsed.columns
    assert parsed["latitude"].iloc[0] == pytest.approx(41.3888)
    assert parsed["longitude"].iloc[0] == pytest.approx(2.159)
    assert parsed["latitude"].iloc[1] == pytest.approx(-33.8688)
    assert parsed["longitude"].iloc[1] == pytest.approx(151.2093)
    assert pd.isna(parsed["latitude"].iloc[2])


def test_raw_dataset_loading():
    """Verify loading and dimensions of the original dataset."""
    df = load_raw_dataset()

    assert len(df) == 6077
    assert df["City"].nunique() == 103
    assert df["Country"].nunique() == 35
    assert "DATE_PARSED" in df.columns
    assert "latitude" in df.columns
    assert "longitude" in df.columns


def test_schema_validation():
    """Verify valid schema acceptance and invalid schema rejection."""
    valid_df = pd.DataFrame({
        "City": ["Test"], "Country": ["XX"], "DATE": ["2025-01-01"],
        "TMIN": [5.0], "TAVG": [10.0], "TMAX": [15.0], "PRCP": [0.0], "WSPD": [12.0]
    })
    is_valid, missing = validate_weather_schema(valid_df)
    assert is_valid
    assert len(missing) == 0

    invalid_df = pd.DataFrame({"City": ["Test"], "TMIN": [5.0]})
    is_valid_inv, missing_inv = validate_weather_schema(invalid_df)
    assert not is_valid_inv
    assert "TMAX" in missing_inv
    assert "PRCP" in missing_inv


def test_physical_consistency_audit():
    """Verify meteorological boundary checks and detection of known anomalies."""
    df = load_raw_dataset()
    audited, summary = audit_physical_consistency(df)

    assert "is_physically_inconsistent" in audited.columns
    assert summary["total_records_checked"] == 6077
    assert summary["tmin_greater_than_tmax_count"] == 0
    # Exactly 7 records in dataset have TAVG outside [TMIN, TMAX]
    assert summary["tavg_outside_bounds_count"] == 7
    assert summary["total_inconsistent_records"] == 7
    assert summary["negative_precipitation_count"] == 0
    assert summary["negative_wind_speed_count"] == 0


def test_data_quality_report():
    """Verify comprehensive quality report generation."""
    df = load_raw_dataset()
    report = generate_data_quality_report(df)

    assert report["total_records"] == 6077
    assert report["duplicate_rows"] == 0
    assert report["unique_cities"] == 103
    assert "missing_counts" in report
    assert "physical_consistency" in report
    assert report["missing_counts"]["TMIN"] == 0
    assert report["missing_counts"]["PRCP"] == 472
