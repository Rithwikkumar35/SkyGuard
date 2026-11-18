"""
tests/test_integration.py
End-to-end integration tests, pipeline round-trip, Streamlit contracts, and edge-case simulation.
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
    validate_weather_schema,
    audit_physical_consistency,
    convert_measurement_string,
)
from src.preprocessing import WeatherPreprocessor
from src.train import load_trained_model, train_isolation_forest
from src.evaluate import compare_unsupervised_algorithms
from src.utils import get_models_dir, get_results_dir


def test_full_pipeline_roundtrip(tmp_path):
    """
    Verify complete pipeline execution:
    Raw data -> Cleaning -> Preprocessing -> Model training -> Anomaly scoring.
    """
    raw_df = load_raw_dataset()
    assert len(raw_df) == 6077

    # Fit preprocessor
    preprocessor = WeatherPreprocessor(feature_mode="baseline", scaler_type="standard")
    X = preprocessor.fit_transform(raw_df)
    assert X.shape == (6077, 5)

    # Train model
    model, _, results_df, metadata = train_isolation_forest(
        df=raw_df,
        contamination=0.03,
        n_estimators=100,
        random_state=42,
        save_artifacts=False,
    )

    assert len(results_df) == 6077
    assert results_df["anomaly"].sum() == 183
    assert "severity" in results_df.columns
    assert set(results_df["severity"].unique()).issubset({"Normal", "Severe", "High", "Moderate"})


def test_streamlit_backend_contracts():
    """
    Verify that the Streamlit application data and model loading contracts
    load successfully and conform to expected schemas.
    """
    from app.app import load_application_data, load_models

    audited_df, results_df, physical_audit = load_application_data()
    assert len(audited_df) == 6077
    assert len(results_df) == 6077
    assert physical_audit["total_inconsistent_records"] == 7

    model, preprocessor, metadata = load_models()
    assert model is not None
    assert preprocessor is not None
    assert metadata["project"] == "SkyGuard Weather Anomaly Detection"
    assert metadata["total_records_trained"] == 6077
    assert metadata["flagged_observations"] == 183


def test_live_upload_inference_simulation():
    """
    Simulate user uploading a CSV in Streamlit:
    Verify schema validation, measurement cleaning, transformation, and live prediction
    WITHOUT retraining the model.
    """
    # 1. Simulate uploaded data with unit strings, normal days, and extreme days
    user_upload_raw = pd.DataFrame({
        "City": ["TestCity", "TestCity", "TestCity", "TestCity"],
        "Country": ["TC", "TC", "TC", "TC"],
        "Elevation": [50, 50, 50, 50],
        "Timezone": ["UTC", "UTC", "UTC", "UTC"],
        "Coordinates": ["10.0, 20.0", "10.0, 20.0", "10.0, 20.0", "10.0, 20.0"],
        "DATE": ["2025-03-01", "2025-03-02", "2025-03-03", "2025-03-04"],
        "TMIN": ["12.0 °C", "-35.0 °C", "15.0 °C", "10.0 °C"],
        "TAVG": ["16.5 °C", "-30.0 °C", "20.0 °C", "14.0 °C"],
        "TMAX": ["22.0 °C", "-25.0 °C", "25.0 °C", "19.0 °C"],
        "PRCP": ["0.0 mm", "0.0 mm", "125.0 mm", "—"],
        "WSPD": ["10.0 km/h", "15.0 km/h", "60.0 km/h", "8.0 km/h"],
    })

    # 2. Validate schema
    is_valid, missing_cols = validate_weather_schema(user_upload_raw)
    assert is_valid
    assert len(missing_cols) == 0

    # 3. Clean measurements
    clean_upload = user_upload_raw.copy()
    for col in ["TMIN", "TAVG", "TMAX", "PRCP", "WSPD"]:
        clean_upload[col] = convert_measurement_string(clean_upload[col])

    assert clean_upload["TMIN"].iloc[1] == -35.0
    assert clean_upload["PRCP"].iloc[2] == 125.0
    assert pd.isna(clean_upload["PRCP"].iloc[3])  # '—' converted to NaN

    # 4. Load pre-trained preprocessor and model
    preprocessor = WeatherPreprocessor.load()
    model, _ = load_trained_model()

    # 5. Transform and predict
    X_user = preprocessor.transform(clean_upload)
    assert X_user.shape == (4, 5)
    assert not np.isnan(X_user).any()  # Missing value in PRCP was cleanly imputed

    preds = model.predict(X_user)
    scores = model.decision_function(X_user)

    assert len(preds) == 4
    assert len(scores) == 4
    # Observations 1 (-35°C polar vortex) and 2 (125mm rain + 60km/h wind) should be anomalies
    assert preds[1] == -1
    assert preds[2] == -1
    # Observation 0 (normal weather) should be classified normal
    assert preds[0] == 1


def test_edge_case_single_observation():
    """Verify inference pipeline resilience on a single observation."""
    preprocessor = WeatherPreprocessor.load()
    model, _ = load_trained_model()

    single_row = pd.DataFrame({
        "TMIN": [15.0], "TAVG": [18.0], "TMAX": [22.0], "PRCP": [0.0], "WSPD": [10.0]
    })
    X_single = preprocessor.transform(single_row)
    assert X_single.shape == (1, 5)

    pred = model.predict(X_single)
    score = model.decision_function(X_single)
    assert len(pred) == 1
    assert len(score) == 1


def test_edge_case_all_missing_optional_measurements():
    """Verify that a record with missing PRCP is cleanly imputed using training median."""
    preprocessor = WeatherPreprocessor.load()
    model, _ = load_trained_model()

    missing_prcp = pd.DataFrame({
        "TMIN": [10.0], "TAVG": [15.0], "TMAX": [20.0], "PRCP": [np.nan], "WSPD": [12.0]
    })
    X = preprocessor.transform(missing_prcp)
    assert not np.isnan(X).any()
    pred = model.predict(X)
    assert pred[0] in {1, -1}


def test_results_consistency_with_saved_artifacts():
    """
    Verify that loaded artifacts in models/ and results/ match bit-for-bit:
    Saved anomaly count must match model_metadata.json.
    """
    metadata_path = get_models_dir() / "model_metadata.json"
    results_path = get_results_dir() / "anomaly_results_with_locations.csv"

    assert metadata_path.exists()
    assert results_path.exists()

    import json
    with open(metadata_path, "r") as f:
        meta = json.load(f)

    res_df = pd.read_csv(results_path)
    assert len(res_df) == meta["total_records_trained"]
    assert int(res_df["anomaly"].sum()) == meta["flagged_observations"]
    assert round(float(res_df["anomaly_score"].mean()), 4) == round(meta["score_statistics"]["mean"], 4)
