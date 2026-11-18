"""
tests/test_preprocessing.py
Unit tests for feature engineering, preprocessing pipeline, and serialization.
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

from src.data_loader import load_raw_dataset
from src.feature_engineering import compute_engineered_features, get_feature_list
from src.preprocessing import WeatherPreprocessor


def test_feature_engineering_computation():
    """Verify mathematical calculation of engineered meteorological features."""
    sample = pd.DataFrame({
        "City": ["Rome", "Rome"],
        "TMIN": [5.0, 10.0],
        "TAVG": [10.0, 14.0],
        "TMAX": [15.0, 20.0],
        "PRCP": [0.0, 9.0],
        "WSPD": [12.0, 15.0],
    })
    feat_df = compute_engineered_features(sample)

    # TEMP_RANGE = TMAX - TMIN
    assert feat_df["TEMP_RANGE"].iloc[0] == 10.0
    assert feat_df["TEMP_RANGE"].iloc[1] == 10.0

    # TEMP_DEVIATION = TAVG - (TMAX + TMIN)/2
    assert feat_df["TEMP_DEVIATION"].iloc[0] == 0.0  # 10 - 10
    assert feat_df["TEMP_DEVIATION"].iloc[1] == -1.0  # 14 - 15

    # PRCP_LOG = log1p(PRCP)
    assert feat_df["PRCP_LOG"].iloc[0] == pytest.approx(0.0)
    assert feat_df["PRCP_LOG"].iloc[1] == pytest.approx(np.log1p(9.0))


def test_preprocessor_fit_transform_baseline():
    """Verify baseline preprocessing: shape, scaling, zero missing values."""
    df = load_raw_dataset()
    preprocessor = WeatherPreprocessor(feature_mode="baseline", scaler_type="standard")
    X = preprocessor.fit_transform(df)

    assert X.shape == (6077, 5)
    assert not np.isnan(X).any()
    # Means should be approximately 0 and standard deviations approximately 1
    assert np.allclose(X.mean(axis=0), 0.0, atol=1e-3)
    assert np.allclose(X.std(axis=0), 1.0, atol=1e-3)


def test_preprocessor_fit_transform_engineered():
    """Verify engineered preprocessing produces 9 features and valid scaling."""
    df = load_raw_dataset()
    preprocessor = WeatherPreprocessor(feature_mode="engineered", scaler_type="standard")
    X = preprocessor.fit_transform(df)

    assert X.shape == (6077, 9)
    assert not np.isnan(X).any()
    assert len(preprocessor.get_feature_names()) == 9


def test_missing_value_imputation():
    """Verify that fitted medians are correctly imputed without data leakage."""
    train_data = pd.DataFrame({
        "TMIN": [10.0, 20.0, 30.0],
        "TAVG": [15.0, 25.0, 35.0],
        "TMAX": [20.0, 30.0, 40.0],
        "PRCP": [0.0, 5.0, 10.0],  # Median is 5.0
        "WSPD": [10.0, 15.0, 20.0],
    })
    test_data = pd.DataFrame({
        "TMIN": [15.0],
        "TAVG": [20.0],
        "TMAX": [25.0],
        "PRCP": [np.nan],  # Missing! Should be imputed with 5.0
        "WSPD": [12.0],
    })

    preprocessor = WeatherPreprocessor(feature_mode="baseline")
    preprocessor.fit(train_data)
    assert preprocessor.medians_["PRCP"] == 5.0

    X_test = preprocessor.transform(test_data)
    assert not np.isnan(X_test).any()
    assert X_test.shape == (1, 5)


def test_preprocessor_serialization(tmp_path):
    """Verify saving and loading fitted preprocessor."""
    df = load_raw_dataset()
    preprocessor = WeatherPreprocessor(feature_mode="baseline")
    X_orig = preprocessor.fit_transform(df)

    save_path = tmp_path / "test_preprocessor.joblib"
    preprocessor.save(save_path)

    loaded = WeatherPreprocessor.load(save_path)
    X_loaded = loaded.transform(df)

    assert np.allclose(X_orig, X_loaded)
    assert loaded.get_feature_names() == preprocessor.get_feature_names()


def test_invalid_input_handling():
    """Verify that ValueError is raised when essential features are missing."""
    invalid_data = pd.DataFrame({"City": ["Test"], "Elevation": [100]})
    preprocessor = WeatherPreprocessor(feature_mode="baseline")
    with pytest.raises(ValueError):
        preprocessor.fit(invalid_data)
