"""
tests/test_model.py
Unit tests for model training, inference, serialization, reproducibility, and comparative algorithms.
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
from src.preprocessing import WeatherPreprocessor
from src.train import train_isolation_forest, load_trained_model
from src.evaluate import evaluate_contamination_sensitivity, compare_unsupervised_algorithms


def test_isolation_forest_training():
    """Verify Isolation Forest training, record count, and anomaly flag rate."""
    df = load_raw_dataset()
    model, preprocessor, results_df, metadata = train_isolation_forest(
        df=df,
        contamination=0.03,
        n_estimators=100,
        random_state=42,
        save_artifacts=False,
    )

    assert len(results_df) == 6077
    assert "anomaly_score" in results_df.columns
    assert "anomaly" in results_df.columns
    assert "severity" in results_df.columns
    assert results_df["anomaly"].sum() == 183
    assert metadata["flagged_percentage"] == pytest.approx(3.01, abs=0.05)


def test_model_serialization_and_loading(tmp_path):
    """Verify serializing and loading trained model and metadata."""
    df = load_raw_dataset()
    model, _, _, metadata = train_isolation_forest(
        df=df,
        contamination=0.03,
        n_estimators=50,
        random_state=42,
        save_artifacts=False,
    )

    import joblib
    import json

    m_path = tmp_path / "test_model.joblib"
    meta_path = tmp_path / "test_metadata.json"

    joblib.dump(model, m_path)
    with open(meta_path, "w") as f:
        json.dump(metadata, f)

    loaded_model, loaded_meta = load_trained_model(m_path, meta_path)

    assert loaded_model.n_estimators == 50
    assert loaded_meta["contamination_setting"] == 0.03
    assert loaded_meta["flagged_observations"] == 183


def test_anomaly_score_invariants():
    """Verify mathematical properties of decision function scores and anomaly flags."""
    df = load_raw_dataset()
    _, _, results_df, _ = train_isolation_forest(
        df=df,
        contamination=0.03,
        n_estimators=50,
        random_state=42,
        save_artifacts=False,
    )

    flagged = results_df[results_df["anomaly"]]
    normal = results_df[~results_df["anomaly"]]

    # In scikit-learn IsolationForest, negative decision_function values are anomalies
    assert (flagged["anomaly_score"] <= 0.0).all()
    assert (normal["anomaly_score"] > 0.0).all()
    assert flagged["anomaly_score"].mean() < normal["anomaly_score"].mean()


def test_reproducibility():
    """Verify that identical random_state yields identical anomaly scores."""
    df = load_raw_dataset()
    _, _, res1, _ = train_isolation_forest(
        df=df, contamination=0.03, n_estimators=50, random_state=42, save_artifacts=False
    )
    _, _, res2, _ = train_isolation_forest(
        df=df, contamination=0.03, n_estimators=50, random_state=42, save_artifacts=False
    )

    assert np.allclose(res1["anomaly_score"].to_numpy(), res2["anomaly_score"].to_numpy())
    assert (res1["anomaly"] == res2["anomaly"]).all()


def test_contamination_sensitivity_range():
    """Verify that increasing contamination increases flagged anomaly count proportionally."""
    df = load_raw_dataset()
    comp_df = evaluate_contamination_sensitivity(df, contamination_levels=[0.01, 0.05])

    count_1 = comp_df.loc[comp_df["contamination_setting"] == 0.01, "flagged_records"].values[0]
    count_5 = comp_df.loc[comp_df["contamination_setting"] == 0.05, "flagged_records"].values[0]

    assert count_5 > count_1
    assert count_1 == 61
    assert count_5 == 304


def test_comparison_algorithms_execution():
    """Verify that comparative benchmark (IF, LOF, DBSCAN) completes and returns valid results."""
    df = load_raw_dataset()
    comp_df, overlap_info = compare_unsupervised_algorithms(df, contamination=0.03)

    assert len(comp_df) == 3
    assert set(comp_df["algorithm"]) == {"Isolation Forest", "Local Outlier Factor", "DBSCAN"}
    assert "jaccard_if_lof" in overlap_info
    assert overlap_info["if_count"] == 183
    assert overlap_info["lof_count"] == 183


def test_inference_on_new_observations():
    """Verify that a fitted preprocessor and model can predict on unseen test data."""
    df = load_raw_dataset()
    model, preprocessor, _, _ = train_isolation_forest(
        df=df, contamination=0.03, n_estimators=50, random_state=42, save_artifacts=False
    )

    # Synthetic observations: 1 normal day, 1 extreme polar cold day (-40°C), 1 extreme rainstorm (150 mm)
    new_obs = pd.DataFrame({
        "City": ["TestCity", "TestCity", "TestCity"],
        "TMIN": [10.0, -40.0, 15.0],
        "TAVG": [15.0, -35.0, 18.0],
        "TMAX": [20.0, -30.0, 22.0],
        "PRCP": [0.0, 0.0, 150.0],
        "WSPD": [10.0, 15.0, 65.0],
    })

    X_new = preprocessor.transform(new_obs)
    preds = model.predict(X_new)
    scores = model.decision_function(X_new)

    assert len(preds) == 3
    assert len(scores) == 3
    # Extreme cold and extreme storm should be flagged as anomalies (-1)
    assert preds[1] == -1  # -40°C cold day
    assert preds[2] == -1  # 150mm rain + 65km/h storm
    assert scores[1] < scores[0]
    assert scores[2] < scores[0]
