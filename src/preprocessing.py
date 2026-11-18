"""
src/preprocessing.py
Reusable, leakage-free preprocessing pipeline and feature transformation.
"""

import sys
from pathlib import Path

# Add project root to sys.path to support both direct and module execution
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from typing import Dict, List, Optional, Tuple, Union, Any
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler

from src.feature_engineering import compute_engineered_features, get_feature_list
from src.utils import get_models_dir, setup_logger

logger = setup_logger("Preprocessing")


class WeatherPreprocessor(BaseEstimator, TransformerMixin):
    """
    A scikit-learn compatible preprocessor for weather sensor records.
    Ensures zero data leakage by fitting imputation statistics and scaling
    parameters strictly on training data.
    """

    def __init__(
        self,
        feature_mode: str = "baseline",
        scaler_type: str = "standard",
        imputation_strategy: str = "median",
    ):
        self.feature_mode = feature_mode
        self.scaler_type = scaler_type
        self.imputation_strategy = imputation_strategy
        self.feature_names_: List[str] = []
        self.imputer_: Optional[SimpleImputer] = None
        self.scaler_: Optional[Union[StandardScaler, RobustScaler]] = None
        self.medians_: Dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y=None) -> "WeatherPreprocessor":
        """
        Fit the imputer and scaler on the training dataset.
        """
        X_df = X.copy()

        # Generate engineered features if specified
        if self.feature_mode == "engineered":
            X_df = compute_engineered_features(X_df)
            self.feature_names_ = get_feature_list("engineered")
        else:
            self.feature_names_ = get_feature_list("baseline")

        # Validate that all expected features exist in X_df
        available = [col for col in self.feature_names_ if col in X_df.columns]
        if len(available) < len(self.feature_names_):
            missing = set(self.feature_names_) - set(available)
            raise ValueError(f"Features missing from input dataframe: {missing}")

        features_data = X_df[self.feature_names_].replace([np.inf, -np.inf], np.nan)

        # Store training medians for reference and audit
        self.medians_ = features_data.median().to_dict()

        # Fit imputer
        self.imputer_ = SimpleImputer(strategy=self.imputation_strategy)
        imputed_array = self.imputer_.fit_transform(features_data)

        # Fit scaler
        if self.scaler_type == "robust":
            self.scaler_ = RobustScaler()
        else:
            self.scaler_ = StandardScaler()

        self.scaler_.fit(imputed_array)
        logger.info(
            f"Fitted WeatherPreprocessor [mode={self.feature_mode}, "
            f"scaler={self.scaler_type}, features={len(self.feature_names_)}]"
        )
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform new observations using the fitted imputer and scaler.
        """
        if self.scaler_ is None or self.imputer_ is None:
            raise RuntimeError("WeatherPreprocessor must be fitted before calling transform().")

        X_df = X.copy()
        if self.feature_mode == "engineered":
            X_df = compute_engineered_features(X_df)

        for col in self.feature_names_:
            if col not in X_df.columns:
                # If an engineered or raw feature is missing, fill with fitted median
                X_df[col] = self.medians_.get(col, 0.0)

        features_data = X_df[self.feature_names_].replace([np.inf, -np.inf], np.nan)
        imputed = self.imputer_.transform(features_data)
        scaled = self.scaler_.transform(imputed)
        return scaled

    def fit_transform(self, X: pd.DataFrame, y=None) -> np.ndarray:
        """Fit to data, then transform it."""
        return self.fit(X, y).transform(X)

    def get_feature_names(self) -> List[str]:
        """Return list of transformed feature names."""
        return list(self.feature_names_)

    def save(self, filepath: Optional[Path] = None) -> Path:
        """Serialize the fitted preprocessor to disk."""
        if filepath is None:
            filepath = get_models_dir() / "preprocessor.joblib"
        joblib.dump(self, filepath)
        logger.info(f"Preprocessor saved to {filepath}")
        return filepath

    @classmethod
    def load(cls, filepath: Optional[Path] = None) -> "WeatherPreprocessor":
        """Load a serialized preprocessor from disk."""
        if filepath is None:
            filepath = get_models_dir() / "preprocessor.joblib"
        if not filepath.exists():
            raise FileNotFoundError(f"Preprocessor file not found at: {filepath}")
        loaded = joblib.load(filepath)
        logger.info(f"Loaded WeatherPreprocessor from {filepath}")
        return loaded
