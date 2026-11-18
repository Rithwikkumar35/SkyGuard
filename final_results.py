
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).parent
RAW_PATH = BASE / "data" / "weather_data_world_2025-01-01_2025-02-28.csv"
OUT_DIR = BASE / "results"
OUT_DIR.mkdir(exist_ok=True)

raw = pd.read_csv(RAW_PATH)

features = ["TMIN", "TAVG", "TMAX", "PRCP", "WSPD"]

# Convert measurement strings to numbers
for col in features:
    raw[col] = pd.to_numeric(
        raw[col].astype(str)
        .str.replace(",", "", regex=False)
        .str.extract(r"([-+]?\d*\.?\d+)")[0],
        errors="coerce"
    )

# Keep city/date metadata and prepare model features
X = raw[features].replace([np.inf, -np.inf], np.nan).copy()
X = X.dropna(axis=1, how="all")
X = X.loc[:, X.nunique(dropna=True) > 1]
X = X.fillna(X.median())

# Fit the 3% Isolation Forest model
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = IsolationForest(
    n_estimators=150,
    contamination=0.03,
    random_state=42
)
predictions = model.fit_predict(X_scaled)
scores = model.decision_function(X_scaled)

# Preserve source-row alignment and metadata
results = raw.loc[X.index, ["City", "Country", "DATE"]].copy()
results = results.join(X)
results["anomaly_score"] = scores
results["anomaly"] = predictions == -1

results = results.sort_values("anomaly_score")
results.to_csv(OUT_DIR / "anomaly_results_with_locations.csv", index=False)

print("Records analyzed:", len(results))
print("Flagged observations:", int(results["anomaly"].sum()))
print("\nMost unusual observations:")
print(
    results[results["anomaly"]]
    .head(15)
    .round(3)
    .to_string(index=False)
)
print("\nSaved: results/anomaly_results_with_locations.csv")
