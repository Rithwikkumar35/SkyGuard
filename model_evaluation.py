
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "weather_cleaned.csv"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# 1. Load prepared data
df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("SKYGUARD: ISOLATION FOREST MODEL EVALUATION")
print("=" * 60)
print("\nDataset shape:", df.shape)
print("Features:", df.columns.tolist())

# 2. Validate model input
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()

if len(df) < 10:
    raise ValueError("At least 10 complete records are required.")

# 3. Standardize features for consistent model input
scaler = StandardScaler()
X = scaler.fit_transform(df)

print("\nStandardization completed.")
print("Rows used:", len(X))

# 4. Compare different contamination settings
contamination_values = [0.01, 0.03, 0.05, 0.10]
comparison = []
saved_results = {}

for contamination in contamination_values:
    model = IsolationForest(
        n_estimators=150,
        contamination=contamination,
        random_state=42
    )

    predictions = model.fit_predict(X)
    scores = model.decision_function(X)
    anomaly_flags = predictions == -1

    count = int(anomaly_flags.sum())
    percentage = count / len(df) * 100

    comparison.append({
        "contamination_setting": contamination,
        "records_analyzed": len(df),
        "flagged_records": count,
        "flagged_percentage": round(percentage, 2),
        "mean_anomaly_score": round(float(scores.mean()), 4)
    })

    if contamination == 0.03:
        saved_results = df.copy()
        saved_results["anomaly_score"] = scores
        saved_results["anomaly"] = anomaly_flags

# 5. Print comparison
comparison_df = pd.DataFrame(comparison)

print("\n1. CONTAMINATION COMPARISON")
print(comparison_df.to_string(index=False))

comparison_df.to_csv(
    RESULTS_DIR / "contamination_comparison.csv",
    index=False
)

# 6. Save the results for the 3% setting
saved_results.to_csv(
    RESULTS_DIR / "anomaly_results.csv",
    index=False
)

# 7. Plot flagged records by setting
plt.figure(figsize=(9, 5))
plt.bar(
    comparison_df["contamination_setting"].astype(str),
    comparison_df["flagged_records"]
)
plt.title("Isolation Forest: Flagged Records by Contamination Setting")
plt.xlabel("Contamination setting")
plt.ylabel("Number of flagged records")
plt.tight_layout()
plt.savefig(
    RESULTS_DIR / "contamination_comparison.png",
    dpi=150
)
plt.close()

# 8. Show the most unusual records
print("\n2. MOST UNUSUAL RECORDS AT 3% SETTING")
print(
    saved_results[saved_results["anomaly"]]
    .sort_values("anomaly_score")
    .head(10)
    .round(4)
    .to_string(index=False)
)

print("\n3. SAVED OUTPUTS")
print("results/contamination_comparison.csv")
print("results/anomaly_results.csv")
print("results/contamination_comparison.png")
print("\nModel training and comparison completed.")
print(
    "\nImportant: without verified anomaly labels, these results "
    "measure model output, not classification accuracy."
)
