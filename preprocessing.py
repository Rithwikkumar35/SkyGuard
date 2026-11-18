
import pandas as pd
import numpy as np
from pathlib import Path

# 1. Load the original weather dataset
DATA_PATH = Path(__file__).parent / "data" / (
    "weather_data_world_2025-01-01_2025-02-28.csv"
)

df = pd.read_csv(DATA_PATH)

print("=" * 55)
print("SKYGUARD: DATA PREPROCESSING AND ANALYSIS")
print("=" * 55)

# 2. Understand the original dataset
print("\n1. ORIGINAL DATA")
print("Dataset shape:", df.shape)
print("Unique cities:", df["City"].nunique())
print("Duplicate rows:", df.duplicated().sum())
print("Date range:", df["DATE"].min(), "to", df["DATE"].max())

# 3. Convert weather measurements to numeric values
weather_columns = [
    "TMIN", "TAVG", "TMAX", "PRCP",
    "SNOW", "WSPD", "WPGT", "PRES", "TSUN"
]

for column in weather_columns:
    df[column] = pd.to_numeric(
        df[column].astype(str)
        .str.replace(",", "", regex=False)
        .str.extract(r"([-+]?\d*\.?\d+)")[0],
        errors="coerce"
    )

# 4. Count unavailable measurements
print("\n2. MISSING VALUES AFTER CONVERSION")
print(df[weather_columns].isna().sum().to_string())

# 5. Calculate descriptive statistics
features = ["TMIN", "TAVG", "TMAX", "PRCP", "WSPD"]

print("\n3. DESCRIPTIVE STATISTICS")
print(df[features].describe().round(2).to_string())

# 6. Prepare model input
model_data = df[features].replace(
    [np.inf, -np.inf], np.nan
).copy()

# Remove columns that have no usable values or no variation
model_data = model_data.dropna(axis=1, how="all")
model_data = model_data.loc[
    :, model_data.nunique(dropna=True) > 1
]

# Fill missing values with each feature's median
model_data = model_data.fillna(model_data.median())

# 7. Save prepared data
output_path = Path(__file__).parent / "data" / "weather_cleaned.csv"
model_data.to_csv(output_path, index=False)

print("\n4. PREPARED MODEL INPUT")
print("Rows:", len(model_data))
print("Features:", model_data.columns.tolist())
print("Missing values remaining:", model_data.isna().sum().sum())
print("Saved to:", output_path)
print("\nPreprocessing completed successfully.")
