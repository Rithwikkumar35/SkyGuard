
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "weather_cleaned.csv"
OUTPUT_DIR = BASE_DIR / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)

print("=" * 55)
print("SKYGUARD: EXPLORATORY DATA ANALYSIS")
print("=" * 55)

# 1. Summary statistics
print("\n1. DESCRIPTIVE STATISTICS")
print(df.describe().round(2).to_string())

# 2. Histograms
df.hist(figsize=(12, 8), bins=30, edgecolor="black")
plt.suptitle("Weather Feature Distributions")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "feature_distributions.png", dpi=150)
plt.close()

# 3. Box plots to inspect extreme values
plt.figure(figsize=(11, 6))
df.boxplot()
plt.title("Weather Features: Box Plot Analysis")
plt.ylabel("Measurement value (original feature units)")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "weather_boxplots.png", dpi=150)
plt.close()

# 4. Correlation matrix
correlation = df.corr(numeric_only=True)
print("\n2. CORRELATION MATRIX")
print(correlation.round(2).to_string())

plt.figure(figsize=(8, 6))
image = plt.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(image, label="Correlation")
plt.xticks(range(len(correlation.columns)), correlation.columns, rotation=45)
plt.yticks(range(len(correlation.index)), correlation.index)

for i in range(len(correlation.index)):
    for j in range(len(correlation.columns)):
        plt.text(j, i, f"{correlation.iloc[i, j]:.2f}",
                 ha="center", va="center", fontsize=9)

plt.title("Weather Feature Correlation Matrix")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "correlation_matrix.png", dpi=150)
plt.close()

print("\n3. EDA OUTPUTS")
print("Saved:", OUTPUT_DIR / "feature_distributions.png")
print("Saved:", OUTPUT_DIR / "weather_boxplots.png")
print("Saved:", OUTPUT_DIR / "correlation_matrix.png")
print("\nEDA completed successfully.")
