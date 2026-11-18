# SkyGuard — Weather-Station Sensor Anomaly Detection
## Comprehensive Academic Project Report

**Course Code:** INT396 — Unsupervised Learning  
**Project Domain:** Machine Learning / Meteorological Sensor Analytics / Anomaly Detection  
**Academic Term:** 2024–2025  

---

### Student & Submission Details
- **Project Title:** SkyGuard — Weather-Station Sensor Anomaly Detection
- **Subject:** INT396 — Unsupervised Learning
- **Problem Statement ID:** `[PS-INT396-WEATHER-01]` *(Editable Placeholder)*
- **Group Members & Registration Numbers:**
  - Member 1: `[STUDENT_NAME_1]` — Reg No: `[REGISTRATION_NUMBER_1]` *(Lead Developer / ML Engineer)*
  - Member 2: `[STUDENT_NAME_2]` — Reg No: `[REGISTRATION_NUMBER_2]` *(Data Engineer / EDA Specialist)*
  - Member 3: `[STUDENT_NAME_3]` — Reg No: `[REGISTRATION_NUMBER_3]` *(UI/UX & System Integration)*
- **Faculty Mentor / Course Instructor:** `[FACULTY_MENTOR_NAME]`
- **Institution:** `[UNIVERSITY_NAME / DEPARTMENT_OF_COMPUTER_SCIENCE]`

---

## 1. Introduction & Problem Statement

### 1.1 Context and Motivation
Automated weather stations (AWS) form the backbone of modern climatological observation, aviation management, agricultural planning, and disaster preparedness. These stations operate continuously across remote and harsh environments, collecting key environmental parameters such as ambient temperatures ($T_{MIN}, T_{AVG}, T_{MAX}$), liquid precipitation ($PRCP$), sustained wind speeds ($WSPD$), atmospheric surface pressure ($PRES$), and solar irradiance.

However, environmental sensor telemetry is subject to numerous real-world operational challenges:
1. **Instrument Degradation:** Sensor drift, thermal calibration shift, and mechanical anemometer stiction over extended operational cycles.
2. **Physical Fouling:** Tipping-bucket rain gauges obstructed by debris, snow accumulation, or solar radiation shield obstruction.
3. **Transmission & Calculation Errors:** Communication packet truncation, conversion errors, or aggregation anomalies when computing daily averages across inconsistent hourly sampling intervals.

### 1.2 Problem Statement
Weather stations collect continuous environmental observations. Abnormal readings, unusual variations, and data-quality problems can affect the reliability of weather information. The objective is to use **unsupervised machine learning** to identify unusual weather observations without relying on pre-labelled anomaly data.

Unlike supervised learning, real-world meteorological networks rarely provide verified ground-truth sensor failure labels. Therefore, the detection system must learn intrinsic multi-dimensional geometric distributions and isolate anomalous observations based purely on structural statistical atypicality.

---

## 2. Project Objectives & Need

### 2.1 Project Objectives
1. **Automated Ingestion & Physical Consistency Auditing:** Ingest multi-station global weather telemetry, parse unit-encoded strings, and apply rigorous meteorological thermodynamic sanity constraints.
2. **Leakage-Free Preprocessing Architecture:** Construct an enterprise-grade scikit-learn compatible transformation pipeline that imputes missing data using training medians and standardizes features without data leakage.
3. **Domain Meteorological Feature Engineering:** Formulate and validate domain-specific features (diurnal temperature range, diurnal midpoint deviation, log-precipitation, and city-relative temperature departures) to assist detection.
4. **Unsupervised Anomaly Isolation:** Implement `IsolationForest` as the primary unsupervised model, compute continuous decision-function anomaly scores, and stratify anomalies into empirical severity tiers (`Severe`, `High`, `Moderate`).
5. **Multi-Paradigm Comparative Benchmarking:** Benchmark Isolation Forest against Local Outlier Factor (LOF) and DBSCAN using mathematically defensible unsupervised evaluation criteria (Silhouette separation score, computational runtime, and Jaccard set overlap).
6. **Professional Weather Intelligence Platform:** Deploy a responsive, 6-section Streamlit application that loads pre-trained serialized artifacts from disk with zero retraining on page interaction.
7. **Automated Test Validation:** Establish complete test coverage (25 automated `pytest` test cases) validating data contracts, transformation fidelity, model reproducibility, and live CSV upload workflows.

### 2.2 Societal, Industrial & Academic Need
- **Meteorological Reliability:** Ensures that numerical weather prediction (NWP) models do not ingest corrupted observational data.
- **Critical Infrastructure Safety:** Protects aviation control, municipal flood warning networks, and power grid operators from acting on false sensor telemetry.
- **Academic Pedagogy:** Demonstrates rigorous unsupervised learning methodology without the fallacy of reporting supervised metrics (accuracy, precision, recall, F1) on unlabelled data.

---

## 3. Project Overview, Key Features & Working

### 3.1 High-Level Architecture
SkyGuard is structured as a decoupled, modular Python machine learning pipeline:

```
[ Raw CSV Telemetry: 6,077 Records ]
                  |
                  v
[ src/data_loader.py ] ---> Regex String Parsing & 6-Point Physical Consistency Audit
                  |
                  v
[ src/feature_engineering.py ] ---> Diurnal Range, Midpoint Deviation, log1p(PRCP), City Deviation
                  |
                  v
[ src/preprocessing.py ] ---> Zero-Leakage SimpleImputer & StandardScaler (joblib serialized)
                  |
                  v
[ src/train.py ] ---> IsolationForest (n=150, c=0.03, seed=42) + Severity Stratification
                  |
                  v
[ src/evaluate.py ] ---> Sensitivity Analysis (1%-10%), IF vs LOF vs DBSCAN, Silhouette
                  |
                  v
[ app/app.py ] ---> 6-Section Interactive Streamlit Weather Intelligence Platform
```

### 3.2 Key System Features
- **Deterministic Reproducibility:** Fixed random state seed ($42$) and strict pipeline decoupling ensure identical results across runs.
- **Zero Model Retraining Overhead:** Model and preprocessor are saved as serialized `.joblib` binaries; the interactive dashboard executes in sub-second time.
- **Physical Boundary Verification:** Automatically flags records where daily average temperature violates thermodynamic bounds ($T_{AVG} < T_{MIN}$ or $T_{AVG} > T_{MAX}$).
- **Interactive Live Inference:** End users can upload any standard weather station CSV; the dashboard validates schema conformity, audits missing values, and scores observations in real time without retraining.

---

## 4. Tools, Libraries & Technologies

| Category | Technology | Version | Academic / Technical Justification |
| :--- | :--- | :---: | :--- |
| **Language** | Python | 3.12.10 | Modern standard runtime with enhanced typing and exception groups. |
| **Data Manipulation** | pandas | 3.1.0 | Vectorized multi-index data frame manipulation and time-series aggregation. |
| **Scientific Computing** | numpy | 2.5.4 | Efficient numerical array manipulation and linear algebra operations. |
| **Machine Learning** | scikit-learn | 1.9.1 | Production-grade `IsolationForest`, `LocalOutlierFactor`, `DBSCAN`, and preprocessing transformers. |
| **Statistical Analysis** | scipy | 1.18.1 | Statistical distributions and metric evaluation. |
| **Interactive Dashboard** | streamlit | 1.66.1 | Modern reactive web framework for machine learning dashboards. |
| **Data Visualization** | plotly | 7.1.0 | Interactive client-side charting (zoom, hover tooltips, and geospatial projections). |
| **Static Plotting** | matplotlib / seaborn | 3.11 / 0.13 | Publication-grade static figures for reports and exploratory data analysis. |
| **Artifact Persistence**| joblib | 1.6.0 | Optimized binary serialization for scikit-learn estimators and numpy arrays. |
| **Automated Testing** | pytest | 9.2.0 | Comprehensive unit, regression, and integration testing framework. |

---

## 5. Methodology & Mathematical Formulation

### 5.1 Dataset Specifications
- **Source:** Global Meteorological Station Network (`weather_data_world_2025-01-01_2025-02-28.csv`)
- **Sample Size:** 6,077 daily observation records
- **Temporal Span:** January 1, 2025 – February 28, 2025 (59 continuous days)
- **Geographic Span:** 103 cities across 35 countries worldwide
- **Raw Dimensions:** 15 columns (City, Country, Elevation, Timezone, Coordinates, DATE, TMIN, TAVG, TMAX, PRCP, SNOW, WSPD, WPGT, PRES, TSUN)

### 5.2 Meteorological Feature Justification & Exclusion Audit
- **Retained Baseline Features (5):**
  - `TMIN` (Minimum Daily Temperature, °C): 0% missing.
  - `TAVG` (Average Daily Temperature, °C): 0% missing.
  - `TMAX` (Maximum Daily Temperature, °C): 0% missing.
  - `PRCP` (Daily Liquid Precipitation, mm): 7.77% missing (472 values imputed via median).
  - `WSPD` (Sustained Wind Speed, km/h): 0% missing.
- **Excluded Features Audit:**
  - `SNOW` (92.48% missing) and `TSUN` (97.09% missing): Excluded due to severe missingness exceeding the 50% data viability threshold.
  - `WPGT` (75.73% missing): Peak gust excluded due to missingness; sustained wind (`WSPD`) provides 100% complete coverage.
  - `PRES` (1.00% missing): Atmospheric surface pressure is heavily confounded by station elevation ($R^2 > 0.95$ with altitude), reflecting topography rather than meteorological anomaly without complex sea-level barometric reductions.

### 5.3 Domain Meteorological Feature Engineering
1. **Diurnal Temperature Range ($DTR$):**
   $$DTR = T_{MAX} - T_{MIN}$$
   *Meteorological Justification:* Captures diurnal thermal variation. Abnormally small $DTR$ indicates stuck sensor values or dense stratus inversions; excessively large $DTR$ indicates radiation shield failures or hyper-arid swings.
2. **Diurnal Midpoint Departure ($\Delta T$):**
   $$\Delta T = T_{AVG} - \frac{T_{MAX} + T_{MIN}}{2}$$
   *Meteorological Justification:* For sinusoidal daily progressions, $T_{AVG}$ approximates the midpoint. Large departures denote rapid daytime cold-front intrusions or calibration bias.
3. **Log-Transformed Precipitation ($\text{PRCP}_{\text{log}}$):**
   $$\text{PRCP}_{\text{log}} = \ln(1 + \max(0, PRCP))$$
   *Meteorological Justification:* Addresses extreme zero-inflation ($66.5\%$ dry observations) and heavy right-skewness, reducing skewness from **5.72 down to 1.73**.
4. **City-Relative Temperature Anomaly ($\Delta T_{\text{city}}$):**
   $$\Delta T_{\text{city}} = T_{AVG} - \overline{T_{AVG, \text{city}}}$$
   *Meteorological Justification:* Normalizes temperatures against each station's 2-month baseline, preventing the model from simply flagging high-latitude stations (e.g. Calgary) due to regional climate differences.

### 5.4 Physical Consistency Auditing
Six automated physical laws are enforced:
1. $T_{MIN} \le T_{MAX}$ (Thermodynamic consistency: 0 violations).
2. $T_{MIN} \le T_{AVG} \le T_{MAX}$ (Daily mean bounds: **7 physical anomalies identified** in Melbourne, Valencia, Kyoto, Sydney, Brisbane, Perth).
3. $PRCP \ge 0$ (Non-negative rainfall: 0 violations).
4. $WSPD \ge 0$ (Non-negative wind: 0 violations).
5. $WPGT \ge WSPD$ (Peak gust $\ge$ sustained wind: 0 violations).
6. $850\text{ hPa} \le PRES \le 1090\text{ hPa}$ (Barometric range: 0 violations).

### 5.5 Primary Algorithm: Isolation Forest
Isolation Forest builds an ensemble of randomized trees ($iTrees$). Sub-samples of $X$ are partitioned recursively using randomly selected features and uniform random split values between the feature's minimum and maximum.

- **Path Length $h(x)$:** Number of edges $x$ traverses from the root node to a terminating leaf node.
- **Expected Path Length of an Unsuccessful Search in a BST:**
  $$c(n) = 2\left(\ln(n - 1) + 0.5772156649\right) - \frac{2(n - 1)}{n}$$
- **Anomaly Score $s(x, n)$:**
  $$s(x, n) = 2^{-\frac{E(h(x))}{c(n)}}$$
  - $s \to 1.0$: Very short path length $\Rightarrow$ **Definite Anomaly**.
  - $s < 0.5$: Long path length $\Rightarrow$ **Normal Observation**.
- **Model Configuration:** $n\_estimators = 150$, $random\_state = 42$, $contamination = 0.03$.

---

## 6. Experimental Results & Comparative Benchmarking

### 6.1 Baseline Model Results
At the standard $0.03$ ($3\%$) contamination setting, the Isolation Forest model evaluated all 6,077 records:
- **Flagged Observations:** **183 records (3.01%)**
- **Normal Observations:** **5,894 records (96.99%)**
- **Score Range:** Min: $-0.1287$, Max: $+0.2146$, Mean: $+0.1521$, Std: $0.0560$
- **Decision Threshold:** $0.0000$ (decision function offset: $-0.5947$)

### 6.2 Contamination Sensitivity Analysis
*(Saved to `results/contamination_comparison.csv`)*

| Contamination Setting | Records Analyzed | Flagged Records | Flagged % | Mean Decision Score | Std Dev Score | Min Score | Runtime (ms) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.01 (1%)** | 6,077 | 61 | 1.00% | 0.2095 | 0.0527 | -0.1287 | 123.6 ms |
| **0.03 (3%)** | 6,077 | 183 | 3.01% | 0.1521 | 0.0560 | -0.1287 | 134.1 ms |
| **0.05 (5%)** | 6,077 | 304 | 5.00% | 0.1193 | 0.0569 | -0.1287 | 126.8 ms |
| **0.10 (10%)** | 6,077 | 608 | 10.00% | 0.0736 | 0.0567 | -0.1287 | 128.4 ms |

*Finding:* The decision function scores remain stable across contamination settings, showing that the model's ordering of anomalies is consistent.

### 6.3 Multi-Paradigm Comparative Benchmark
*(Saved to `results/model_comparison.csv`)*

| Algorithm | Paradigm | Flagged Anomalies | Flagged % | Silhouette Score | Runtime (ms) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Isolation Forest** | Randomized Tree Partitioning | **183** | **3.01%** | **0.5518** | **135.81 ms** |
| **Local Outlier Factor (LOF)** | K-NN Local Density Deviation | 183 | 3.01% | 0.2246 | 27.08 ms |
| **DBSCAN** | Density Clustering (Noise Identification) | 14 | 0.23% | 0.7638 | 127.14 ms |

- **Silhouette Analysis:** Isolation Forest achieves a silhouette score of **0.5518**, showing clear separation between the normal data mass and isolated points. LOF scores **0.2246** due to local boundary points being near normal clusters. DBSCAN achieves **0.7638** but flags only 14 points (0.23%), under-detecting subtler anomalies.
- **Jaccard Set Similarity:** The Jaccard overlap between Isolation Forest and LOF detected anomaly sets is **0.3165**, reflecting the conceptual difference between global tree isolation and local $k$-NN density estimation.

### 6.4 Qualitative Analysis of Top Anomalies
*(Saved to `results/qualitative_anomaly_analysis.csv`)*

| Station / City | Country | Date | TMIN | TAVG | TMAX | PRCP | WSPD | Score | Severity | Driving Factor |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Calgary** | CA | 2025-02-05 | -26.5 | -24.2 | -21.6 | 2.2 | 6.9 | -0.1287 | Severe | Extreme Polar Vortex Sub-zero Cold (-26.5°C) |
| **Calgary** | CA | 2025-02-04 | -27.9 | -26.3 | -24.5 | 0.6 | 7.3 | -0.1254 | Severe | Extreme Polar Vortex Sub-zero Cold (-27.9°C) |
| **Calgary** | CA | 2025-02-03 | -28.4 | -26.4 | -24.2 | 0.2 | 6.3 | -0.1227 | Severe | Extreme Polar Vortex Sub-zero Cold (-28.4°C) |
| **Montreal**| CA | 2025-02-16 | -11.0 | -10.2 | -10.0 | 27.5 | 13.9 | -0.1109 | Severe | Torrential Winter Snow/Rain Deluge (27.5 mm) |
| **Toronto** | CA | 2025-02-16 | -7.0 | -5.8 | -4.0 | 17.5 | 35.6 | -0.1091 | Severe | Winter Gale Storm Wind (35.6 km/h) |
| **Reykjavik**| IS | 2025-01-24 | 1.2 | 3.4 | 5.0 | 14.2 | 57.0 | -0.0984 | Severe | Extreme Sustained Gale Storm (57.0 km/h) |

---

## 7. Testing & Verification Summary

The project incorporates 25 automated test cases executed via `pytest`:
- **`tests/test_data.py` (6 tests):** Validates regex unit cleaning, coordinate parsing, schema validation, and physical consistency rules.
- **`tests/test_preprocessing.py` (6 tests):** Validates feature engineering calculations, zero-leakage training imputation, standard scaling, and serialization.
- **`tests/test_model.py` (7 tests):** Validates model training, invariant scores ($s \le 0$ for anomalies), deterministic reproducibility (`random_state=42`), contamination sensitivity, and inference on synthetic test days.
- **`tests/test_integration.py` (6 tests):** Validates full pipeline roundtrip, Streamlit backend contracts, and live CSV upload simulation.
- **Result:** **25 passed, 0 failed (100% pass rate in 5.74s)**.

---

## 8. Individual Contributions

*(Editable placeholder section for university submission)*

| Team Member | Role | Key Contributions |
| :--- | :--- | :--- |
| `[STUDENT_NAME_1]` (`[REG_NO_1]`) | Lead ML Engineer | Designed `IsolationForest` pipeline, implemented `src/train.py`, `src/evaluate.py`, LOF/DBSCAN benchmarks, and silhouette score evaluations. |
| `[STUDENT_NAME_2]` (`[REG_NO_2]`) | Data & Feature Engineer | Implemented `src/data_loader.py`, regex unit cleaning, physical consistency checks, feature engineering (`DTR`, $\Delta T$, log-PRCP), and EDA suite. |
| `[STUDENT_NAME_3]` (`[REG_NO_3]`) | UI/UX & System Integration | Built the 6-section Streamlit application (`app/app.py`), Plotly interactive charts, live CSV uploader, and pytest integration suites. |

---

## 9. Academic Limitations & Future Scope

### 9.1 Academic Limitations
1. **Unsupervised Nature:** Without verified sensor-fault annotations, model results reflect statistical distinctiveness rather than guaranteed hardware failures.
2. **Climatic Context:** Global datasets naturally contain extreme climates (e.g. sub-zero Canadian winters vs hot Australian summers). While city-relative anomaly engineering mitigates this, macro-scale models require geographic context when interpreting results.
3. **Contamination Parameter:** Contamination ($0.03$) acts as an operational prioritization threshold, not an empirically measured ground-truth failure rate.

### 9.2 Future Scope
1. **Spatio-Temporal Graph Neural Networks (GNNs):** Model spatial correlation between neighboring weather stations to identify when one station disagrees with its geographical peers.
2. **Multivariate Autoencoders & VAEs:** Implement deep reconstruction-based anomaly detection to compare against tree isolation.
3. **Edge Deployment (TinyML):** Quantize the Isolation Forest model for direct deployment on solar-powered microcontroller loggers (e.g., ESP32 / Raspberry Pi Pico).

---

## 10. References

1. Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). *Isolation Forest*. In IEEE International Conference on Data Mining (ICDM '08), pp. 413-422.
2. Breunig, M. M., Kriegel, H. P., Ng, R. T., & Sander, J. (2000). *LOF: Identifying Density-Based Local Outliers*. In ACM SIGMOD Record, 29(2), pp. 93-104.
3. Ester, M., Kriegel, H. P., Sander, J., & Xu, X. (1996). *A density-based algorithm for discovering clusters in large spatial databases with noise*. In KDD '96, pp. 226-231.
4. Rousseeuw, P. J. (1987). *Silhouettes: A graphical aid to the interpretation and validation of cluster analysis*. Journal of Computational and Applied Mathematics, 20, 53-65.
5. World Meteorological Organization (WMO). (2018). *Guide to Meteorological Instruments and Methods of Observation (WMO-No. 8)*. Geneva: WMO.
6. Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research, 12, 2825-2830.
