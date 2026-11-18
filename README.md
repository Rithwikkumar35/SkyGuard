# 🛡️ SkyGuard — Weather-Station Sensor Anomaly Detection

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.66-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.9-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Tests Passing](https://img.shields.io/badge/pytest-25%20passed-brightgreen.svg)](tests/)

> **Academic Project Submission for INT396: Unsupervised Learning (2024–2025)**  
> An end-to-end unsupervised machine learning system designed to discover sensor malfunctions, physical inconsistencies, and extreme climatological events across global weather station telemetry without ground-truth labels.

---

## 📋 Table of Contents
- [1. Project Overview](#1-project-overview)
- [2. Problem Statement & Need](#2-problem-statement--need)
- [3. Key Features](#3-key-features)
- [4. System Architecture](#4-system-architecture)
- [5. Technology Stack](#5-technology-stack)
- [6. Dataset Information](#6-dataset-information)
- [7. Installation & Setup](#7-installation--setup)
- [8. Execution Instructions](#8-execution-instructions)
- [9. Automated Testing](#9-automated-testing)
- [10. Empirical Results & Comparative Benchmark](#10-empirical-results--comparative-benchmark)
- [11. Project Directory Structure](#11-project-directory-structure)
- [12. Deployment Instructions](#12-deployment-instructions)
- [13. Academic Limitations & Viva Defense](#13-academic-limitations--viva-defense)
- [14. Contributors & Academic Metadata](#14-contributors--academic-metadata)

---

## 1. Project Overview

Automated Weather Stations (AWS) continuously record surface environmental observations worldwide. Physical sensor fouling (blocked rain gauges), electrical calibration drift, and mathematical averaging truncations frequently corrupt field telemetry.

**SkyGuard** provides an end-to-end unsupervised machine learning pipeline that:
1. Ingests raw multi-country weather observations and executes automated 6-point physical thermodynamic constraint audits.
2. Formulates domain meteorological features (Diurnal Temperature Range, Diurnal Midpoint Departure, log-precipitation, and station-relative temperature deviations).
3. Preprocesses data with zero data leakage (freezing training medians and standardization scalers).
4. Employs **Isolation Forest** ($n=150, \text{contamination}=0.03, \text{random\_state}=42$) to compute continuous decision-function anomaly scores and stratify records into `Severe`, `High`, and `Moderate` tiers.
5. Benchmarks Isolation Forest against **Local Outlier Factor (LOF)** and **DBSCAN** using Silhouette separation scores and runtime metrics.
6. Delivers an interactive 6-section **Streamlit Weather Intelligence Platform** that supports real-time multi-parameter investigation and live CSV upload detection with zero model retraining.

---

## 2. Problem Statement & Need

### Problem Statement
Weather stations collect continuous environmental observations. Abnormal readings, unusual variations, and data-quality problems can affect the reliability of weather information. The objective is to use **unsupervised machine learning** to identify unusual weather observations without relying on pre-labelled anomaly data.

### Academic & Practical Significance
In real-world meteorological networks, observational datasets do not come with pre-annotated sensor failure ground truth. Reporting supervised metrics (accuracy, precision, recall, F1) on unlabelled data constitutes academic fabrication. SkyGuard solves this legitimately using multi-dimensional geometric isolation and density-based unsupervised learning.

---

## 3. Key Features

- **Automated Physical Inconsistency Discovery:** Enforces thermodynamic laws, discovering **7 genuine physical anomalies** where daily mean temperature exceeded daily maximum temperature ($T_{AVG} > T_{MAX}$) or dropped below minimum temperature.
- **Meteorological Feature Engineering:**
  - *Diurnal Temperature Range ($DTR = T_{MAX} - T_{MIN}$):* Catches stuck probes (near $0^\circ\text{C}$) or radiation shield detachment.
  - *Diurnal Midpoint Departure ($\Delta T = T_{AVG} - \frac{T_{MAX} + T_{MIN}}{2}$):* Identifies daytime frontal intrusions and thermal calibration skew.
  - *Log-Precipitation ($\log(1 + PRCP)$):* Reduces rainfall right-skewness from **5.72 down to 1.73**.
  - *Station-Relative Temperature Departure ($T_{AVG} - \overline{T_{\text{city}}}$):* Distinguishes local abnormalities from broad latitudinal gradients.
- **Zero-Leakage Transformation Pipeline:** Scikit-learn compatible `WeatherPreprocessor` that freezes training medians and scalers.
- **Empirical Severity Stratification:** Ranks anomalies into `Severe`, `High`, and `Moderate` tiers using decision-function quantiles.
- **Multi-Algorithm Benchmarking:** Compares Isolation Forest against Local Outlier Factor and DBSCAN.
- **Zero-Retraining Dashboard:** Streamlit platform loads serialized `.joblib` binaries from `models/` with zero retraining on user interaction.
- **Live CSV Upload Detection:** Users can upload any new weather CSV; the system validates schema conformity, audits missing values, and scores observations in real time.
- **Automated Testing Suite:** 25 passing `pytest` test cases covering ingestion, transformation, serialization, invariants, and integration contracts.

---

## 4. System Architecture

```
+-------------------------------------------------------------+
|                 SkyGuard Pipeline Architecture              |
+-------------------------------------------------------------+
                               |
                [ 1. Data Ingestion & Auditing ]
                               |
        +----------------------+----------------------+
        | Raw CSV Ingestion    | Physical Constraints |
        | (String / Unit regex)| (TMIN <= TAVG <= TMAX|
        +----------------------+----------------------+
                               |
            [ 2. Zero-Leakage Preprocessor ]
                               |
        +----------------------+----------------------+
        | Training Median Fill | Standard / Robust    |
        | (No Leakage)         | Feature Scaling      |
        +----------------------+----------------------+
                               |
              [ 3. Domain Feature Engineering ]
                               |
        +----------------------+----------------------+
        | Diurnal Range (DTR)  | Log-Precipitation    |
        | Midpoint Deviation   | City Temp Anomaly    |
        +----------------------+----------------------+
                               |
            [ 4. Unsupervised Model Ensemble ]
                               |
        +----------------------+----------------------+
        | Isolation Forest     | Benchmarking vs      |
        | (n=150, c=0.03)      | LOF & DBSCAN         |
        +----------------------+----------------------+
                               |
            [ 5. Decision & Stratification ]
                               |
        +----------------------+----------------------+
        | Continuous Score s   | Severity Ranking     |
        | (Decision Function)  | (Severe/High/Mod)    |
        +----------------------+----------------------+
                               |
             [ 6. Streamlit Web Platform ]
                               |
        +----------------------+----------------------+
        | 6 Dashboard Sections | Live CSV Upload &    |
        | Plotly Visualizations| Zero Retraining      |
        +----------------------+----------------------+
```

---

## 5. Technology Stack

- **Core Runtime:** Python 3.12.10
- **Data Engineering:** `pandas 3.1.0`, `numpy 2.5.4`
- **Machine Learning:** `scikit-learn 1.9.1`, `scipy 1.18.1`
- **Model Serialization:** `joblib 1.6.0`
- **Interactive UI & Visualization:** `streamlit 1.66.1`, `plotly 7.1.0`
- **Static Publication Visuals:** `matplotlib 3.11.2`, `seaborn 0.13.2`
- **Automated Testing:** `pytest 9.2.0`
- **Documentation Engines:** `reportlab 5.1.0` (PDF), `python-pptx 1.0.2` (PPTX)

---

## 6. Dataset Information

- **File:** `data/weather_data_world_2025-01-01_2025-02-28.csv`
- **Total Records:** 6,077 daily observations
- **Observation Period:** January 1, 2025 – February 28, 2025 (59 continuous days)
- **Geographic Coverage:** 103 cities across 35 countries worldwide
- **Baseline Features:**
  - `TMIN`: Daily minimum temperature (°C) — 0% missing
  - `TAVG`: Daily average temperature (°C) — 0% missing
  - `TMAX`: Daily maximum temperature (°C) — 0% missing
  - `PRCP`: Daily precipitation (mm) — 7.77% missing (median imputed)
  - `WSPD`: Sustained wind speed (km/h) — 0% missing
- **Excluded Features (> 50% missing):** `SNOW` (92.5%), `TSUN` (97.1%), `WPGT` (75.7%), `PRES` (elevation confounded).

---

## 7. Installation & Setup

### Prerequisites
- Python 3.10+ (Python 3.12 recommended)
- Git

### Step-by-Step Installation

```bash
# 1. Clone the repository
git clone https://github.com/Rithwikkumar35/SkyGuard.git
cd SkyGuard

# 2. Create and activate a Python virtual environment
python3 -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate

# 3. Upgrade pip and install all frozen dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 8. Execution Instructions

### A. Launch Interactive Web Dashboard
Run the Streamlit application using either entrypoint:

```bash
streamlit run app.py
# Or: streamlit run app/app.py
```
*Access the interface in your browser at `http://localhost:8501`.*

### B. Run the Modular ML Pipeline from CLI

```bash
# 1. Run Data Ingestion & Full Exploratory Data Analysis (generates charts in results/)
PYTHONPATH=. python src/eda.py

# 2. Train and Serialize the Isolation Forest Model (saves to models/ and results/)
PYTHONPATH=. python src/train.py

# 3. Execute Multi-Paradigm Benchmark & Sensitivity Analysis (IF vs LOF vs DBSCAN)
PYTHONPATH=. python src/evaluate.py
```

### C. Recompile University Report (PDF) & Presentation (PPTX)

```bash
# Generate docs/PROJECT_REPORT.pdf
PYTHONPATH=. python docs/generate_pdf_report.py

# Generate docs/PRESENTATION.pptx
PYTHONPATH=. python docs/generate_presentation.py
```

---

## 9. Automated Testing

Run the complete 25-case automated `pytest` test suite:

```bash
PYTHONPATH=. pytest tests/ -v
```

### Test Suite Breakdown
| Test File | Test Count | Scope |
| :--- | :---: | :--- |
| `tests/test_data.py` | 6 | Data ingestion, string cleaning regex, coordinates parsing, schema validation, physical consistency. |
| `tests/test_preprocessing.py` | 6 | Feature engineering calculations, zero-leakage training imputation, standard scaling, joblib serialization. |
| `tests/test_model.py` | 7 | Model training, scoring invariants ($s \le 0$ for anomalies), reproducibility, contamination sensitivity, inference on synthetic test days. |
| `tests/test_integration.py` | 6 | Full pipeline round-trip, Streamlit backend contracts, and live CSV upload simulation. |
| **Total** | **25 Passed** | **100% Pass Rate in ~5 seconds** |

---

## 10. Empirical Results & Comparative Benchmark

### Multi-Paradigm Algorithm Benchmark
*(Saved to `results/model_comparison.csv`)*

| Algorithm | Paradigm | Flagged Anomalies | Flagged % | Silhouette Score | Runtime (ms) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Isolation Forest** | Randomized Tree Partitioning | **183** | **3.01%** | **0.5518** | **135.81 ms** |
| **Local Outlier Factor (LOF)** | K-NN Local Density Deviation | 183 | 3.01% | 0.2246 | 27.08 ms |
| **DBSCAN** | Density Clustering (Noise Identification) | 14 | 0.23% | 0.7638 | 127.14 ms |

### Contamination Sensitivity Analysis
*(Saved to `results/contamination_comparison.csv`)*

| Setting | Flagged Observations | Flagged % | Mean Score | Std Dev Score | Runtime (ms) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **0.01 (1%)** | 61 | 1.00% | 0.2095 | 0.0527 | 123.6 ms |
| **0.03 (3%)** | 183 | 3.01% | 0.1521 | 0.0560 | 134.1 ms |
| **0.05 (5%)** | 304 | 5.00% | 0.1193 | 0.0569 | 126.8 ms |
| **0.10 (10%)** | 608 | 10.00% | 0.0736 | 0.0567 | 128.4 ms |

### Top Qualitative Anomalies Discovered
- **Polar Vortex Extreme Cold:** Calgary (CA) plunging to **-29.0 °C** (Score: `-0.1287`, Severe).
- **Torrential Deluges:** Montreal (CA) recording **27.5 mm** liquid precipitation (Score: `-0.1109`, Severe); Sydney recording **111.5 mm**.
- **Winter Gale Storms:** Toronto (CA) **35.6 km/h** winds; Reykjavik (IS) **57.0 km/h** sustained gale.
- **Physical Sensor Violations:** Melbourne (AU) flagged for daily mean temperature exceeding recorded maximum ($T_{AVG} = 19.4^\circ\text{C} > T_{MAX} = 18.4^\circ\text{C}$).

---

## 11. Project Directory Structure

```
SkyGuard/
├── .gitignore                      # Git exclusion rules
├── PROJECT_LINKS.txt               # University submission metadata & URLs
├── README.md                       # Comprehensive project documentation
├── requirements.txt                # Frozen Python environment dependencies
├── app.py                          # Root entrypoint for Streamlit
│
├── app/                            # Interactive Web Dashboard
│   ├── __init__.py
│   └── app.py                      # 6-section Streamlit intelligence platform
│
├── data/                           # Meteorological Datasets
│   ├── weather_data_world_2025-01-01_2025-02-28.csv  # Raw dataset (6,077 rows)
│   └── weather_cleaned.csv         # Cleaned baseline dataset
│
├── docs/                           # University Documentation & Artifacts
│   ├── PROJECT_REPORT.pdf          # Publication-grade academic report PDF (921 KB)
│   ├── PRESENTATION.pptx           # 16:9 widescreen presentation slides (47 KB)
│   ├── report_content.md           # 10-section academic report manuscript
│   ├── ppt_structure.md            # Slide breakdown & viva defense guide
│   ├── generate_pdf_report.py      # ReportLab compilation script
│   └── generate_presentation.py    # Python-pptx generation script
│
├── models/                         # Serialized Model Artifacts
│   ├── isolation_forest.joblib     # Pre-trained Isolation Forest model
│   ├── preprocessor.joblib         # Fitted WeatherPreprocessor pipeline
│   ├── scaler.joblib               # Fitted StandardScaler
│   └── model_metadata.json         # Audit metadata and score quantiles
│
├── notebooks/                      # Academic Analysis Notebooks
│   └── SkyGuard_Analysis.ipynb     # Interactive Jupyter notebook
│
├── results/                        # Generated Visualizations & Evaluation Data
│   ├── anomaly_results.csv
│   ├── anomaly_results_with_locations.csv
│   ├── contamination_comparison.csv
│   ├── model_comparison.csv
│   ├── qualitative_anomaly_analysis.csv
│   ├── descriptive_statistics.csv
│   ├── data_quality_report.json
│   ├── feature_distributions.png
│   ├── correlation_matrix.png
│   ├── temperature_analysis.png
│   ├── rainfall_distribution.png
│   ├── wind_speed_analysis.png
│   ├── city_weather_comparisons.png
│   ├── outliers_boxplots.png
│   ├── missing_values_analysis.png
│   ├── model_comparison.png
│   ├── contamination_sensitivity.png
│   └── anomaly_score_distribution.png
│
├── src/                            # Modular Python ML Package
│   ├── __init__.py
│   ├── data_loader.py              # Ingestion, regex parsing & physical audit
│   ├── feature_engineering.py      # DTR, midpoint deviation, log-PRCP, city departures
│   ├── preprocessing.py            # Zero-leakage WeatherPreprocessor pipeline
│   ├── train.py                    # Isolation Forest training & serialization
│   ├── evaluate.py                 # Multi-paradigm benchmark & sensitivity
│   └── utils.py                    # Path resolution & academic plotting styles
│
└── tests/                          # Automated Pytest Suite (25 Tests)
    ├── __init__.py
    ├── test_data.py                # Ingestion & schema validation tests
    ├── test_preprocessing.py       # Preprocessing & leakage tests
    ├── test_model.py               # Model training, invariants & reproducibility tests
    └── test_integration.py         # End-to-end pipeline & live upload simulation tests
```

---

## 12. Deployment Instructions

### Deploying to Streamlit Community Cloud (Free)
1. Push this repository to GitHub (see instructions below).
2. Go to [share.streamlit.io](https://share.streamlit.io/) and log in with GitHub.
3. Click **"New app"**.
4. Select your repository: `username/SkyGuard`.
5. Set Main file path: `app.py` (or `app/app.py`).
6. Click **"Deploy!"**. Streamlit Cloud will install dependencies from `requirements.txt` and launch the app with a public URL.

---

## 13. Academic Limitations & Viva Defense

### Viva Defense Talking Points
1. **Why are accuracy, precision, recall, and F1-score not reported?**  
   *Because real-world meteorological networks collect observational data without pre-existing sensor failure labels. In unsupervised learning, evaluating against fabricated labels is scientifically ungrounded. We evaluate via Silhouette separation score (0.5518), score distributions, sensitivity curves, and qualitative meteorological verification.*
2. **What does an Isolation Forest anomaly flag signify?**  
   *It indicates structural multi-dimensional statistical atypicality. It captures genuine severe storms (polar vortex, gale winds) as well as physical sensor glitches (such as the 7 records where $T_{AVG} > T_{MAX}$).*
3. **How was data leakage avoided?**  
   *Missing value imputation statistics (medians) and scaling parameters were fitted strictly on training observations and frozen. Transformation of test or live uploaded data reuses those frozen parameters without recomputing statistics.*

---

## 14. Contributors & Academic Metadata

- **Course:** INT396 — Unsupervised Learning
- **Problem Statement ID:** `[PS-INT396-WEATHER-01]`
- **Team Members:**
  - `[STUDENT_NAME_1]` (`[REGISTRATION_NUMBER_1]`) — Lead Machine Learning Engineer
  - `[STUDENT_NAME_2]` (`[REGISTRATION_NUMBER_2]`) — Data & Feature Engineer
  - `[STUDENT_NAME_3]` (`[REGISTRATION_NUMBER_3]`) — UI/UX & System Integration Lead
- **Faculty Mentor:** `[FACULTY_MENTOR_NAME]`
- **Institution:** `[DEPARTMENT_OF_COMPUTER_SCIENCE / UNIVERSITY_NAME]`
