# SkyGuard — Presentation Structure & Viva Defense Guide

**Subject:** INT396 — Unsupervised Learning  
**Project:** SkyGuard — Weather-Station Sensor Anomaly Detection  

---

## Slide 1: Title Slide
- **Title:** SkyGuard: Weather-Station Sensor Anomaly Detection
- **Subtitle:** An Unsupervised Machine Learning Framework for Quality Assurance in Meteorological Telemetry
- **Course:** INT396 — Unsupervised Learning
- **Presenters (Editable Placeholders):**
  - `[STUDENT_NAME_1]` (`[REG_NO_1]`) — Lead ML Engineer
  - `[STUDENT_NAME_2]` (`[REG_NO_2]`) — Data & Feature Engineer
  - `[STUDENT_NAME_3]` (`[REG_NO_3]`) — UI/UX & Integration Lead
- **Faculty Supervisor:** `[FACULTY_MENTOR_NAME]`
- **Department & Institution:** `[DEPARTMENT_OF_COMPUTER_SCIENCE / UNIVERSITY_NAME]`

---

## Slide 2: Problem Statement & Academic Need
- **The Telemetry Challenge:** Automated weather stations collect critical continuous telemetry ($T_{MIN}, T_{AVG}, T_{MAX}, PRCP, WSPD$). Telemetry is vulnerable to sensor degradation, physical fouling (blocked gauges), and calculation/averaging artifacts.
- **The Unsupervised Reality:** Real-world weather stations operate without pre-labelled sensor-failure annotations. Supervised learning cannot be applied without fabricating artificial ground truth.
- **Core Objective:** Design an end-to-end unsupervised pipeline that isolates multivariate anomalous observations purely from intrinsic geometrical distributions, while ensuring zero data leakage.

---

## Slide 3: End-to-End Pipeline Architecture
- **Multi-Stage Processing Flow:**
  1. *Ingestion & Cleaning:* Regex unit parsing (`°C`, `mm`, `km/h`), coordinate extraction.
  2. *Physical Consistency Audit:* Automated enforcement of 6 physical thermodynamic laws.
  3. *Zero-Leakage Preprocessing:* Scikit-learn transformer with frozen training medians and standard scaling.
  4. *Domain Feature Engineering:* Diurnal range ($DTR$), midpoint deviation ($\Delta T$), $\log(1 + PRCP)$, and city-relative anomaly.
  5. *Unsupervised Modeling:* Isolation Forest ($n=150, c=0.03, \text{seed}=42$) and empirical severity ranking (`Severe`, `High`, `Moderate`).
  6. *Deployment:* High-performance Streamlit dashboard loading pre-trained artifacts from disk.

---

## Slide 4: Dataset Quality & Physical Consistency Audit
- **Telemetry Corpus:** 6,077 daily observation records from 103 cities across 35 countries (Jan 1 – Feb 28, 2025).
- **Severe Missingness Exclusions:** Excluded `SNOW` (92.5% missing), `TSUN` (97.1% missing), and `WPGT` (75.7% missing).
- **Groundbreaking Physical Audit Finding:**
  - $T_{MIN} \le T_{MAX}$: 0 violations across all 6,077 records.
  - **Identified exactly 7 physical anomalies** where $T_{AVG}$ fell outside $[T_{MIN}, T_{MAX}]$ (e.g. Melbourne row 5610: $T_{MIN}=16.4^\circ\text{C}, T_{MAX}=18.4^\circ\text{C}, T_{AVG}=19.4^\circ\text{C}$!).

---

## Slide 5: Domain-Specific Meteorological Feature Engineering
- **1. Diurnal Temperature Range ($DTR = T_{MAX} - T_{MIN}$):** Identifies stuck sensors (range near $0^\circ\text{C}$) or extreme radiative swings.
- **2. Diurnal Midpoint Deviation ($\Delta T = T_{AVG} - \frac{T_{MAX} + T_{MIN}}{2}$):** Measures thermal progression asymmetry caused by daytime frontal passages or sensor skew.
- **3. Log-Transformed Precipitation ($\log(1 + PRCP)$):** Dampens the extreme right-skewness of rainfall, reducing distribution skewness from **5.72 to 1.73**.
- **4. City-Relative Temperature Departure ($T_{AVG} - \overline{T_{\text{city}}}$):** Normalizes local climate baselines so high-latitude stations are not flagged merely for being cold in winter.

---

## Slide 6: Machine Learning Methodology — Isolation Forest
- **Algorithmic Principle:** Isolates anomalies rather than profiling normal data. Because anomalies have extreme values, they require **fewer random splits** to become isolated into terminal tree nodes.
- **Mathematical Anomaly Score:**
  $$s(x, n) = 2^{-\frac{E(h(x))}{c(n)}}$$
  - $h(x)$: Path length in isolation trees.
  - $c(n)$: Average unsuccessful search depth in Binary Search Trees.
  - Score $s \to 1.0$: Rapid isolation $\Rightarrow$ Anomaly.
- **Operational Configuration:** 150 isolation estimators, $0.03$ contamination factor, random seed $42$.

---

## Slide 7: Unsupervised Model Evaluation & Sensitivity Analysis
- **Contamination Sensitivity (1% to 10%):**
  - Contamination 0.01: 61 flagged records (1.00%), mean score 0.2095
  - Contamination 0.03: 183 flagged records (3.01%), mean score 0.1521
  - Contamination 0.05: 304 flagged records (5.00%), mean score 0.1193
  - Contamination 0.10: 608 flagged records (10.00%), mean score 0.0736
- **Stability Finding:** The model's anomaly ordering remains consistent across contamination variations, confirming stability.

---

## Slide 8: Multi-Paradigm Algorithm Benchmark
- **Fair Comparison on Standardized Weather Telemetry:**
  - **Isolation Forest:** Flagged 183 (3.01%) | **Silhouette Score: 0.5518** | Latency: 135.8 ms
  - **Local Outlier Factor (LOF):** Flagged 183 (3.01%) | Silhouette Score: 0.2246 | Latency: 27.1 ms
  - **DBSCAN:** Flagged 14 (0.23%) | Silhouette Score: 0.7638 | Latency: 127.1 ms
- **Technical Conclusion:** Isolation Forest achieves the optimal balance of geometric cluster separation (0.5518) and sensitivity to multivariate outliers without being constrained by density variation.

---

## 9. Slide 9: Professional Streamlit Intelligence Platform
- **Production-Grade Application (`app/app.py`):**
  - Section 1: KPI Metrics & Global Geospatial Station Map.
  - Section 2: Interactive Plotly Visualizations (diurnal trends, correlation matrix, city comparisons).
  - Section 3: Model Evaluation & Benchmarks.
  - Section 4: Interactive Anomaly Filter Table with Severity Badges & CSV downloads.
  - Section 5: Transparent System Architecture & Academic Methodology.
  - Section 6: Live CSV Upload & Anomaly Scoring with pre-trained model (Zero Retraining).

---

## Slide 10: Academic Viva Defense, Limitations & Future Scope
- **Key Viva Defense Points:**
  - *Why no accuracy or F1-score?* Because reporting supervised metrics on unlabelled real-world telemetry is unscientific.
  - *What do flagged anomalies represent?* Severe polar cold waves (Calgary $-29.0^\circ\text{C}$), storm deluges (Montreal $27.5\text{ mm}$), gale winds (Toronto $35.6\text{ km/h}$), and physical inconsistencies.
- **Future Scope:**
  - Spatio-temporal Graph Neural Networks (GNNs) across adjacent station networks.
  - Edge deployment using TinyML on microcontrollers (ESP32).
