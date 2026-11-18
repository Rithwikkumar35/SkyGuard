"""
app/app.py
SkyGuard — Weather-Station Sensor Anomaly Detection Dashboard.
A modular, interactive, university-grade Streamlit application for INT396 Unsupervised Learning.
"""

import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import json
from typing import Dict, Tuple, Any
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_loader import (
    load_raw_dataset,
    validate_weather_schema,
    audit_physical_consistency,
    generate_data_quality_report,
    convert_measurement_string,
    parse_coordinates,
)
from src.preprocessing import WeatherPreprocessor
from src.train import load_trained_model
from src.utils import get_data_dir, get_models_dir, get_results_dir

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="SkyGuard | Weather Intelligence & Sensor Anomaly Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern weather intelligence styling
st.markdown(
    """
    <style>
        /* General app styling */
        .main {
            background-color: #f8fafc;
        }
        /* Top header card */
        .skyguard-header {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            padding: 24px 32px;
            border-radius: 12px;
            color: #ffffff;
            margin-bottom: 24px;
            border-left: 6px solid #38bdf8;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        .skyguard-title {
            font-size: 28px;
            font-weight: 700;
            color: #ffffff;
            margin: 0;
            letter-spacing: -0.5px;
        }
        .skyguard-subtitle {
            font-size: 14px;
            color: #94a3b8;
            margin-top: 6px;
            margin-bottom: 0;
        }
        /* Metric card styling */
        .metric-card {
            background: #ffffff;
            padding: 16px 20px;
            border-radius: 10px;
            border: 1px solid #e2e8f0;
            border-top: 4px solid #0284c7;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            text-align: center;
        }
        .metric-value {
            font-size: 26px;
            font-weight: 700;
            color: #0f172a;
        }
        .metric-label {
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
            color: #64748b;
            letter-spacing: 0.5px;
            margin-top: 4px;
        }
        /* Badges */
        .badge-severe {
            background-color: #fee2e2;
            color: #991b1b;
            padding: 4px 10px;
            border-radius: 6px;
            font-weight: 600;
            font-size: 12px;
        }
        .badge-high {
            background-color: #ffedd5;
            color: #9a3412;
            padding: 4px 10px;
            border-radius: 6px;
            font-weight: 600;
            font-size: 12px;
        }
        .badge-moderate {
            background-color: #fef9c3;
            color: #854d0e;
            padding: 4px 10px;
            border-radius: 6px;
            font-weight: 600;
            font-size: 12px;
        }
        .badge-normal {
            background-color: #dcfce7;
            color: #166534;
            padding: 4px 10px;
            border-radius: 6px;
            font-weight: 600;
            font-size: 12px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# CACHED DATA & MODEL LOADERS
# -----------------------------------------------------------------------------
@st.cache_data
def load_application_data() -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Load cached dataset, audited consistency records, and model results."""
    raw_df = load_raw_dataset(parse_measurements=True)
    audited_df, physical_summary = audit_physical_consistency(raw_df)

    results_path = get_results_dir() / "anomaly_results_with_locations.csv"
    if results_path.exists():
        results_df = pd.read_csv(results_path)
    else:
        # Fallback if results CSV not yet generated
        from src.train import train_isolation_forest
        _, _, results_df, _ = train_isolation_forest(raw_df, save_artifacts=True)

    return audited_df, results_df, physical_summary


@st.cache_resource
def load_models() -> Tuple[Any, WeatherPreprocessor, Dict[str, Any]]:
    """Load pre-trained model and preprocessor without refitting."""
    model, metadata = load_trained_model()
    preprocessor = WeatherPreprocessor.load()
    return model, preprocessor, metadata


# Load cached resources
try:
    data_df, results_df, physical_audit = load_application_data()
    model, preprocessor, model_metadata = load_models()
except Exception as e:
    st.error(f"Initialization Error: {e}. Please ensure src/train.py has been executed.")
    st.stop()

# -----------------------------------------------------------------------------
# SIDEBAR NAVIGATION & CONTROLS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://raw.githubusercontent.com/FortAwesome/Font-Awesome/master/svgs/solid/shield-halved.svg", width=48)
    st.markdown("## **SkyGuard**")
    st.caption("Weather-Station Sensor Anomaly Detection")
    st.caption("Subject: **INT396 — Unsupervised Learning**")
    st.divider()

    selected_section = st.radio(
        "Navigation",
        [
            "1. Overview & Summary",
            "2. Exploratory Weather Analysis",
            "3. Model Evaluation & Benchmarks",
            "4. Anomaly Detection & Records",
            "5. Methodology & Architecture",
            "6. Data Management & Upload",
        ],
        index=0,
    )

    st.divider()
    st.markdown("### **Model Status**")
    st.success("✅ Pre-trained Model Loaded")
    st.write(f"- **Algorithm:** {model_metadata.get('algorithm', 'Isolation Forest')}")
    st.write(f"- **Trees:** {model_metadata.get('n_estimators', 150)}")
    st.write(f"- **Contamination:** {model_metadata.get('contamination_setting', 0.03):.0%}")
    st.write(f"- **Features:** {len(model_metadata.get('features_used', []))}")
    st.caption("Model is loaded directly from `models/` (Zero retraining on refresh).")

# Header banner
st.markdown(
    """
    <div class="skyguard-header">
        <h1 class="skyguard-title">🛡️ SkyGuard — Weather Sensor Anomaly Detection</h1>
        <p class="skyguard-subtitle">
            Unsupervised Machine Learning for Sensor Fault & Extreme Weather Discovery • INT396 University Project
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# SECTION 1: OVERVIEW & EXECUTIVE SUMMARY
# -----------------------------------------------------------------------------
if selected_section == "1. Overview & Summary":
    st.header("1. Overview & Executive Summary")
    st.write(
        "SkyGuard continuously evaluates historical weather station observations across the globe. "
        "Using unsupervised tree isolation, the system identifies statistically irregular sensor recordings "
        "and extreme climatological events without requiring pre-labeled ground truth."
    )

    total_obs = len(results_df)
    flagged_count = int(results_df["anomaly"].sum())
    normal_count = total_obs - flagged_count
    anomaly_rate = flagged_count / total_obs if total_obs else 0.0
    unique_cities = results_df["City"].nunique() if "City" in results_df.columns else 103
    unique_countries = results_df["Country"].nunique() if "Country" in results_df.columns else 35

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-value">{total_obs:,}</div>
                <div class="metric-label">Total Observations</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-value">{unique_cities}</div>
                <div class="metric-label">Weather Stations</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""<div class="metric-card" style="border-top-color: #ef4444;">
                <div class="metric-value" style="color: #dc2626;">{flagged_count:,}</div>
                <div class="metric-label">Flagged Anomalies</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f"""<div class="metric-card" style="border-top-color: #10b981;">
                <div class="metric-value" style="color: #059669;">{normal_count:,}</div>
                <div class="metric-label">Normal Observations</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with col5:
        st.markdown(
            f"""<div class="metric-card" style="border-top-color: #f59e0b;">
                <div class="metric-value" style="color: #d97706;">{anomaly_rate:.2%}</div>
                <div class="metric-label">Anomaly Proportion</div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Geospatial Station Map
    st.subheader("Global Weather Station Network & Anomaly Density")
    if "latitude" in results_df.columns and "longitude" in results_df.columns:
        city_summary = results_df.groupby(["City", "Country", "latitude", "longitude"]).agg(
            total_records=("anomaly_score", "count"),
            flagged_anomalies=("anomaly", "sum"),
            mean_temp=("TAVG", "mean"),
        ).reset_index()

        city_summary["anomaly_rate_pct"] = (city_summary["flagged_anomalies"] / city_summary["total_records"] * 100).round(1)

        fig_map = px.scatter_geo(
            city_summary,
            lat="latitude",
            lon="longitude",
            color="flagged_anomalies",
            size="flagged_anomalies",
            size_max=18,
            hover_name="City",
            hover_data={
                "Country": True,
                "mean_temp": ":.1f °C",
                "flagged_anomalies": True,
                "total_records": True,
                "anomaly_rate_pct": True,
                "latitude": False,
                "longitude": False,
            },
            color_continuous_scale="Reds",
            title="Global Weather Stations: Anomaly Frequency Distribution",
            projection="natural earth",
        )
        fig_map.update_layout(
            margin=dict(l=0, r=0, t=40, b=0),
            height=480,
            geo=dict(showcoastlines=True, showland=True, landcolor="#f1f5f9", oceancolor="#e0f2fe", showocean=True),
        )
        st.plotly_chart(fig_map, use_container_width=True)

    # Climatological Extremes Summary Cards
    st.subheader("Observed Climatological Extremes (Jan–Feb 2025)")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.info(f"❄️ **Coldest Record:**\n**-29.0 °C** (Calgary, CA)\n*Polar Vortex event*")
    with c2:
        st.warning(f"🔥 **Hottest Record:**\n**43.6 °C** (Perth, AU)\n*Summer heatwave*")
    with c3:
        st.success(f"🌧️ **Max 24h Rainfall:**\n**111.5 mm** (Sydney, AU)\n*Severe deluge*")
    with c4:
        st.error(f"💨 **Max Wind Speed:**\n**57.0 km/h** (Reykjavik, IS)\n*Winter gale*")

    st.caption("Data source: Global Meteorological Observations (Jan 01, 2025 – Feb 28, 2025).")

# -----------------------------------------------------------------------------
# SECTION 2: EXPLORATORY WEATHER ANALYSIS
# -----------------------------------------------------------------------------
elif selected_section == "2. Exploratory Weather Analysis":
    st.header("2. Exploratory Weather Analysis (EDA)")
    st.write(
        "Interactive exploratory data analysis reveals feature distributions, correlation structures, "
        "and physical anomalies across 103 global stations."
    )

    eda_tabs = st.tabs([
        "🌡️ Temperature Analysis",
        "🌧️ Precipitation & Wind",
        "🔗 Correlation Structure",
        "🏙️ City-Wise Comparisons",
        "⚠️ Physical Inconsistencies",
    ])

    with eda_tabs[0]:
        st.subheader("Daily Temperature Trends by Station")
        city_list = sorted(data_df["City"].dropna().unique().tolist())
        selected_city = st.selectbox("Select Station / City:", city_list, index=city_list.index("Barcelona") if "Barcelona" in city_list else 0)

        city_data = data_df[data_df["City"] == selected_city].sort_values("DATE")

        fig_temp = go.Figure()
        fig_temp.add_trace(go.Scatter(
            x=city_data["DATE"], y=city_data["TMAX"],
            name="TMAX (Max Temp)", line=dict(color="#ef4444", width=2)
        ))
        fig_temp.add_trace(go.Scatter(
            x=city_data["DATE"], y=city_data["TAVG"],
            name="TAVG (Daily Mean)", line=dict(color="#3b82f6", width=2.5)
        ))
        fig_temp.add_trace(go.Scatter(
            x=city_data["DATE"], y=city_data["TMIN"],
            name="TMIN (Min Temp)", line=dict(color="#06b6d4", width=2),
            fill="tonexty", fillcolor="rgba(59, 130, 246, 0.1)"
        ))
        fig_temp.update_layout(
            title=f"Daily Temperature Progression: {selected_city} (Jan-Feb 2025)",
            xaxis_title="Observation Date",
            yaxis_title="Temperature (°C)",
            hovermode="x unified",
            height=420,
        )
        st.plotly_chart(fig_temp, use_container_width=True)

        # Diurnal Range Distribution
        if "TMAX" in data_df.columns and "TMIN" in data_df.columns:
            data_df["DTR"] = data_df["TMAX"] - data_df["TMIN"]
            fig_dtr = px.histogram(
                data_df, x="DTR", nbins=40, color_discrete_sequence=["#f59e0b"],
                title="Diurnal Temperature Range (DTR = TMAX - TMIN) Distribution Across All Stations",
                labels={"DTR": "Diurnal Temperature Range (°C)"}
            )
            fig_dtr.update_layout(height=380)
            st.plotly_chart(fig_dtr, use_container_width=True)

    with eda_tabs[1]:
        st.subheader("Precipitation & Wind Dynamics")
        col_rain1, col_rain2 = st.columns(2)
        with col_rain1:
            wet_days = data_df[data_df["PRCP"] > 0]["PRCP"].dropna()
            fig_rain = px.histogram(
                wet_days, x=wet_days, nbins=35,
                color_discrete_sequence=["#10b981"],
                title="Precipitation Distribution on Wet Days (PRCP > 0 mm)",
                labels={"x": "Daily Rainfall (mm)"}
            )
            fig_rain.update_layout(height=380)
            st.plotly_chart(fig_rain, use_container_width=True)

        with col_rain2:
            fig_wind = px.histogram(
                data_df, x="WSPD", nbins=35,
                color_discrete_sequence=["#8b5cf6"],
                title="Sustained Wind Speed Distribution (km/h)",
                labels={"WSPD": "Wind Speed (km/h)"}
            )
            fig_wind.add_vline(x=data_df["WSPD"].quantile(0.99), line_dash="dash", line_color="red",
                               annotation_text="99th Percentile")
            fig_wind.update_layout(height=380)
            st.plotly_chart(fig_wind, use_container_width=True)

    with eda_tabs[2]:
        st.subheader("Weather Feature Correlation Heatmap")
        corr_cols = [c for c in ["TMIN", "TAVG", "TMAX", "PRCP", "WSPD"] if c in data_df.columns]
        corr_matrix = data_df[corr_cols].corr().round(2)

        fig_corr = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="Pearson Correlation Matrix (Baseline Weather Features)",
        )
        fig_corr.update_layout(height=420)
        st.plotly_chart(fig_corr, use_container_width=True)
        st.caption("Note: Temperature variables exhibit high mutual collinearity; precipitation and wind speed act as quasi-orthogonal dimensions.")

    with eda_tabs[3]:
        st.subheader("City-Wise Weather Comparison")
        city_stats = data_df.groupby("City").agg({
            "TAVG": "mean",
            "PRCP": "sum",
            "WSPD": "mean",
        }).reset_index()

        c_comp1, c_comp2 = st.columns(2)
        with c_comp1:
            cold_warm = pd.concat([
                city_stats.nsmallest(5, "TAVG"),
                city_stats.nlargest(5, "TAVG")
            ]).sort_values("TAVG")
            fig_cw = px.bar(
                cold_warm, x="TAVG", y="City", orientation="h",
                color="TAVG", color_continuous_scale="icefire",
                title="Top 5 Coldest and Top 5 Warmest Stations (Mean TAVG °C)",
            )
            fig_cw.update_layout(height=400)
            st.plotly_chart(fig_cw, use_container_width=True)

        with c_comp2:
            wet_top = city_stats.nlargest(10, "PRCP").sort_values("PRCP")
            fig_wt = px.bar(
                wet_top, x="PRCP", y="City", orientation="h",
                color_discrete_sequence=["#0284c7"],
                title="Top 10 Wettest Cities (Cumulative Jan-Feb Rain mm)",
            )
            fig_wt.update_layout(height=400)
            st.plotly_chart(fig_wt, use_container_width=True)

    with eda_tabs[4]:
        st.subheader("Identified Physical & Instrumental Inconsistencies")
        inconsistent_df = data_df[data_df["is_physically_inconsistent"]]
        st.write(
            f"During the Phase 1 & 2 automated physical audit, exactly **{len(inconsistent_df)} observations** "
            "were detected where $T_{AVG}$ violated physical bounds ($T_{AVG} < T_{MIN}$ or $T_{AVG} > T_{MAX}$)."
        )
        st.dataframe(
            inconsistent_df[["City", "Country", "DATE", "TMIN", "TAVG", "TMAX", "inconsistency_reason"]],
            use_container_width=True,
        )
        st.info("These cases represent sensor calibration drift, hourly averaging discrepancy, or boundary-window truncation.")

# -----------------------------------------------------------------------------
# SECTION 3: MODEL EVALUATION & BENCHMARKS
# -----------------------------------------------------------------------------
elif selected_section == "3. Model Evaluation & Benchmarks":
    st.header("3. Model Evaluation & Comparative Benchmarks")
    st.write(
        "Comprehensive unsupervised evaluation comparing Isolation Forest against Local Outlier Factor (LOF) "
        "and DBSCAN, including contamination sensitivity and anomaly score distributions."
    )

    eval_tabs = st.tabs([
        "🌳 Isolation Forest Working",
        "📈 Score Distribution",
        "🎛️ Contamination Sensitivity",
        "⚖️ Model Comparison Benchmark",
    ])

    with eval_tabs[0]:
        st.subheader("How Isolation Forest Operates on Sensor Data")
        st.markdown(
            """
            **Theoretical Principle:**
            Isolation Forest isolates anomalies instead of profiling normal points. It constructs an ensemble of 
            randomized decision trees ($iTrees$). Because anomalies have extreme or unusual feature values, 
            they require **fewer random splits** to become isolated into terminal leaf nodes.

            **Mathematical Anomaly Score Formulation:**
            $$s(x, n) = 2^{-\\frac{E(h(x))}{c(n)}}$$
            - $h(x)$ is the path length (number of edges $x$ traverses from the root to the leaf).
            - $E(h(x))$ is the average path length across an ensemble of 150 isolation trees.
            - $c(n) = 2\\ln(n - 1) + 0.5772156649 - \\frac{2(n - 1)}{n}$ is the average path length of an unsuccessful search in a Binary Search Tree.

            **Interpretation:**
            - When $E(h(x)) \\to 0$, $s \\to 1$: The record isolates almost immediately $\\Rightarrow$ **Definite Anomaly**.
            - When $E(h(x)) \\to c(n)$, $s \\to 0.5$: Average path length $\\Rightarrow$ **Normal Record**.
            """
        )

    with eval_tabs[1]:
        st.subheader("Decision Function Anomaly Score Distribution")
        fig_score = px.histogram(
            results_df,
            x="anomaly_score",
            color="anomaly_label",
            nbins=50,
            color_discrete_map={"Normal": "#3b82f6", "Flagged": "#ef4444"},
            marginal="box",
            title="Distribution of Isolation Forest Decision Function Scores",
            labels={"anomaly_score": "Anomaly Decision Score", "anomaly_label": "Classification"},
        )
        fig_score.add_vline(x=0.0, line_dash="dash", line_color="black", annotation_text="Decision Boundary (Score = 0.0)")
        fig_score.update_layout(height=420)
        st.plotly_chart(fig_score, use_container_width=True)
        st.caption("Negative scores denote flagged anomalies; lower scores correspond to more severe abnormalities.")

    with eval_tabs[2]:
        st.subheader("Contamination Parameter Sensitivity Analysis")
        comp_csv_path = get_results_dir() / "contamination_comparison.csv"
        if comp_csv_path.exists():
            comp_df = pd.read_csv(comp_csv_path)
            st.dataframe(comp_df, use_container_width=True)

            fig_cont = go.Figure()
            fig_cont.add_trace(go.Bar(
                x=[f"{c:.0%}" for c in comp_df["contamination_setting"]],
                y=comp_df["flagged_records"],
                name="Flagged Records",
                marker_color="#0284c7",
            ))
            fig_cont.update_layout(
                title="Number of Flagged Observations vs Contamination Parameter",
                xaxis_title="Contamination Setting",
                yaxis_title="Flagged Observations Count",
                height=380,
            )
            st.plotly_chart(fig_cont, use_container_width=True)

    with eval_tabs[3]:
        st.subheader("Fair Benchmark: Isolation Forest vs LOF vs DBSCAN")
        model_comp_path = get_results_dir() / "model_comparison.csv"
        if model_comp_path.exists():
            model_comp = pd.read_csv(model_comp_path)
            st.dataframe(model_comp, use_container_width=True)

            col_m1, col_m2 = st.columns(2)
            with col_m1:
                fig_sil = px.bar(
                    model_comp, x="algorithm", y="silhouette_score",
                    color="algorithm",
                    title="Silhouette Score: Cluster Separation Quality",
                    labels={"silhouette_score": "Silhouette Score", "algorithm": "Algorithm"}
                )
                fig_sil.update_layout(height=360, showlegend=False)
                st.plotly_chart(fig_sil, use_container_width=True)

            with col_m2:
                fig_rt = px.bar(
                    model_comp, x="algorithm", y="runtime_ms",
                    color="algorithm",
                    title="Computational Latency / Execution Time (ms)",
                    labels={"runtime_ms": "Runtime (ms)", "algorithm": "Algorithm"}
                )
                fig_rt.update_layout(height=360, showlegend=False)
                st.plotly_chart(fig_rt, use_container_width=True)

# -----------------------------------------------------------------------------
# SECTION 4: ANOMALY DETECTION & INVESTIGATION
# -----------------------------------------------------------------------------
elif selected_section == "4. Anomaly Detection & Records":
    st.header("4. Flagged Anomaly Investigation Table")
    st.write(
        "Filter, inspect, and analyze sensor anomalies flagged by the Isolation Forest model. "
        "Filter by station, date range, severity tier, or meteorological parameters."
    )

    # Multi-parameter interactive filters
    with st.expander("🔍 Filter Controls", expanded=True):
        fcol1, fcol2, fcol3 = st.columns(3)
        with fcol1:
            all_cities = ["All Cities"] + sorted(results_df["City"].dropna().unique().tolist())
            filter_city = st.selectbox("Filter City:", all_cities)
        with fcol2:
            severity_options = ["All Severities", "Severe", "High", "Moderate"]
            filter_severity = st.selectbox("Severity Tier:", severity_options)
        with fcol3:
            show_only_anomalies = st.checkbox("Show Only Flagged Anomalies", value=True)

    # Apply filters
    filtered_df = results_df.copy()
    if filter_city != "All Cities":
        filtered_df = filtered_df[filtered_df["City"] == filter_city]
    if filter_severity != "All Severities" and "severity" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["severity"] == filter_severity]
    if show_only_anomalies:
        filtered_df = filtered_df[filtered_df["anomaly"]]

    st.subheader(f"Matching Records: {len(filtered_df):,} observations")

    display_cols = [
        c for c in ["City", "Country", "DATE", "TMIN", "TAVG", "TMAX", "PRCP", "WSPD", "anomaly_score", "severity"]
        if c in filtered_df.columns
    ]
    st.dataframe(filtered_df[display_cols].sort_values("anomaly_score"), use_container_width=True)

    # Downloads
    dcol1, dcol2 = st.columns(2)
    with dcol1:
        st.download_button(
            "📥 Download Filtered Results (CSV)",
            data=filtered_df.to_csv(index=False).encode("utf-8"),
            file_name="skyguard_filtered_anomalies.csv",
            mime="text/csv",
        )
    with dcol2:
        st.download_button(
            "📥 Download All Flagged Anomalies (183 Records)",
            data=results_df[results_df["anomaly"]].to_csv(index=False).encode("utf-8"),
            file_name="skyguard_all_flagged_observations.csv",
            mime="text/csv",
        )

    # Anomaly Driver Chart
    qual_path = get_results_dir() / "qualitative_anomaly_analysis.csv"
    if qual_path.exists():
        qual_data = pd.read_csv(qual_path)
        st.subheader("Dominant Anomaly Drivers in Top Observations")
        fig_driver = px.bar(
            qual_data.head(15),
            x="anomaly_score",
            y="City",
            color="severity",
            hover_data=["DATE", "anomaly_driver", "TMIN", "PRCP", "WSPD"],
            title="Top 15 Most Unusual Weather Records & Dominant Driver",
            orientation="h",
            color_discrete_map={"Severe": "#ef4444", "High": "#f97316", "Moderate": "#eab308"}
        )
        fig_driver.update_layout(height=450)
        st.plotly_chart(fig_driver, use_container_width=True)

# -----------------------------------------------------------------------------
# SECTION 5: METHODOLOGY & ARCHITECTURE
# -----------------------------------------------------------------------------
elif selected_section == "5. Methodology & Architecture":
    st.header("5. Methodology, System Architecture & Academic Rigor")

    st.subheader("System Architecture & Processing Pipeline")
    st.markdown(
        """
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
        ```
        """
    )

    st.subheader("Academic Limitations & Defensive Statement")
    st.warning(
        """
        **Critical Academic Clarification for INT396:**
        - **Absence of Ground Truth:** Historical meteorological stations collect observations without verified sensor-fault labels.
        - **Metric Integrity:** We do **not** report supervised accuracy, precision, recall, or F1-score because doing so would fabricate ground-truth annotations.
        - **Interpretation:** Flagged records represent statistically rare multi-dimensional outliers. An unusual reading may indicate:
          1. Genuine severe weather events (e.g. polar vortex or stormfronts).
          2. Local station microclimate idiosyncrasies.
          3. Physical sensor malfunction or calibration drift.
        - **Contamination Parameter:** The contamination hyperparameter ($0.03$) represents an operational detection budget, not an empirically measured sensor failure rate.
        """
    )

# -----------------------------------------------------------------------------
# SECTION 6: DATA MANAGEMENT & LIVE INFERENCE
# -----------------------------------------------------------------------------
elif selected_section == "6. Data Management & Upload":
    st.header("6. Data Management & Live Model Inference")
    st.write(
        "Upload a new weather CSV file to validate schema conformity, perform missing-value checks, "
        "and execute live anomaly detection using the pre-trained Isolation Forest model without retraining."
    )

    uploaded_file = st.file_uploader("Upload Weather CSV File:", type=["csv"])

    if uploaded_file is not None:
        try:
            user_df = pd.read_csv(uploaded_file)
            st.success(f"File uploaded successfully! Loaded {len(user_df):,} rows.")

            # Schema validation
            is_valid, missing_cols = validate_weather_schema(user_df)
            if not is_valid:
                st.error(f"Schema Validation Failed: Missing required columns: {missing_cols}")
            else:
                st.success("✅ Schema Validation Passed: All required weather features are present.")

                # Clean measurements
                clean_user_df = user_df.copy()
                for col in ["TMIN", "TAVG", "TMAX", "PRCP", "WSPD"]:
                    if col in clean_user_df.columns:
                        clean_user_df[col] = convert_measurement_string(clean_user_df[col])

                # Missing value audit
                st.subheader("Uploaded Dataset Missing-Value Audit")
                missing_report = pd.DataFrame({
                    "Missing Count": clean_user_df[["TMIN", "TAVG", "TMAX", "PRCP", "WSPD"]].isna().sum(),
                    "Missing Pct (%)": (clean_user_df[["TMIN", "TAVG", "TMAX", "PRCP", "WSPD"]].isna().mean() * 100).round(2),
                })
                st.dataframe(missing_report.T, use_container_width=True)

                if st.button("🚀 Run Live Anomaly Detection with Pre-trained Model"):
                    with st.spinner("Executing inference using pre-trained Isolation Forest..."):
                        # Transform with pre-trained preprocessor
                        X_user = preprocessor.transform(clean_user_df)
                        preds_user = model.predict(X_user)
                        scores_user = model.decision_function(X_user)

                        clean_user_df["anomaly_score"] = scores_user
                        clean_user_df["anomaly"] = preds_user == -1
                        clean_user_df["anomaly_label"] = np.where(clean_user_df["anomaly"], "Flagged", "Normal")

                        u_total = len(clean_user_df)
                        u_flagged = int(clean_user_df["anomaly"].sum())
                        u_pct = (u_flagged / u_total * 100) if u_total else 0.0

                        st.markdown("### **Live Inference Results**")
                        ucol1, ucol2, ucol3 = st.columns(3)
                        ucol1.metric("Uploaded Records", f"{u_total:,}")
                        ucol2.metric("Flagged Anomalies", f"{u_flagged:,}")
                        ucol3.metric("Anomaly Rate", f"{u_pct:.2f}%")

                        st.dataframe(
                            clean_user_df.sort_values("anomaly_score"),
                            use_container_width=True,
                        )

                        st.download_button(
                            "📥 Download Live Detection Results (CSV)",
                            data=clean_user_df.to_csv(index=False).encode("utf-8"),
                            file_name="skyguard_live_inference_results.csv",
                            mime="text/csv",
                        )

        except Exception as e:
            st.error(f"Error processing uploaded CSV: {e}")
    else:
        st.info("Upload any weather station CSV containing columns: City, Country, DATE, TMIN, TAVG, TMAX, PRCP, WSPD.")

st.divider()
st.caption("SkyGuard | INT396 Unsupervised Learning | Professional Academic Project")
