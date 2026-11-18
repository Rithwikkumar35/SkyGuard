"""
docs/generate_presentation.py
Generates the professional university presentation slides in PPTX format for SkyGuard.
Uses python-pptx with a modern 16:9 widescreen layout, custom typography, and metric cards.
"""

import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

from src.utils import get_docs_dir, get_results_dir


def create_presentation():
    docs_dir = get_docs_dir()
    results_dir = get_results_dir()
    pptx_path = docs_dir / "PRESENTATION.pptx"

    prs = Presentation()
    # 16:9 Widescreen dimensions
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # Color palette
    COLOR_BG_DARK = RGBColor(15, 23, 42)     # #0f172a
    COLOR_BG_LIGHT = RGBColor(248, 250, 252) # #f8fafc
    COLOR_CARD = RGBColor(255, 255, 255)
    COLOR_CARD_BORDER = RGBColor(226, 232, 240)
    COLOR_TEXT_DARK = RGBColor(15, 23, 42)
    COLOR_TEXT_MUTED = RGBColor(100, 116, 139) # #64748b
    COLOR_PRIMARY = RGBColor(2, 132, 199)    # #0284c7 (Cyan-Blue)
    COLOR_ACCENT = RGBColor(16, 185, 129)    # #10b981 (Emerald)
    COLOR_DANGER = RGBColor(239, 68, 68)     # #ef4444 (Crimson)
    COLOR_WHITE = RGBColor(255, 255, 255)

    def set_slide_background(slide, color):
        background = slide.background
        fill = background.fill
        fill.solid()
        fill.fore_color.rgb = color

    def add_header(slide, title_text, category_text="SKYGUARD | INT396 UNSUPERVISED LEARNING"):
        # Header banner text
        txBox = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.733), Inches(1.1))
        tf = txBox.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

        p_cat = tf.paragraphs[0]
        p_cat.text = category_text.upper()
        p_cat.font.name = "Helvetica"
        p_cat.font.size = Pt(10)
        p_cat.font.bold = True
        p_cat.font.color.rgb = COLOR_PRIMARY
        p_cat.space_after = Pt(2)

        p_title = tf.add_paragraph()
        p_title.text = title_text
        p_title.font.name = "Helvetica"
        p_title.font.size = Pt(22)
        p_title.font.bold = True
        p_title.font.color.rgb = COLOR_TEXT_DARK

    def add_card(slide, left, top, width, height, title, content_items, border_color=COLOR_CARD_BORDER):
        shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        shape.fill.solid()
        shape.fill.fore_color.rgb = COLOR_CARD
        shape.line.color.rgb = border_color
        shape.line.width = Pt(1.5)

        tf = shape.text_frame
        tf.word_wrap = True
        tf.vertical_anchor = MSO_ANCHOR.TOP
        tf.margin_left = Inches(0.3)
        tf.margin_top = Inches(0.25)
        tf.margin_right = Inches(0.3)
        tf.margin_bottom = Inches(0.25)

        p_title = tf.paragraphs[0]
        p_title.text = title
        p_title.font.name = "Helvetica"
        p_title.font.size = Pt(13)
        p_title.font.bold = True
        p_title.font.color.rgb = COLOR_TEXT_DARK
        p_title.space_after = Pt(8)

        for item in content_items:
            p = tf.add_paragraph()
            p.text = f"• {item}" if not item.startswith("  -") else f"    {item}"
            p.font.name = "Helvetica"
            p.font.size = Pt(10)
            p.font.color.rgb = COLOR_TEXT_MUTED
            p.space_after = Pt(4)

    # -------------------------------------------------------------------------
    # SLIDE 1: TITLE SLIDE (Dark Navy)
    # -------------------------------------------------------------------------
    s1 = prs.slides.add_slide(blank_layout)
    set_slide_background(s1, COLOR_BG_DARK)

    # Accent top border
    bar = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.2), Inches(0.15), Inches(4.5))
    bar.fill.solid()
    bar.fill.fore_color.rgb = COLOR_PRIMARY
    bar.line.fill.background()

    tb1 = s1.shapes.add_textbox(Inches(1.2), Inches(1.2), Inches(11.0), Inches(4.5))
    tf1 = tb1.text_frame
    tf1.word_wrap = True

    p = tf1.paragraphs[0]
    p.text = "INT396 — UNSUPERVISED MACHINE LEARNING"
    p.font.name = "Helvetica"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY
    p.space_after = Pt(12)

    p = tf1.add_paragraph()
    p.text = "🛡️ SkyGuard: Weather-Station Sensor Anomaly Detection"
    p.font.name = "Helvetica"
    p.font.size = Pt(32)
    p.font.bold = True
    p.font.color.rgb = COLOR_WHITE
    p.space_after = Pt(10)

    p = tf1.add_paragraph()
    p.text = "An Unsupervised Framework for Sensor Quality Assurance & Extreme Weather Discovery"
    p.font.name = "Helvetica"
    p.font.size = Pt(16)
    p.font.color.rgb = RGBColor(148, 163, 184)
    p.space_after = Pt(28)

    p = tf1.add_paragraph()
    p.text = "Team Members: [STUDENT_NAME_1] (Lead ML) | [STUDENT_NAME_2] (Data) | [STUDENT_NAME_3] (UI/UX)\n" \
             "Faculty Mentor: [FACULTY_MENTOR_NAME] | Department of Computer Science & Engineering\n" \
             "Problem Statement ID: [PS-INT396-WEATHER-01] | Academic Year 2024-2025"
    p.font.name = "Helvetica"
    p.font.size = Pt(11)
    p.font.color.rgb = RGBColor(203, 213, 225)

    # -------------------------------------------------------------------------
    # SLIDE 2: PROBLEM STATEMENT & MOTIVATION
    # -------------------------------------------------------------------------
    s2 = prs.slides.add_slide(blank_layout)
    set_slide_background(s2, COLOR_BG_LIGHT)
    add_header(s2, "Problem Statement, Academic Need & Telemetry Challenges")

    add_card(s2, Inches(0.8), Inches(1.8), Inches(3.64), Inches(4.8),
             "1. The Telemetry Challenge",
             ["Automated Weather Stations (AWS) gather continuous environmental telemetry worldwide.",
              "Vulnerable to mechanical stiction, rain gauge clogging, and calibration drift.",
              "Transmission errors and calculation discrepancies corrupt numerical weather forecasting.",
              "Fouled observations compromise municipal flood warnings and aviation safety."],
             COLOR_PRIMARY)

    add_card(s2, Inches(4.84), Inches(1.8), Inches(3.64), Inches(4.8),
             "2. The Unsupervised Reality",
             ["Real-world sensor networks operate completely without ground-truth failure annotations.",
              "Supervised techniques (accuracy, precision, recall) cannot be legitimately reported.",
              "Fabricating synthetic labels invalidates scientific credibility.",
              "Must discover multidimensional geometric anomalies without pre-labelled data."],
             COLOR_PRIMARY)

    add_card(s2, Inches(8.88), Inches(1.8), Inches(3.64), Inches(4.8),
             "3. The SkyGuard Objective",
             ["Develop a leakage-free unsupervised anomaly detection system using Isolation Forest.",
              "Perform rigorous meteorological feature engineering (DTR, midpoint deviation).",
              "Execute multi-paradigm benchmarking against Local Outlier Factor and DBSCAN.",
              "Deploy a production-grade Streamlit intelligence platform with zero retraining overhead."],
             COLOR_ACCENT)

    # -------------------------------------------------------------------------
    # SLIDE 3: SYSTEM ARCHITECTURE
    # -------------------------------------------------------------------------
    s3 = prs.slides.add_slide(blank_layout)
    set_slide_background(s3, COLOR_BG_LIGHT)
    add_header(s3, "End-to-End System Pipeline & Decoupled Architecture")

    stages = [
        ("Phase 1 & 2: Ingestion & Audit", Inches(0.8), Inches(1.8), Inches(5.66), Inches(2.2),
         ["Ingests 6,077 global observation records across 103 cities and 35 countries.",
          "Strips unit strings (°C, mm, km/h, hPa) and extracts geographic coordinates.",
          "Automated 6-point physical audit flags 7 thermodynamic violations (TAVG > TMAX)."]),
        ("Phase 2: Feature Engineering", Inches(6.86), Inches(1.8), Inches(5.66), Inches(2.2),
         ["Diurnal Temperature Range (DTR = TMAX - TMIN) captures thermal oscillation.",
          "Diurnal Midpoint Deviation (TAVG - (TMAX+TMIN)/2) catches sensor skew.",
          "Log-precipitation reduces right-skewness from 5.72 to 1.73; city-relative departures."]),
        ("Phase 3: Unsupervised Modeling", Inches(0.8), Inches(4.4), Inches(5.66), Inches(2.2),
         ["Isolation Forest ensemble (150 trees, 0.03 contamination, random seed 42).",
          "Calculates continuous decision scores; stratifies into Severe, High, Moderate.",
          "Zero data leakage: simple imputer medians and scalers frozen and serialized."]),
        ("Phase 4: Web Intelligence Platform", Inches(6.86), Inches(4.4), Inches(5.66), Inches(2.2),
         ["6-section interactive Streamlit platform utilizing Plotly for dynamic exploration.",
          "Zero model retraining: loads pre-trained .joblib binaries directly from models/.",
          "Supports live CSV upload, schema validation, and real-time inference."]),
    ]
    for title, l, t, w, h, items in stages:
        add_card(s3, l, t, w, h, title, items, COLOR_PRIMARY)

    # -------------------------------------------------------------------------
    # SLIDE 4: DATASET QUALITY & PHYSICAL AUDIT
    # -------------------------------------------------------------------------
    s4 = prs.slides.add_slide(blank_layout)
    set_slide_background(s4, COLOR_BG_LIGHT)
    add_header(s4, "Dataset Integrity, Missingness Audit & Physical Anomaly Discovery")

    add_card(s4, Inches(0.8), Inches(1.8), Inches(5.66), Inches(4.8),
             "Dataset Properties & Exclusion Audit",
             ["Corpus: 6,077 daily records, 103 cities, 35 countries (Jan 01 – Feb 28, 2025).",
              "Retained Baseline Features (100% complete or < 8% missing):",
              "  - TMIN, TAVG, TMAX: Minimum, Mean, and Maximum Temperature (0% missing).",
              "  - PRCP: Daily Liquid Precipitation (7.77% missing; median imputed).",
              "  - WSPD: Sustained Wind Speed (0% missing).",
              "Excluded High-Missingness Features (> 50% Threshold):",
              "  - SNOW: 92.48% missing (5,620 null values).",
              "  - TSUN: 97.09% missing (5,900 null values).",
              "  - WPGT: 75.73% missing (4,602 null values).",
              "  - PRES: Confounded by station elevation (reflects altitude, not weather)."],
             COLOR_PRIMARY)

    add_card(s4, Inches(6.86), Inches(1.8), Inches(5.66), Inches(4.8),
             "Groundbreaking Physical Boundary Audit Findings",
             ["Automated physical law checks revealed real-world sensor artifacts:",
              "1. Thermodynamic check: TMIN <= TMAX (0 violations across 6,077 records).",
              "2. Daily Mean check: TMIN <= TAVG <= TMAX -> EXACTLY 7 PHYSICAL VIOLATIONS!",
              "  - Melbourne (Row 5610): TMIN=16.4°C, TMAX=18.4°C, TAVG=19.4°C! (Average > Max!)",
              "  - Valencia (Rows 188, 225): TAVG is strictly lower than TMIN.",
              "  - Kyoto, Sydney, Brisbane, Perth: TAVG slightly below recorded TMIN.",
              "Root Cause: Boundary-window hourly averaging truncations & calibration bias.",
              "Non-negativity: Precipitation >= 0 and Wind Speed >= 0 (0 violations)."],
             COLOR_DANGER)

    # -------------------------------------------------------------------------
    # SLIDE 5: METEOROLOGICAL FEATURE ENGINEERING
    # -------------------------------------------------------------------------
    s5 = prs.slides.add_slide(blank_layout)
    set_slide_background(s5, COLOR_BG_LIGHT)
    add_header(s5, "Domain Meteorological Feature Engineering")

    feat_cards = [
        ("1. Diurnal Temperature Range (DTR)", Inches(0.8), Inches(1.8), Inches(5.66), Inches(2.2),
         ["Formula: DTR = TMAX - TMIN",
          "Mean: 8.35 °C | Min: 0.80 °C | Max: 25.0 °C",
          "Significance: Near-zero DTR detects stuck temperature probes or thick inversions; extreme DTR detects radiation shield detachment."]),
        ("2. Diurnal Midpoint Deviation (ΔT)", Inches(6.86), Inches(1.8), Inches(5.66), Inches(2.2),
         ["Formula: ΔT = TAVG - (TMAX + TMIN) / 2",
          "Mean: -0.31 °C | Min: -6.85 °C | Max: +6.10 °C",
          "Significance: Measures daily thermal progression asymmetry caused by sudden daytime cold fronts or thermal lag."]),
        ("3. Log-Transformed Rainfall", Inches(0.8), Inches(4.4), Inches(5.66), Inches(2.2),
         ["Formula: PRCP_log = ln(1 + max(0, PRCP))",
          "Addresses extreme 66.5% zero-inflation of precipitation.",
          "Dramatic Mathematical Impact: Skewness reduced from 5.72 down to 1.73! Prevents torrential extremes from distorting hyperplanes."]),
        ("4. City-Relative Temperature Anomaly", Inches(6.86), Inches(4.4), Inches(5.66), Inches(2.2),
         ["Formula: ΔT_city = TAVG - Mean(TAVG_city)",
          "Normalizes station temperatures relative to local 2-month baseline.",
          "Crucial Insight: Prevents sub-zero Canadian winter stations from being flagged purely due to geographical latitude."]),
    ]
    for title, l, t, w, h, items in feat_cards:
        add_card(s5, l, t, w, h, title, items, COLOR_PRIMARY)

    # -------------------------------------------------------------------------
    # SLIDE 6: ISOLATION FOREST METHODOLOGY
    # -------------------------------------------------------------------------
    s6 = prs.slides.add_slide(blank_layout)
    set_slide_background(s6, COLOR_BG_LIGHT)
    add_header(s6, "Machine Learning Methodology: Isolation Forest Mathematical Theory")

    add_card(s6, Inches(0.8), Inches(1.8), Inches(5.66), Inches(4.8),
             "Algorithmic Mechanics & Formulation",
             ["Isolates anomalies instead of profiling normal clusters.",
              "Builds an ensemble of randomized decision trees (iTrees).",
              "Recursive random cuts: Anomalies require fewer splits to isolate into terminal leaves.",
              "Mathematical Anomaly Score Formulation:",
              "  s(x, n) = 2^(-E(h(x)) / c(n))",
              "  - h(x): Observation path length from root to leaf.",
              "  - E(h(x)): Average path length across 150 isolation trees.",
              "  - c(n) = 2(ln(n-1) + 0.5772156649) - 2(n-1)/n (BST search depth).",
              "Score Interpretation:",
              "  - s -> 1.0 (Short path): High-confidence Anomaly.",
              "  - s < 0.5 (Long path): Standard Normal observation."],
             COLOR_PRIMARY)

    add_card(s6, Inches(6.86), Inches(1.8), Inches(5.66), Inches(4.8),
             "Operational Configuration & Severity Ranking",
             ["Production Model Hyperparameters:",
              "  - Estimators (n_estimators): 150 isolation trees.",
              "  - Contamination Setting: 0.03 (3.0% expected budget).",
              "  - Random Seed: 42 (Guarantees strict bit-for-bit reproducibility).",
              "  - Preprocessing: Zero-leakage median imputation + StandardScaler.",
              "Continuous Decision Function & Empirical Severity:",
              "  - Decision threshold offset: -0.5947 (Scores <= 0.0 flagged as anomalous).",
              "  - Severe Tier (Score <= p33 of anomalies): Top extreme events.",
              "  - High Tier (p33 < Score <= p66): Moderate-to-high anomalies.",
              "  - Moderate Tier (Score > p66): Marginal boundary anomalies."],
             COLOR_ACCENT)

    # -------------------------------------------------------------------------
    # SLIDE 7: SENSITIVITY ANALYSIS
    # -------------------------------------------------------------------------
    s7 = prs.slides.add_slide(blank_layout)
    set_slide_background(s7, COLOR_BG_LIGHT)
    add_header(s7, "Model Evaluation & Contamination Sensitivity Analysis")

    add_card(s7, Inches(0.8), Inches(1.8), Inches(5.66), Inches(4.8),
             "Contamination Sensitivity Experiments",
             ["Evaluated model across 4 operational contamination settings:",
              "• Contamination 0.01 (1%): 61 flagged records | Mean score: 0.2095",
              "• Contamination 0.03 (3%): 183 flagged records | Mean score: 0.1521",
              "• Contamination 0.05 (5%): 304 flagged records | Mean score: 0.1193",
              "• Contamination 0.10 (10%): 608 flagged records | Mean score: 0.0736",
              "Empirical Finding:",
              "  - Tree traversal path lengths and anomaly rankings remain strictly invariant.",
              "  - The contamination parameter acts solely as an operational threshold offset.",
              "  - Sub-150ms execution speed across all contamination levels."],
             COLOR_PRIMARY)

    add_card(s7, Inches(6.86), Inches(1.8), Inches(5.66), Inches(4.8),
             "Decision-Function Score Distribution",
             ["Empirical Score Statistics (6,077 observations):",
              "  - Mean Anomaly Score: +0.1521 (Std: 0.0560)",
              "  - Minimum Score: -0.1287 (Severe Polar Cold in Calgary)",
              "  - Maximum Score: +0.2146 (Prototypical temperate observation)",
              "Clear Separation Profile:",
              "  - Flagged anomalies (183 records) form a distinct negative tail (score <= 0.0).",
              "  - Normal observations (5,894 records) exhibit a clean Gaussian-like distribution centered at +0.17.",
              "  - High peak separation confirms model selectivity."],
             COLOR_PRIMARY)

    # -------------------------------------------------------------------------
    # SLIDE 8: ALGORITHM BENCHMARK
    # -------------------------------------------------------------------------
    s8 = prs.slides.add_slide(blank_layout)
    set_slide_background(s8, COLOR_BG_LIGHT)
    add_header(s8, "Multi-Paradigm Comparative Benchmark: IF vs LOF vs DBSCAN")

    add_card(s8, Inches(0.8), Inches(1.8), Inches(3.64), Inches(4.8),
             "Isolation Forest",
             ["Paradigm: Randomized Tree Partitioning",
              "Flagged Anomalies: 183 (3.01%)",
              "Silhouette Score: 0.5518 (Excellent separation)",
              "Runtime Latency: 135.8 ms",
              "Strengths: Isolates multivariate outliers across continuous features without distance calculation bottlenecks.",
              "Rank: PRIMARY RECOMMENDED MODEL"],
             COLOR_ACCENT)

    add_card(s8, Inches(4.84), Inches(1.8), Inches(3.64), Inches(4.8),
             "Local Outlier Factor (LOF)",
             ["Paradigm: K-NN Local Density Deviation",
              "Flagged Anomalies: 183 (3.01%)",
              "Silhouette Score: 0.2246 (Lower separation)",
              "Runtime Latency: 27.1 ms",
              "Analysis: Highly sensitive to local density pockets; flags boundary points near clusters rather than global outliers.",
              "Jaccard Overlap with IF: 31.65%"],
             COLOR_PRIMARY)

    add_card(s8, Inches(8.88), Inches(1.8), Inches(3.64), Inches(4.8),
             "DBSCAN",
             ["Paradigm: Density Clustering (Noise Detection)",
              "Flagged Anomalies: 14 (0.23%)",
              "Silhouette Score: 0.7638 (High score on tiny subset)",
              "Runtime Latency: 127.1 ms",
              "Analysis: Severe under-detection in 5-D standardized weather data; misses subtler multi-feature anomalies.",
              "Jaccard Overlap with IF: 7.11%"],
             COLOR_PRIMARY)

    # -------------------------------------------------------------------------
    # SLIDE 9: STREAMLIT PLATFORM
    # -------------------------------------------------------------------------
    s9 = prs.slides.add_slide(blank_layout)
    set_slide_background(s9, COLOR_BG_LIGHT)
    add_header(s9, "Professional Streamlit Weather Intelligence Platform")

    add_card(s9, Inches(0.8), Inches(1.8), Inches(5.66), Inches(4.8),
             "6 Production Dashboard Sections",
             ["Section 1: Overview & Executive Summary",
              "  - KPI cards, geospatial natural-earth map of 103 global stations, extreme records.",
              "Section 2: Exploratory Weather Analysis",
              "  - Daily temperature progression curves, DTR histogram, rainfall and wind distributions.",
              "Section 3: Model Evaluation & Benchmarks",
              "  - Theoretical explanations, score distributions, contamination curves, LOF/DBSCAN tables.",
              "Section 4: Anomaly Detection & Records",
              "  - Multi-parameter filtering (City, Country, Severity), CSV exports, anomaly driver chart.",
              "Section 5: Methodology & System Architecture",
              "  - Transparent pipeline diagrams, mathematical formulas, and academic limitation notes.",
              "Section 6: Data Management & Live Inference",
              "  - Upload new CSVs, schema validation, and live scoring with zero retraining."],
             COLOR_PRIMARY)

    add_card(s9, Inches(6.86), Inches(1.8), Inches(5.66), Inches(4.8),
             "Grounded Anomaly Drivers Discovered",
             ["Top Anomalies Correlate with Verified Meteorological Events:",
              "• Polar Vortex Cold Waves:",
              "  - Calgary (CA): -29.0°C Min, -24.2°C Mean (Score: -0.1287, Severe).",
              "  - Chicago (US): -21.1°C Min during mid-January Arctic plunge.",
              "• Torrential Winter Storm Deluges:",
              "  - Montreal (CA): 27.5 mm liquid equivalent precipitation (Score: -0.1109).",
              "  - Sydney (AU): 111.5 mm deluge recorded.",
              "• Severe Gale Wind Storms:",
              "  - Reykjavik (IS): 57.0 km/h sustained gale.",
              "  - Toronto (CA): 35.6 km/h storm wind during winter blizzard.",
              "• Physical Sensor Inconsistencies:",
              "  - Melbourne: Flagged for TAVG exceeding recorded TMAX."],
             COLOR_ACCENT)

    # -------------------------------------------------------------------------
    # SLIDE 10: VIVA DEFENSE, LIMITATIONS & FUTURE SCOPE
    # -------------------------------------------------------------------------
    s10 = prs.slides.add_slide(blank_layout)
    set_slide_background(s10, COLOR_BG_LIGHT)
    add_header(s10, "Academic Viva Defense, Limitations & Future Scope")

    add_card(s10, Inches(0.8), Inches(1.8), Inches(5.66), Inches(4.8),
             "Viva Defense & Academic Integrity",
             ["Why are Accuracy, Precision, Recall, and F1 Omitted?",
              "  - Because real-world weather stations lack ground-truth sensor failure labels.",
              "  - Reporting supervised metrics on unlabelled telemetry is scientifically unsound.",
              "What Does an Isolation Forest Flag Actually Mean?",
              "  - It denotes multidimensional statistical atypicality.",
              "  - Can indicate genuine severe weather, local station microclimates, or physical sensor malfunction.",
              "How Was Data Leakage Prevented?",
              "  - Missing value imputation medians and scaling parameters were fitted strictly on training data and frozen.",
              "Comprehensive Automated Testing:",
              "  - 25 pytest test cases passed in 5.74 seconds with 100% pass rate."],
             COLOR_PRIMARY)

    add_card(s10, Inches(6.86), Inches(1.8), Inches(5.66), Inches(4.8),
             "Future Scope & Industrial Extension",
             ["1. Spatio-Temporal Graph Neural Networks (GNNs):",
              "  - Exploit spatial correlations between neighboring weather stations to detect localized sensor drift.",
              "2. Multivariate Variational Autoencoders (VAEs):",
              "  - Reconstruct weather profiles to isolate non-linear reconstruction errors.",
              "3. Edge-Device Deployment (TinyML):",
              "  - Quantize the Isolation Forest model to run directly on solar-powered microcontroller loggers (ESP32, RP2040).",
              "4. Real-Time Streaming Ingestion:",
              "  - Integrate Apache Kafka / MQTT for continuous real-time anomaly detection."],
             COLOR_ACCENT)

    prs.save(str(pptx_path))
    print(f"Presentation PPTX generated successfully at: {pptx_path}")
    return pptx_path


if __name__ == "__main__":
    create_presentation()
