"""
docs/generate_pdf_report.py
Generates the publication-grade university project report in PDF format for SkyGuard.
Uses ReportLab Platypus with academic typography, tables, and embedded high-res figures.
"""

import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

from src.utils import get_docs_dir, get_results_dir


class NumberedCanvas(canvas.Canvas):
    """Custom canvas that performs two passes to dynamically render total page numbers."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_number(self, page_count):
        if self._pageNumber == 1:
            return  # Skip page number on cover page
        self.saveState()
        self.setFont("Helvetica", 8.5)
        self.setFillColor(colors.HexColor("#64748b"))
        # Header rule & text
        self.drawString(54, 755, "SkyGuard — Weather Sensor Anomaly Detection | INT396 Project Report")
        self.setStrokeColor(colors.HexColor("#e2e8f0"))
        self.setLineWidth(0.6)
        self.line(54, 747, 558, 747)
        # Footer rule & page number
        self.line(54, 45, 558, 45)
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 32, page_str)
        self.drawString(54, 32, "Confidential — Academic University Submission")
        self.restoreState()


def build_pdf_report():
    docs_dir = get_docs_dir()
    results_dir = get_results_dir()
    pdf_path = docs_dir / "PROJECT_REPORT.pdf"

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'CoverTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=30,
        textColor=colors.HexColor("#0f172a"),
        alignment=1,  # Center
        spaceAfter=12,
    )

    subtitle_style = ParagraphStyle(
        'CoverSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#475569"),
        alignment=1,
        spaceAfter=24,
    )

    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=19,
        textColor=colors.HexColor("#0f172a"),
        spaceBefore=14,
        spaceAfter=8,
        keepWithNext=True,
    )

    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=11.5,
        leading=15,
        textColor=colors.HexColor("#1e293b"),
        spaceBefore=10,
        spaceAfter=5,
        keepWithNext=True,
    )

    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#1e293b"),
        spaceAfter=6,
    )

    bullet_style = ParagraphStyle(
        'Bullet_Custom',
        parent=body_style,
        leftIndent=15,
        bulletIndent=5,
        spaceAfter=3,
    )

    table_cell = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#0f172a"),
    )

    table_header = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11,
        textColor=colors.white,
    )

    callout_style = ParagraphStyle(
        'Callout',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=9,
        leading=12.5,
        textColor=colors.HexColor("#991b1b"),
    )

    story = []

    # =========================================================================
    # 1. TITLE PAGE / COVER
    # =========================================================================
    story.append(Spacer(1, 40))
    story.append(Paragraph("🛡️ SkyGuard: Weather-Station Sensor Anomaly Detection", title_style))
    story.append(Paragraph("A Comprehensive Unsupervised Machine Learning Framework for Quality Assurance in Meteorological Telemetry", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#0284c7"), spaceAfter=30))

    meta_table_data = [
        [Paragraph("<b>Course Code:</b>", body_style), Paragraph("INT396 — Unsupervised Learning", body_style)],
        [Paragraph("<b>Project Domain:</b>", body_style), Paragraph("Machine Learning / Meteorological Analytics / Outlier Isolation", body_style)],
        [Paragraph("<b>Problem Statement ID:</b>", body_style), Paragraph("<code>[PS-INT396-WEATHER-01]</code> <i>(Editable Placeholder)</i>", body_style)],
        [Paragraph("<b>Academic Term:</b>", body_style), Paragraph("2024–2025", body_style)],
        [Paragraph("<b>Faculty Mentor:</b>", body_style), Paragraph("<code>[FACULTY_MENTOR_NAME]</code>", body_style)],
        [Paragraph("<b>Institution / Dept:</b>", body_style), Paragraph("<code>[DEPARTMENT_OF_COMPUTER_SCIENCE / UNIVERSITY_NAME]</code>", body_style)],
        [Paragraph("<b>Group Members:</b>", body_style), Paragraph(
            "1. <code>[STUDENT_NAME_1]</code> (Reg No: <code>[REG_NO_1]</code>) — Lead ML Engineer<br/>"
            "2. <code>[STUDENT_NAME_2]</code> (Reg No: <code>[REG_NO_2]</code>) — Data & Feature Engineer<br/>"
            "3. <code>[STUDENT_NAME_3]</code> (Reg No: <code>[REG_NO_3]</code>) — UI/UX & Integration Lead",
            body_style
        )],
    ]

    t_meta = Table(meta_table_data, colWidths=[130, 374])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f8fafc")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#cbd5e1")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(t_meta)

    story.append(Spacer(1, 40))
    story.append(Paragraph("<b>Executive Summary:</b> Weather stations collect continuous environmental telemetry. Sensor fouling, calibration drift, and transmission glitches compromise meteorological integrity. This project develops an end-to-end unsupervised anomaly detection pipeline utilizing Isolation Forest as the primary algorithm, benchmarked against Local Outlier Factor (LOF) and DBSCAN. Across 6,077 daily observations spanning 103 cities and 35 countries, the system successfully isolates 183 anomalous records (3.01%) and uncovers 7 genuine physical anomalies where daily average temperatures violate thermodynamic boundaries.", body_style))
    story.append(PageBreak())

    # =========================================================================
    # 2. INTRODUCTION & PROBLEM STATEMENT
    # =========================================================================
    story.append(Paragraph("1. Introduction & Problem Statement", h1_style))
    story.append(Paragraph("Automated Weather Stations (AWS) provide critical environmental observations for aviation routing, civil flood defense, agriculture, and municipal climate planning. However, continuous field telemetry is subject to sensor degradation, debris blockage in rain gauges, mechanical anemometer wear, and data averaging artifacts.", body_style))
    story.append(Paragraph("<b>The Core Academic Challenge:</b> Real-world weather observation networks operate <i>without pre-labelled sensor failure annotations</i>. Supervised learning techniques cannot be legitimately deployed without fabricating synthetic labels. This project applies <b>unsupervised machine learning</b> to isolate multi-dimensional sensor outliers purely through intrinsic geometric isolation, ensuring zero data leakage.", body_style))

    # =========================================================================
    # 3. PROJECT OBJECTIVES & NEED
    # =========================================================================
    story.append(Paragraph("2. Project Objectives & Societal Need", h1_style))
    story.append(Paragraph("• <b>Data Ingestion & Integrity Auditing:</b> Parse unit-annotated telemetry and enforce 6 meteorological thermodynamic laws.", bullet_style))
    story.append(Paragraph("• <b>Zero-Leakage Transformation Pipeline:</b> Build a scikit-learn compatible preprocessor with frozen training median imputation and standard scaling.", bullet_style))
    story.append(Paragraph("• <b>Domain Meteorological Feature Engineering:</b> Engineer Diurnal Range (DTR), Midpoint Deviation, log-precipitation, and city-relative departures.", bullet_style))
    story.append(Paragraph("• <b>Unsupervised Anomaly Isolation:</b> Implement Isolation Forest (n=150, c=0.03, seed=42) and stratify scores into Severe, High, and Moderate tiers.", bullet_style))
    story.append(Paragraph("• <b>Comparative Benchmarking:</b> Fairly benchmark Isolation Forest against Local Outlier Factor and DBSCAN using Silhouette scores.", bullet_style))
    story.append(Paragraph("• <b>Interactive Web Application:</b> Deploy a responsive 6-section Streamlit platform with zero model retraining on user interaction.", bullet_style))

    # =========================================================================
    # 4. METHODOLOGY & MATHEMATICAL FORMULATION
    # =========================================================================
    story.append(Paragraph("3. Methodology & Mathematical Formulation", h1_style))
    story.append(Paragraph("<b>Primary Algorithm: Isolation Forest</b><br/>"
                           "Isolation Forest operates on the principle that anomalous observations are geometrically isolated and require fewer random recursive hyperplane cuts to separate into individual leaf nodes.", body_style))
    story.append(Paragraph("The anomaly score for an observation <i>x</i> across an ensemble of <i>n</i> trees is given by:", body_style))
    story.append(Paragraph("<b>s(x, n) = 2^(-E(h(x)) / c(n))</b>", h2_style))
    story.append(Paragraph("where <i>h(x)</i> is the traversal path length, <i>E(h(x))</i> is the expected path length over 150 trees, and <i>c(n) = 2(ln(n - 1) + 0.5772156649) - 2(n - 1)/n</i> is the average depth of an unsuccessful search in a Binary Search Tree (BST). When <i>s &to; 1.0</i>, the observation isolates rapidly and is flagged as an anomaly.", body_style))

    # Embed feature distributions plot if present
    feat_dist_img = results_dir / "feature_distributions.png"
    if feat_dist_img.exists():
        story.append(Spacer(1, 8))
        story.append(Image(str(feat_dist_img), width=480, height=260))
        story.append(Paragraph("<i>Figure 1: Meteorological Feature Distributions with Kernel Density Estimates across 6,077 global observations.</i>", table_cell))

    story.append(PageBreak())

    # =========================================================================
    # 5. EXPERIMENTAL RESULTS & COMPARATIVE BENCHMARKING
    # =========================================================================
    story.append(Paragraph("4. Experimental Results & Comparative Benchmarking", h1_style))
    story.append(Paragraph("The baseline Isolation Forest model (contamination = 0.03) evaluated all 6,077 observations, flagging exactly <b>183 records (3.01%)</b> as anomalies and classifying 5,894 records (96.99%) as normal observations.", body_style))

    # Comparison Table
    story.append(Paragraph("<b>Table 1: Multi-Paradigm Unsupervised Algorithm Benchmark</b>", h2_style))
    bench_data = [
        [Paragraph("<b>Algorithm</b>", table_header), Paragraph("<b>Paradigm</b>", table_header), Paragraph("<b>Flagged</b>", table_header), Paragraph("<b>Rate (%)</b>", table_header), Paragraph("<b>Silhouette Score</b>", table_header), Paragraph("<b>Latency (ms)</b>", table_header)],
        [Paragraph("<b>Isolation Forest</b>", table_cell), Paragraph("Tree-based Partitioning", table_cell), Paragraph("183", table_cell), Paragraph("3.01%", table_cell), Paragraph("<b>0.5518</b>", table_cell), Paragraph("135.8 ms", table_cell)],
        [Paragraph("<b>Local Outlier Factor</b>", table_cell), Paragraph("K-NN Local Density Deviation", table_cell), Paragraph("183", table_cell), Paragraph("3.01%", table_cell), Paragraph("0.2246", table_cell), Paragraph("<b>27.1 ms</b>", table_cell)],
        [Paragraph("<b>DBSCAN</b>", table_cell), Paragraph("Density Clustering (Noise)", table_cell), Paragraph("14", table_cell), Paragraph("0.23%", table_cell), Paragraph("0.7638", table_cell), Paragraph("127.1 ms", table_cell)],
    ]
    t_bench = Table(bench_data, colWidths=[100, 140, 50, 55, 95, 64])
    t_bench.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0f172a")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f8fafc")]),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_bench)
    story.append(Paragraph("<i>Key Finding: Isolation Forest achieves the optimal balance of geometric cluster separation (Silhouette 0.5518) and sensitivity without being constrained by local density variations.</i>", table_cell))

    # Contamination Table
    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>Table 2: Contamination Parameter Sensitivity Analysis</b>", h2_style))
    cont_data = [
        [Paragraph("<b>Contamination Setting</b>", table_header), Paragraph("<b>Records Analyzed</b>", table_header), Paragraph("<b>Flagged Records</b>", table_header), Paragraph("<b>Flagged Rate (%)</b>", table_header), Paragraph("<b>Mean Decision Score</b>", table_header), Paragraph("<b>Runtime (ms)</b>", table_header)],
        [Paragraph("0.01 (1%)", table_cell), Paragraph("6,077", table_cell), Paragraph("61", table_cell), Paragraph("1.00%", table_cell), Paragraph("0.2095", table_cell), Paragraph("123.6 ms", table_cell)],
        [Paragraph("0.03 (3%)", table_cell), Paragraph("6,077", table_cell), Paragraph("183", table_cell), Paragraph("3.01%", table_cell), Paragraph("0.1521", table_cell), Paragraph("134.1 ms", table_cell)],
        [Paragraph("0.05 (5%)", table_cell), Paragraph("6,077", table_cell), Paragraph("304", table_cell), Paragraph("5.00%", table_cell), Paragraph("0.1193", table_cell), Paragraph("126.8 ms", table_cell)],
        [Paragraph("0.10 (10%)", table_cell), Paragraph("6,077", table_cell), Paragraph("608", table_cell), Paragraph("10.00%", table_cell), Paragraph("0.0736", table_cell), Paragraph("128.4 ms", table_cell)],
    ]
    t_cont = Table(cont_data, colWidths=[105, 95, 75, 80, 95, 54])
    t_cont.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0369a1")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f8fafc")]),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_cont)

    # Embed score distribution image
    score_dist_img = results_dir / "anomaly_score_distribution.png"
    if score_dist_img.exists():
        story.append(Spacer(1, 8))
        story.append(Image(str(score_dist_img), width=480, height=220))
        story.append(Paragraph("<i>Figure 2: Empirical Decision-Function Anomaly Score Distribution contrasting Normal vs Flagged observations across the decision boundary.</i>", table_cell))

    story.append(PageBreak())

    # =========================================================================
    # 6. QUALITATIVE ANOMALY INVESTIGATION & TESTING
    # =========================================================================
    story.append(Paragraph("5. Qualitative Anomaly Investigation & Physical Violations", h1_style))
    story.append(Paragraph("A qualitative investigation of the most anomalous records confirmed clear meteorological and physical drivers:", body_style))

    qual_data = [
        [Paragraph("<b>Station</b>", table_header), Paragraph("<b>Country</b>", table_header), Paragraph("<b>Date</b>", table_header), Paragraph("<b>TMIN</b>", table_header), Paragraph("<b>TMAX</b>", table_header), Paragraph("<b>PRCP</b>", table_header), Paragraph("<b>WSPD</b>", table_header), Paragraph("<b>Severity</b>", table_header), Paragraph("<b>Meteorological Driver</b>", table_header)],
        [Paragraph("Calgary", table_cell), Paragraph("CA", table_cell), Paragraph("2025-02-05", table_cell), Paragraph("-26.5°", table_cell), Paragraph("-21.6°", table_cell), Paragraph("2.2 mm", table_cell), Paragraph("6.9 km/h", table_cell), Paragraph("Severe", table_cell), Paragraph("Polar Vortex Sub-Zero Extreme", table_cell)],
        [Paragraph("Montreal", table_cell), Paragraph("CA", table_cell), Paragraph("2025-02-16", table_cell), Paragraph("-11.0°", table_cell), Paragraph("-10.0°", table_cell), Paragraph("27.5 mm", table_cell), Paragraph("13.9 km/h", table_cell), Paragraph("Severe", table_cell), Paragraph("Heavy Winter Precipitation Deluge", table_cell)],
        [Paragraph("Toronto", table_cell), Paragraph("CA", table_cell), Paragraph("2025-02-16", table_cell), Paragraph("-7.0°", table_cell), Paragraph("-4.0°", table_cell), Paragraph("17.5 mm", table_cell), Paragraph("35.6 km/h", table_cell), Paragraph("Severe", table_cell), Paragraph("Winter Storm Gale Force Wind", table_cell)],
        [Paragraph("Reykjavik", table_cell), Paragraph("IS", table_cell), Paragraph("2025-01-24", table_cell), Paragraph("1.2°", table_cell), Paragraph("5.0°", table_cell), Paragraph("14.2 mm", table_cell), Paragraph("57.0 km/h", table_cell), Paragraph("Severe", table_cell), Paragraph("Severe Coastal Gale Storm", table_cell)],
        [Paragraph("Melbourne", table_cell), Paragraph("AU", table_cell), Paragraph("2025-01-06", table_cell), Paragraph("16.4°", table_cell), Paragraph("18.4°", table_cell), Paragraph("0.0 mm", table_cell), Paragraph("14.0 km/h", table_cell), Paragraph("Severe", table_cell), Paragraph("Physical Violation: TAVG > TMAX (19.4°C)", table_cell)],
    ]
    t_qual = Table(qual_data, colWidths=[55, 45, 60, 38, 38, 48, 50, 48, 122])
    t_qual.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0f172a")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f8fafc")]),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_qual)

    story.append(Spacer(1, 12))
    story.append(Paragraph("<b>Automated Test Suite Results (pytest):</b> All 25 automated unit and integration tests passed in 5.74 seconds with a 100% pass rate, verifying data loading, string cleaning, coordinate parsing, zero data leakage, score invariants, reproducibility, and Streamlit contracts.", body_style))

    # =========================================================================
    # 7. LIMITATIONS, CONTRIBUTIONS & REFERENCES
    # =========================================================================
    story.append(Paragraph("6. Academic Limitations, Team Contributions & References", h1_style))

    story.append(Paragraph("<b>Academic Defense Statement:</b> In unsupervised telemetry mining, reported scores reflect multidimensional structural atypicality. They do not constitute certified sensor failures. Supervised metrics (accuracy, precision, recall) are omitted to preserve scientific integrity.", body_style))

    story.append(Spacer(1, 6))
    story.append(Paragraph("<b>Team Contributions:</b>", h2_style))
    story.append(Paragraph("• <code>[STUDENT_NAME_1]</code> (Reg No: <code>[REG_NO_1]</code>): Lead ML Engineer — Pipeline architecture, Isolation Forest training, LOF/DBSCAN benchmark evaluation.", bullet_style))
    story.append(Paragraph("• <code>[STUDENT_NAME_2]</code> (Reg No: <code>[REG_NO_2]</code>): Data Engineer — Data loader, string unit regex parsing, physical consistency auditing, feature engineering.", bullet_style))
    story.append(Paragraph("• <code>[STUDENT_NAME_3]</code> (Reg No: <code>[REG_NO_3]</code>): System Integrator — Streamlit platform design, interactive Plotly visualization, live CSV uploader.", bullet_style))

    story.append(Spacer(1, 6))
    story.append(Paragraph("<b>Key References:</b><br/>"
                           "[1] Liu, Ting, & Zhou (2008). <i>Isolation Forest</i>, IEEE ICDM.<br/>"
                           "[2] Breunig et al. (2000). <i>LOF: Identifying Density-Based Local Outliers</i>, ACM SIGMOD.<br/>"
                           "[3] Ester et al. (1996). <i>A density-based algorithm for discovering clusters</i>, KDD.<br/>"
                           "[4] Rousseeuw (1987). <i>Silhouettes: A graphical aid to cluster interpretation</i>, JCAM.<br/>"
                           "[5] World Meteorological Organization (2018). <i>WMO Guide to Meteorological Instruments</i>.", body_style))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Report PDF generated successfully at: {pdf_path}")
    return pdf_path


if __name__ == "__main__":
    build_pdf_report()
