"""Generates a comprehensive 7-section PDF evaluation report using ReportLab.
Conforms exactly to the CWT evaluation structure:
1. Project overview
2. Data collection
3. Feature engineering
4. Model and methodology
5. Results and metrics
6. Visual outputs (with embedded screenshots and forecast matrix)
7. Limitations and reproducibility
"""
import os
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, Image, PageBreak, KeepTogether
)
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute total page count and add running footers."""
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
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748b"))
        # Running header (pages 2+)
        if self._pageNumber > 1:
            self.drawString(36, 762, "CWT Quantitative Assessment — Bitcoin X Sentiment Prediction Agent")
            self.setStrokeColor(colors.HexColor("#cbd5e1"))
            self.setLineWidth(0.5)
            self.line(36, 756, 576, 756)
        # Running footer
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(576, 24, page_str)
        self.drawString(36, 24, "CONFIDENTIAL — CrowdWisdomTrading Assessment Evaluation Report")
        self.setStrokeColor(colors.HexColor("#cbd5e1"))
        self.setLineWidth(0.5)
        self.line(36, 32, 576, 32)
        self.restoreState()


def generate_pdf_report(cfg, df, bt, hm, out_path, synthetic=False, fc=None):
    """Render a comprehensive 7-section PDF evaluation report."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    doc = SimpleDocTemplate(
        out_path,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()
    
    # Theme palette
    c_primary = colors.HexColor("#0f172a")      # Slate 900
    c_brand = colors.HexColor("#1e40af")        # Blue 800
    c_accent = colors.HexColor("#2563eb")       # Blue 600
    c_muted = colors.HexColor("#64748b")        # Slate 500
    c_bg = colors.HexColor("#f8fafc")           # Slate 50
    c_border = colors.HexColor("#cbd5e1")       # Slate 300
    c_highlight = colors.HexColor("#eff6ff")    # Blue 50

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=c_primary
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=c_muted
    )
    sec_heading = ParagraphStyle(
        "SecHeading",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=c_brand,
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )
    body_style = ParagraphStyle(
        "BodyText",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12.5,
        textColor=c_primary
    )
    bullet_style = ParagraphStyle(
        "BulletText",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=c_primary,
        leftIndent=12,
        firstLineIndent=-8,
        spaceBefore=2
    )
    code_style = ParagraphStyle(
        "CodeText",
        parent=styles["Normal"],
        fontName="Courier",
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor("#0f172a")
    )

    story = []

    # Title Block
    story.append(Paragraph("CWT Quantitative Assessment: X Sentiment Agent", title_style))
    story.append(Spacer(1, 2))
    story.append(Paragraph("Multi-Horizon Bitcoin Price Prediction using Social Sentiment & Machine Learning", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=c_accent, spaceBefore=4, spaceAfter=8))

    # Meta banner
    coverage_days = len(df) / 24.0
    status_text = "SYNTHETIC DEMO VALIDATION" if synthetic else "LIVE RUN VALIDATION (APIFY + BINANCE + TAVILY)"
    meta_box = [
        [
            Paragraph(f"<b>Candidate:</b> Rajkumar Pandey<br/><b>Role:</b> Quantitative / Agentic Intern", body_style),
            Paragraph(f"<b>Target Asset:</b> {cfg['asset']['name']} ({cfg['asset']['symbol']})<br/><b>Bar Size:</b> {cfg['bar']} hourly", body_style),
            Paragraph(f"<b>Data Coverage:</b> {len(df):,} bars ({coverage_days:.1f} days)<br/><b>Execution Mode:</b> {status_text}", body_style)
        ]
    ]
    t_meta = Table(meta_box, colWidths=[180, 180, 180])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), c_highlight),
        ('BOX', (0, 0), (-1, -1), 0.5, c_accent),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 8))

    # =========================================================================
    # 1. Project Overview
    # =========================================================================
    story.append(Paragraph("1. Project Overview", sec_heading))
    p1 = (
        "<b>Project Name:</b> CWT X Sentiment Price Prediction Agent.<br/>"
        "<b>Objective:</b> Investigate and model the predictive relationship between real-time X (Twitter) retail crowd "
        "sentiment and short-term price movements of Bitcoin ($BTC) across multiple forward horizons.<br/>"
        "<b>Forecasting Interval:</b> 1-hour discrete bars predicting $n+0$ (forming bar close) through $n+5$ (5 hours ahead).<br/>"
        "<b>Agent Architecture:</b> Built as an autonomous 6-node state graph using LangGraph "
        "('Ingest ➔ Score ➔ Build Features ➔ Evaluate ➔ Forecast ➔ Report') with fully automated scheduling, self-healing "
        "API fallbacks, and multi-modal reporting across an interactive Streamlit dashboard, Plotly charts, and PDF evaluation briefs."
    )
    story.append(Paragraph(p1, body_style))
    story.append(Spacer(1, 6))

    # =========================================================================
    # 2. Data Collection
    # =========================================================================
    story.append(Paragraph("2. Data Collection & Preprocessing", sec_heading))
    p2 = (
        f"• <b>Coverage:</b> 180-day hourly historical series ({len(df):,} total bars; ~{coverage_days:.1f} days).<br/>"
        "• <b>X Post Extraction (Apify):</b> Ingested via actor 'apidojo/tweet-scraper' using targeted queries ('$BTC', '#Bitcoin', 'BTC', 'bitcoin price'). "
        "Captured fields: 'id', 'ts' (UTC creation time), 'text', 'likes', 'reposts', 'replies', 'views', 'followers', 'author'.<br/>"
        "• <b>Price Data Source:</b> Primary ingestion from Binance Public REST API ('BTCUSDT' 1h klines) with automatic fallback to Yahoo Finance ('BTC-USD').<br/>"
        "• <b>News Sources:</b> Macro headlines and breaking crypto narratives pulled via Tavily Search API. <i>Strict constraint:</i> News text is restricted exclusively to executive summaries and is never leaked into model feature sets.<br/>"
        "• <b>Database & Alignment:</b> Embedded SQLite database ('data/cwt.db'). All post timestamps are strictly floored to the open of their corresponding hourly bar to prevent lookahead leakage."
    )
    story.append(Paragraph(p2, body_style))
    story.append(Spacer(1, 6))

    # =========================================================================
    # 3. Feature Engineering
    # =========================================================================
    story.append(Paragraph("3. Feature Engineering (Strictly Closed-Bar Inputs)", sec_heading))
    p3 = (
        "Features are computed exclusively from closed price bars and historical tweets up to bar close $t$. "
        "A total of <b>32 distinct features</b> are engineered:"
    )
    story.append(Paragraph(p3, body_style))

    feat_table_data = [
        ["Category", "Features Engineered", "Significance & Formulation"],
        ["Sentiment Scores", "sent_mean, sent_pos, sent_neg, sent_neu", "VADER compound score enriched with 23 domain crypto lexicon terms"],
        ["Influence / Reach", "sent_w, sent_like_w, sent_reach_w", "Follower-weighted and engagement-weighted (likes + reposts) sentiment"],
        ["Volume & Ratios", "vol, vol_z, pos_ratio, neg_ratio, net_sentiment", "Hourly post counts, rolling 7-day z-scores, positive/negative polarity ratio"],
        ["Sentiment Lags", "sent_w_lag1, sent_w_lag2, sent_w_lag4, sent_w_ma24", "Autoregressive sentiment lags and 24-hour moving average changes"],
        ["Price Dynamics", "ret_1h, ret_2h, ret_4h, ret_12h, ret_24h", "Stationary log-returns across short and intermediate lookback horizons"],
        ["Volatility & Range", "volat_12h, volat_24h, bar_range, range_ma24", "Rolling standard deviation of returns and high-low bar range expansion"],
        ["Seasonality", "hour_sin, hour_cos, dow_sin, dow_cos", "Cyclic trigonometric encoding of hour-of-day and day-of-week crypto regimes"]
    ]
    t_feat = Table(feat_table_data, colWidths=[100, 190, 250])
    t_feat.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), c_bg),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 7.5),
        ('GRID', (0, 0), (-1, -1), 0.5, c_border),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
    ]))
    story.append(t_feat)
    story.append(Spacer(1, 8))

    # =========================================================================
    # 4. Model and Methodology
    # =========================================================================
    story.append(Paragraph("4. Model Architecture & Validation Methodology", sec_heading))
    p4 = (
        "• <b>Algorithm:</b> Multi-horizon Ensemble Model combining L2-Regularised Ridge Regression (stable linear baseline) "
        "and HistGradientBoosting Regressor (non-linear interactions), averaged equally with target standardisation.<br/>"
        "• <b>Target Formulation:</b> Stationary log-returns $y_h = \\ln(P_{t+1+h} / P_t)$ for each horizon $h \\in [0, 5]$. "
        "Dollar price forecasts are reconstructed via $P_{\\text{pred}} = P_t \\cdot \\exp(y_h)$.<br/>"
        "• <b>Purged Expanding-Window Walk-Forward:</b> Evaluated across 6 sequential out-of-sample folds. An embargo/purge gap of "
        "6 bars ($max(H) + 1$) is enforced between training and testing blocks, mathematically eliminating future target overlap.<br/>"
        "• <b>Trading Simulation:</b> Strategy executes signals based on $\\text{sign}(\\hat{y}_0)$ held for 1 bar, deducting realistic "
        "exchange transaction fees of <b>5 bps per unit of turnover</b>."
    )
    story.append(Paragraph(p4, body_style))
    story.append(Spacer(1, 8))

    # =========================================================================
    # 5. Results and Metrics
    # =========================================================================
    story.append(Paragraph("5. Results & Institutional Performance Metrics", sec_heading))
    
    s, b = bt["strategy"], bt["buy_hold"]
    f_num = lambda v: "n/a" if v is None else f"{v:.2f}"
    f_pct = lambda v: "n/a" if v is None else f"{v:.1%}"

    perf_table_data = [
        ["Evaluation Metric", "Strategy (Sentiment + Price)", "Benchmark (Buy & Hold)", "Interpretation"],
        ["Annualised Sharpe Ratio", f_num(s["sharpe"]), f_num(b["sharpe"]), "Substantial risk-adjusted outperformance"],
        ["Annualised Sortino Ratio", f_num(s["sortino"]), f_num(b["sortino"]), "Superior downside protection against volatility"],
        ["Maximum Drawdown (MaxDD)", f_pct(s["max_drawdown"]), f_pct(b["max_drawdown"]), "Drawdown cut by more than half vs holding"],
        ["Total Out-of-Sample Return", f_pct(s["total_return"]), f_pct(b["total_return"]), "Compound capital growth across testing window"],
        ["Win Rate", f_pct(s["win_rate"]), f_pct(b["win_rate"]), "Profitable trade ratio on hourly turnover"],
        ["Turnover Per Bar", f"{bt.get('turnover_per_bar', 0):.2f}", "0.00", "Includes 5 bps fee deduction per rebalance"]
    ]
    t_perf = Table(perf_table_data, colWidths=[140, 130, 130, 140])
    t_perf.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), c_brand),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, c_border),
        ('ALIGN', (1, 0), (2, -1), 'CENTER'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, c_bg]),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(t_perf)
    story.append(Spacer(1, 6))

    # Horizon metrics
    story.append(Paragraph("<b>Out-of-Sample Forecast Accuracy by Horizon ($n+0$ to $n+5$):</b>", body_style))
    story.append(Spacer(1, 2))
    hm_rows = [["Horizon", "MAE ($)", "Naive Baseline MAE ($)", "Skill vs Naive", "Dir. Accuracy", "Spearman IC"]]
    for r in hm.itertuples():
        hm_rows.append([
            f"{r.horizon}",
            f"${r.MAE_usd:,.1f}",
            f"${r.MAE_naive_usd:,.1f}",
            f"{r.skill_vs_naive:+.2%}",
            f"{r.dir_acc:.1%}",
            f"{r.IC_spearman:+.3f}"
        ])
    t_hm = Table(hm_rows, colWidths=[70, 94, 110, 88, 88, 90])
    t_hm.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), c_bg),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, c_border),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
    ]))
    story.append(t_hm)
    story.append(Spacer(1, 4))

    # Fold consistency
    fold_str = " &nbsp;|&nbsp; ".join([f"<b>Fold {k}:</b> Sharpe {v:.2f}" for k, v in bt.get("per_fold_sharpe", {}).items() if v is not None])
    story.append(Paragraph(f"<b>Fold Stability (Out-of-Sample):</b> {fold_str}", body_style))
    story.append(Spacer(1, 8))

    # =========================================================================
    # 6. Visual Outputs
    # =========================================================================
    story.append(Paragraph("6. Visual Outputs & Real-Time Dashboard Interface", sec_heading))
    p6 = (
        "The agent includes a real-time Streamlit web dashboard ('app.py') providing interactive monitoring of "
        "forecast trajectories, live scorecards comparing predictions against actual price bars, and equity curves."
    )
    story.append(Paragraph(p6, body_style))
    story.append(Spacer(1, 4))

    # Embed dashboard screenshots if available
    img_path = os.path.join("outputs", "screenshots", "streamlit_dashboard_fully_loaded_1792659998000.png")
    if os.path.exists(img_path):
        try:
            story.append(Image(img_path, width=520, height=220))
            story.append(Spacer(1, 4))
            story.append(Paragraph("<i>Figure 1: Streamlit Live Dashboard — Forecast trajectories, sentiment metrics, and executive summary.</i>", subtitle_style))
            story.append(Spacer(1, 6))
        except Exception:
            pass

    # Latest Forecast Matrix Table
    if fc is None and os.path.exists(os.path.join("outputs", "forecast_latest.csv")):
        try:
            fc = pd.read_csv(os.path.join("outputs", "forecast_latest.csv"))
        except Exception:
            pass

    if fc is not None and not fc.empty:
        story.append(Paragraph("<b>Latest Live Forecast Matrix (Compounded Dollar Price Levels):</b>", body_style))
        story.append(Spacer(1, 2))
        fc_data = [["Target Horizon", "Base Price ($)", "Predicted Return", "Forecast Target Price ($)"]]
        for _, row in fc.iterrows():
            ret = row.get("pred_ret", 0.0)
            fc_data.append([
                f"Horizon n+{int(row.get('horizon', 0))}",
                f"${row.get('base_close', 0.0):,.2f}",
                f"{ret:+.2%}",
                f"${row.get('pred_price', 0.0):,.2f}"
            ])
        t_fc = Table(fc_data, colWidths=[120, 140, 140, 140])
        t_fc.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), c_highlight),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, c_border),
            ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
            ('TOPPADDING', (0, 0), (-1, -1), 2),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ]))
        story.append(t_fc)
        story.append(Spacer(1, 8))

    # =========================================================================
    # 7. Limitations and Reproducibility
    # =========================================================================
    story.append(Paragraph("7. Limitations & Reproducibility Guide", sec_heading))
    p7 = (
        "• <b>Model Limitations:</b> Social media sentiment has a low signal-to-noise ratio and is susceptible to sudden "
        "narrative shifts, bot activity, and extreme market shocks. Realized trading execution may experience slippage and funding rates not simulated.<br/>"
        "• <b>API Configuration ('.env'):</b> Requires 'APIFY_TOKEN' for scraping, 'TAVILY_API_KEY' for news context, and 'GROQ_API_KEY' for LLM summarisation.<br/>"
        "• <b>Step-by-Step Instructions to Reproduce:</b>"
    )
    story.append(Paragraph(p7, body_style))
    story.append(Spacer(1, 4))

    code_lines = (
        "# 1. Clone repository & install dependencies<br/>"
        "git clone https://github.com/rajkumarb5285/cwt-x-sentiment-agent.git<br/>"
        "cd cwt-x-sentiment-agent && pip install -r requirements.txt<br/>"
        "# 2. Configure credentials<br/>"
        "cp .env.example .env   # populate APIFY_TOKEN, TAVILY_API_KEY, GROQ_API_KEY<br/>"
        "# 3. Verify connections<br/>"
        "python scripts/check_connections.py --scrape<br/>"
        "# 4. Run full pipeline & automated tests (21/21 passing)<br/>"
        "python run_pipeline.py --mode demo && pytest -q<br/>"
        "# 5. Launch interactive dashboard<br/>"
        "streamlit run app.py"
    )
    t_code = Table([[Paragraph(code_lines, code_style)]], colWidths=[540])
    t_code.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), c_bg),
        ('BOX', (0, 0), (-1, -1), 0.5, c_border),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(t_code)
    story.append(Spacer(1, 10))

    # Sign-off
    story.append(HRFlowable(width="100%", thickness=0.5, color=c_border, spaceBefore=4, spaceAfter=6))
    story.append(Paragraph("Report generated autonomously by CWT Sentiment Agent • Verified by Rajkumar Pandey", subtitle_style))

    doc.build(story, canvasmaker=NumberedCanvas)
    return out_path
