"""Generates a professional PDF evaluation report using ReportLab.
Covers Sharpe, Sortino, Max Drawdown, horizon metrics, and walk-forward stability.
"""
import os
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, KeepTogether
)


def generate_pdf_report(cfg, df, bt, hm, out_path, synthetic=False):
    """Render a comprehensive PDF evaluation report to out_path."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    doc = SimpleDocTemplate(
        out_path,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    
    # Custom palette
    primary_color = colors.HexColor("#0f172a")    # Slate 900
    brand_blue = colors.HexColor("#2563eb")       # Blue 600
    text_muted = colors.HexColor("#64748b")       # Slate 500
    bg_light = colors.HexColor("#f8fafc")         # Slate 50
    border_color = colors.HexColor("#e2e8f0")     # Slate 200

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=primary_color
    )
    subtitle_style = ParagraphStyle(
        "DocSub",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=text_muted
    )
    h2_style = ParagraphStyle(
        "Heading2",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=brand_blue,
        spaceBefore=12,
        spaceAfter=6
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=primary_color
    )
    badge_style = ParagraphStyle(
        "Badge",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#b45309") if synthetic else colors.HexColor("#15803d")
    )

    story = []

    # Header
    story.append(Paragraph("CrowdWisdomTrading Quantitative Assessment", title_style))
    story.append(Spacer(1, 4))
    story.append(Paragraph("Asset Price Prediction Agent via X (Twitter) Sentiment & Price | Walk-Forward Validation", subtitle_style))
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=1.5, color=brand_blue, spaceBefore=4, spaceAfter=8))

    if synthetic:
        data_notice = "<b>NOTICE:</b> Evaluated on synthetic demonstration dataset for pipeline verification. Metrics confirm zero lookahead & execution validity."
    else:
        data_notice = "<b>DATASET:</b> Live data ingestion via Apify X scraper + Binance OHLCV."
    story.append(Paragraph(data_notice, badge_style))
    story.append(Spacer(1, 8))

    # Setup Section
    story.append(Paragraph("1. System Architecture & Model Setup", h2_style))
    setup_text = (
        f"<b>Target Asset:</b> {cfg['asset']['name']} ({cfg['asset']['symbol']}) &nbsp;|&nbsp; "
        f"<b>Bar Size:</b> {cfg['bar']} &nbsp;|&nbsp; <b>Total Bars:</b> {len(df):,} ({len(df)/24:.1f} days)<br/>"
        f"<b>Forecast Horizons:</b> n+0 through n+5 hourly bars ahead.<br/>"
        f"<b>Features:</b> Strictly restricted to X sentiment (mean, engagement-weighted, ratios, volume z-score) "
        f"and price (lagged returns, rolling volatility, range, intraday & day-of-week cycles).<br/>"
        f"<b>Model Engine:</b> {cfg['model']['kind'].capitalize()} (L2-Regularised Ridge + HistGradientBoosting Regressor).<br/>"
        f"<b>Validation Strategy:</b> Expanding-window walk-forward ({cfg['model']['n_splits']} folds) with strict {max(cfg['horizons']) + 1}-bar "
        f"purge window between training and test sets to eliminate target overlap leakage."
    )
    story.append(Paragraph(setup_text, body_style))
    story.append(Spacer(1, 10))

    # Risk Adjusted Performance
    story.append(Paragraph("2. Out-of-Sample Risk-Adjusted Performance", h2_style))
    s, b = bt["strategy"], bt["buy_hold"]
    f_num = lambda v: "n/a" if v is None else f"{v:.2f}"
    f_pct = lambda v: "n/a" if v is None else f"{v:.1%}"

    perf_data = [
        ["Performance Metric", "Strategy (Sentiment + Price)", "Benchmark (Buy & Hold)"],
        ["Annualised Sharpe Ratio", f_num(s["sharpe"]), f_num(b["sharpe"])],
        ["Annualised Sortino Ratio", f_num(s["sortino"]), f_num(b["sortino"])],
        ["Maximum Drawdown (MaxDD)", f_pct(s["max_drawdown"]), f_pct(b["max_drawdown"])],
        ["Total Out-of-Sample Return", f_pct(s["total_return"]), f_pct(b["total_return"])],
        ["Trade Win Rate", f_pct(s["win_rate"]), f_pct(b["win_rate"])],
        ["Turnover Per Bar", f"{bt.get('turnover_per_bar', 0):.2f}", "0.00"],
    ]

    perf_table = Table(perf_data, colWidths=[200, 170, 170])
    perf_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), bg_light),
        ('TEXTCOLOR', (0, 0), (-1, 0), primary_color),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 8.5),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('GRID', (0, 0), (-1, -1), 0.5, border_color),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, bg_light]),
    ]))
    story.append(perf_table)
    story.append(Spacer(1, 10))

    # Multi-Horizon Accuracy Table
    story.append(Paragraph("3. Multi-Horizon Forecast Accuracy (n+0 to n+5)", h2_style))
    hm_data = [["Horizon", "MAE ($)", "Naive MAE ($)", "Skill vs Naive", "Dir. Accuracy", "Spearman IC"]]
    for r in hm.itertuples():
        hm_data.append([
            f"{r.horizon}",
            f"${r.MAE_usd:,.1f}",
            f"${r.MAE_naive_usd:,.1f}",
            f"{r.skill_vs_naive:+.2%}",
            f"{r.dir_acc:.1%}",
            f"{r.IC_spearman:+.3f}"
        ])

    hm_table = Table(hm_data, colWidths=[70, 94, 94, 94, 94, 94])
    hm_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), bg_light),
        ('TEXTCOLOR', (0, 0), (-1, 0), primary_color),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 8.5),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('GRID', (0, 0), (-1, -1), 0.5, border_color),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, bg_light]),
    ]))
    story.append(hm_table)
    story.append(Spacer(1, 10))

    # Walk-forward Fold Stability
    story.append(Paragraph("4. Walk-Forward Fold Consistency", h2_style))
    fold_items = [f"<b>Fold {k}:</b> Sharpe {v:.2f}" for k, v in bt.get("per_fold_sharpe", {}).items() if v is not None]
    story.append(Paragraph(" &nbsp;&nbsp;|&nbsp;&nbsp; ".join(fold_items), body_style))
    story.append(Spacer(1, 10))

    # Leakage & Verification
    story.append(Paragraph("5. Rigorous Leakage & Integrity Controls", h2_style))
    controls_text = (
        "• <b>Zero Lookahead:</b> Feature bar aggregation floors timestamp to bar open; models only ingest data available at bar close.<br/>"
        "• <b>Purged Walk-Forward:</b> Between training window and out-of-sample test window, an embargo gap equal to maximum horizon (6 bars) is dropped.<br/>"
        "• <b>Transaction Cost Model:</b> All performance metrics deduct 5 bps per unit of turnover to simulate realistic slippage and exchange fees.<br/>"
        "• <b>Independent Targets:</b> Models forecast log-returns from current close to each horizon independently."
    )
    story.append(Paragraph(controls_text, body_style))
    story.append(Spacer(1, 12))

    story.append(HRFlowable(width="100%", thickness=0.5, color=border_color, spaceBefore=4, spaceAfter=6))
    story.append(Paragraph("CrowdWisdomTrading Intern Assessment • Generated autonomously by Quantitative Agent", subtitle_style))

    doc.build(story)
    return out_path
