import io
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import numpy as np
import pandas as pd
from docx import Document
from docx.shared import Inches, Pt, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.section import WD_ORIENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

AMOUNT_COL = 'Payment Voucher Amount (SGD, Excluding GST)'
BENFORD_EXPECTED = {d: np.log10(1 + 1 / d) for d in range(1, 10)}

# Colour palette (RGB tuples)
NAVY  = RGBColor(0x1F, 0x38, 0x64)
BLUE  = RGBColor(0x2E, 0x75, 0xB6)
GREY  = RGBColor(0x60, 0x60, 0x60)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
RED   = RGBColor(0xC0, 0x00, 0x00)
GREEN = RGBColor(0x70, 0xAD, 0x47)

# Hex colours for matplotlib
_HEX = {
    'navy':   '#1F3864',
    'blue':   '#2E75B6',
    'orange': '#ED7D31',
    'red':    '#C00000',
    'green':  '#70AD47',
}


# ---------------------------------------------------------------------------
# docx helpers
# ---------------------------------------------------------------------------

def _heading(doc, text, level=1):
    p = doc.add_heading(text, level=level)
    p.runs[0].font.color.rgb = NAVY
    return p


def _body(doc, text, bold=False, italic=False, size=10):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    run = p.add_run(text)
    run.font.size = Pt(size + 2)
    run.bold = bold
    run.italic = italic
    run.font.color.rgb = GREY
    return p


def _bullet(doc, text, size=10):
    p = doc.add_paragraph(style='List Bullet')
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    run = p.add_run(text)
    run.font.size = Pt(size + 2)
    run.font.color.rgb = GREY
    return p


def _coloured_para(doc, label, value, colour=NAVY, size=11):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    r1 = p.add_run(label + ": ")
    r1.bold = True
    r1.font.size = Pt(size + 2)
    r1.font.color.rgb = NAVY
    r2 = p.add_run(str(value))
    r2.font.size = Pt(size + 2)
    r2.font.color.rgb = colour
    return p


def _shade_cell(cell, hex_color):
    tc = cell._tc
    tcPr = tc.find(qn('w:tcPr'))
    if tcPr is None:
        tcPr = OxmlElement('w:tcPr')
        tc.insert(0, tcPr)
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), hex_color)
    tcPr.append(shd)


def _remove_table_borders(tbl):
    tbl_element = tbl._tbl
    tblPr = tbl_element.find(qn('w:tblPr'))
    if tblPr is None:
        tblPr = OxmlElement('w:tblPr')
        tbl_element.insert(0, tblPr)
    tbl_borders = OxmlElement('w:tblBorders')
    for border_name in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV'):
        border = OxmlElement(f'w:{border_name}')
        border.set(qn('w:val'), 'none')
        tbl_borders.append(border)
    tblPr.append(tbl_borders)


def _set_landscape(section):
    section.orientation = WD_ORIENT.LANDSCAPE
    w, h = section.page_height, section.page_width
    section.page_width = w
    section.page_height = h
    section.left_margin  = Cm(1.5)
    section.right_margin = Cm(1.5)
    section.top_margin   = Cm(1.5)
    section.bottom_margin = Cm(1.5)


def _insert_image(doc, img_buf, width_inches, centre=False):
    p = doc.add_paragraph()
    if centre:
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    run.add_picture(img_buf, width=Inches(width_inches))
    return p


# ---------------------------------------------------------------------------
# Matplotlib chart builders  (each returns a BytesIO PNG)
# ---------------------------------------------------------------------------

def _to_image(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return buf


def _chart_benford(benford_stats):
    digits   = list(range(1, 10))
    obs_pct  = [benford_stats['observed_pct'].get(d, 0) * 100 for d in digits]
    exp_pct  = [BENFORD_EXPECTED[d] * 100 for d in digits]
    deviant  = benford_stats.get('deviant_digits', set())
    colors   = [_HEX['red'] if d in deviant else _HEX['blue'] for d in digits]

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.bar(digits, obs_pct, color=colors, alpha=0.85, label='Observed')
    ax.plot(digits, exp_pct, 'o--', color=_HEX['orange'], linewidth=2,
            markersize=5, label="Benford's Expected")
    ax.set_xticks(digits)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0f}%'))
    ax.set_ylabel('Frequency (%)')
    mad        = benford_stats.get('mad', 0)
    conformity = benford_stats.get('conformity', '')
    ax.set_xlabel(f'First Digit  |  MAD: {mad:.4f}  |  Verdict: {conformity}', fontsize=8)
    ax.set_title("Benford's Law — First Digit Distribution",
                 fontsize=11, fontweight='bold', color=_HEX['navy'])
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    return _to_image(fig)


def _chart_risk_distribution(df_vouchers, cutoff_score):
    scores = df_vouchers['voucher_score'].dropna()
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.hist(scores, bins=50, color=_HEX['blue'], alpha=0.75, edgecolor='white')
    ax.axvline(cutoff_score, color=_HEX['red'], linestyle='--', linewidth=2,
               label=f'Selection threshold ({cutoff_score:.3f})')
    ax.set_xlabel('Voucher Risk Score')
    ax.set_ylabel('Number of Vouchers')
    ax.set_title('Voucher Risk Score Distribution',
                 fontsize=11, fontweight='bold', color=_HEX['navy'])
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    return _to_image(fig)


def _chart_amount_distribution(df):
    amounts = df[AMOUNT_COL].dropna().clip(lower=0.01)
    fig, ax = plt.subplots(figsize=(11.0, 3.2))
    ax.hist(np.log10(amounts), bins=60, color=_HEX['navy'], alpha=0.80, edgecolor='white')
    tick_vals   = [0, 1, 2, 3, 4, 5, 6]
    tick_labels = ['1', '10', '100', '1K', '10K', '100K', '1M']
    ax.set_xticks(tick_vals)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel('Amount (SGD)')
    ax.set_ylabel('Number of Transactions')
    ax.set_title('Payment Amount Distribution (log scale)',
                 fontsize=11, fontweight='bold', color=_HEX['navy'])
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    return _to_image(fig)


def _chart_timeline(df):
    df2 = df.copy()
    df2['Month'] = df2['Invoice Date'].dt.to_period('M')
    monthly = (
        df2.groupby('Month')[AMOUNT_COL]
        .agg(Total='sum', Count='count')
        .reset_index()
    )
    monthly['Month_str'] = monthly['Month'].astype(str)

    fig, ax1 = plt.subplots(figsize=(11.0, 3.2))
    x = range(len(monthly))
    ax1.bar(list(x), monthly['Total'], color=_HEX['blue'], alpha=0.80,
            label='Total Amount (SGD)')
    ax1.set_ylabel('Total Amount (SGD)', color=_HEX['blue'], fontsize=9)
    ax1.tick_params(axis='y', labelcolor=_HEX['blue'])
    ax1.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f'{v:,.0f}'))

    ax2 = ax1.twinx()
    ax2.plot(list(x), monthly['Count'], 'o-', color=_HEX['orange'],
             linewidth=2, markersize=4, label='Transaction Count')
    ax2.set_ylabel('Transaction Count', color=_HEX['orange'], fontsize=9)
    ax2.tick_params(axis='y', labelcolor=_HEX['orange'])

    ax1.set_xticks(list(x))
    ax1.set_xticklabels(monthly['Month_str'], rotation=45, ha='right', fontsize=7)
    ax1.set_title('Monthly Payment Timeline',
                  fontsize=11, fontweight='bold', color=_HEX['navy'])

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper left')
    ax1.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    return _to_image(fig)


def _chart_top_vendors(df):
    top_count = (
        df.groupby('Vendor Name')[AMOUNT_COL]
        .agg(Count='count', Total='sum')
        .nlargest(10, 'Count')
        .reset_index()
    )
    top_amt = (
        df.groupby('Vendor Name')[AMOUNT_COL]
        .agg(Count='count', Total='sum')
        .nlargest(10, 'Total')
        .reset_index()
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.2))

    y1 = range(len(top_count))
    ax1.barh(list(y1), top_count['Count'], color=_HEX['blue'], alpha=0.85)
    ax1.set_yticks(list(y1))
    ax1.set_yticklabels(top_count['Vendor Name'], fontsize=7)
    ax1.invert_yaxis()
    ax1.set_xlabel('Transaction Count')
    ax1.set_title('Top 10 Vendors by Transaction Count',
                  fontsize=10, fontweight='bold', color=_HEX['navy'])
    ax1.grid(axis='x', alpha=0.3)

    y2 = range(len(top_amt))
    ax2.barh(list(y2), top_amt['Total'], color=_HEX['navy'], alpha=0.85)
    ax2.set_yticks(list(y2))
    ax2.set_yticklabels(top_amt['Vendor Name'], fontsize=7)
    ax2.invert_yaxis()
    ax2.set_xlabel('Total Amount (SGD)')
    ax2.set_title('Top 10 Vendors by Total Amount',
                  fontsize=10, fontweight='bold', color=_HEX['navy'])
    ax2.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f'{v:,.0f}'))
    ax2.grid(axis='x', alpha=0.3)

    fig.tight_layout()
    return _to_image(fig)


# ---------------------------------------------------------------------------
# Page 1 — Executive Summary
# ---------------------------------------------------------------------------

def _page1(doc, df, df_vouchers, selected_vouchers, benford_stats):
    _heading(doc, "Executive Summary", level=1)

    # ---- Dataset overview ----
    _heading(doc, "Dataset Overview", level=2)

    total_lines  = len(df)
    n_vouchers   = len(df_vouchers)
    avg_lines    = total_lines / n_vouchers if n_vouchers > 0 else 0
    total_amt    = df[AMOUNT_COL].sum()
    date_min     = df['Invoice Date'].min()
    date_max     = df['Invoice Date'].max()
    n_vendors    = df['Vendor ID'].nunique()
    n_indiv      = int(df.get('is_individual_payee', pd.Series(0)).sum())
    n_company    = total_lines - n_indiv
    n_recurring  = int(df.get('is_recurring_payment', pd.Series(0)).sum())

    period = (
        f"{date_min.strftime('%d %B %Y')} to {date_max.strftime('%d %B %Y')}"
        if pd.notna(date_min) and pd.notna(date_max) else "N/A"
    )

    stats = [
        ("Analysis period",                    period),
        ("Total transaction line items",       f"{total_lines:,}"),
        ("Unique payment vouchers",            f"{n_vouchers:,}"),
        ("Average lines per voucher",          f"{avg_lines:.1f}"),
        ("Total payments (SGD)",               f"{total_amt:,.2f}"),
        ("Unique vendors",                     f"{n_vendors:,}"),
        ("Payments to individuals",            f"{n_indiv:,} ({n_indiv/total_lines*100:.1f}%)"),
        ("Payments to companies",              f"{n_company:,} ({n_company/total_lines*100:.1f}%)"),
        ("Recurring payments identified",      f"{n_recurring:,} (excluded from Benford's analysis)"),
    ]

    tbl = doc.add_table(rows=len(stats), cols=2)
    tbl.style = 'Table Grid'
    for i, (label, value) in enumerate(stats):
        tbl.rows[i].cells[0].text = label
        tbl.rows[i].cells[1].text = value
        tbl.rows[i].cells[0].paragraphs[0].runs[0].bold = True
        for cell in tbl.rows[i].cells:
            cell.paragraphs[0].runs[0].font.size = Pt(12)
            cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            if i % 2 == 0:
                _shade_cell(cell, "F2F6FC")

    tbl.columns[0].width = Inches(2.5)
    tbl.columns[1].width = Inches(3.5)

    doc.add_paragraph()

    # ---- Summary of findings ----
    _heading(doc, "Summary of Findings", level=2)

    n_benford  = int(df.get('benford_flag',    pd.Series(0)).sum())
    n_if_high  = int(df.get('if_anomaly',     pd.Series(0)).sum())
    n_lof_high = int(df.get('lof_anomaly',    pd.Series(0)).sum())
    n_z_high   = int(df.get('zscore_anomaly', pd.Series(0)).sum())
    n_rule     = int((df.get('rule_flags_score', pd.Series(0)) > 0).sum())

    n_sel_lines = int(selected_vouchers['voucher_line_count'].sum()) \
        if 'voucher_line_count' in selected_vouchers.columns else len(selected_vouchers)
    score_min = selected_vouchers['voucher_score'].min() \
        if 'voucher_score' in selected_vouchers.columns else 0
    score_max = selected_vouchers['voucher_score'].max() \
        if 'voucher_score' in selected_vouchers.columns else 0

    findings = [
        ("Benford's Law deviations flagged",          f"{n_benford:,} transaction lines"),
        ("Isolation Forest anomalies (top 5% boundary)", f"{n_if_high:,} transaction lines"),
        ("Local outlier anomalies (top 5% boundary)",    f"{n_lof_high:,} transaction lines"),
        ("Statistical z-score outliers",              f"{n_z_high:,} transaction lines"),
        ("Rule-based flags triggered",                f"{n_rule:,} transaction lines"),
        ("Final vouchers selected for audit",         f"{len(selected_vouchers)} payment vouchers ({n_sel_lines:,} line items)"),
        ("Voucher risk score range (selected)",        f"{score_min:.3f} – {score_max:.3f}"),
    ]

    for label, value in findings:
        p = doc.add_paragraph(style='List Bullet')
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        r1 = p.add_run(f"{label}: ")
        r1.bold = True
        r1.font.size = Pt(12)
        r2 = p.add_run(value)
        r2.font.size = Pt(12)

    doc.add_paragraph()
    _body(doc,
          f"The {len(selected_vouchers)} payment vouchers selected represent those with the highest "
          "composite risk scores across all analytical methods. Scoring is performed at line-item "
          "level and then rolled up to payment voucher level, so auditors can pull complete vouchers "
          "for review. Each selected voucher has documented reason codes "
          "(see the 'Selected Vouchers' tab in the accompanying Excel workbook).",
          size=10)


# ---------------------------------------------------------------------------
# Page 2 — Methodology
# ---------------------------------------------------------------------------

def _page2(doc):
    _heading(doc, "Methodology — How the Tool Works", level=1)

    _body(doc,
          "This tool identifies payment transactions that are statistically unusual and therefore "
          "warrant audit examination. It applies five independent analytical methods simultaneously, "
          "combines their outputs into a composite risk score per transaction line item, rolls up "
          "results to payment voucher level, assigns risk tiers, and selects a stratified sample. "
          "The full process is documented below to support independent verification or recalibration.",
          size=10)
    doc.add_paragraph()

    # ---- Stage 1 ----
    _heading(doc, "Stage 1 — Feature Engineering", level=2)
    _body(doc,
          "Before scoring, each transaction line is enriched with computed behavioural features: "
          "the payment amount normalised against the vendor's average amount (z-score) and "
          "against the cost centre average; processing days (Invoice Date to Voucher Accounting "
          "Date); whether the invoice date is a non-working day; whether the amount is round; "
          "whether it falls just below a common approval threshold; whether the payee is an "
          "individual (Singapore NRIC/FIN format); and whether the same amount recurs to the same "
          "vendor without a regular schedule.",
          size=10)
    _body(doc,
          "Caveat: Recurring payments (monthly, quarterly, semi-annual, annual cycles) are detected "
          "and tagged separately. They are excluded from Benford's Law analysis because their fixed "
          "amounts naturally deviate from Benford's expected distribution without being suspicious.",
          italic=True, size=10)
    doc.add_paragraph()

    # ---- Stage 2 ----
    _heading(doc, "Stage 2 — Five Independent Analytical Methods", level=2)
    _body(doc,
          "Each transaction line is independently assessed by five methods. Using multiple independent "
          "methods reduces both false positives (legitimate transactions wrongly flagged) and false "
          "negatives (genuine anomalies missed). No single method is relied upon alone.",
          size=10)
    doc.add_paragraph()

    _heading(doc, "1. Benford's Law", level=2)
    _body(doc,
          "In any large collection of naturally occurring financial amounts, approximately 30% start "
          "with digit 1, 17% with 2, 12% with 3, declining to 5% for digit 9. Significant deviation "
          "from this pattern may indicate amounts were manually entered, rounded, or constructed. "
          "Deviation is measured using the Mean Absolute Deviation (MAD) — with Non-Conformity "
          "defined as MAD > 0.015 (Nigrini, 2012) — and a chi-square significance test.",
          size=10)
    _body(doc,
          "Caveat: Benford's Law is most reliable for large datasets (ideally > 1,000 non-recurring "
          "transactions). Smaller datasets or narrow amount ranges produce less stable results. "
          "The chi-square test is very sensitive for large datasets and may flag minor deviations "
          "as statistically significant even when they are not practically meaningful — MAD is the "
          "primary practical measure.",
          italic=True, size=10)
    doc.add_paragraph()

    _heading(doc, "2. Isolation Forest (Machine Learning)", level=2)
    _body(doc,
          "An unsupervised machine learning model that detects anomalies by repeatedly splitting "
          "the data using random rules until each transaction is isolated. Transactions genuinely "
          "different from the rest require fewer splits to isolate — they are unusual in many "
          "dimensions simultaneously. The model evaluates all engineered features together: amount, "
          "processing time, date attributes, payee type, and vendor patterns.",
          size=10)
    _body(doc,
          "Caveat: Being unsupervised, the model identifies outliers relative to the current dataset. "
          "If the dataset contains pervasive irregularities, they may appear normal relative to each "
          "other and not be flagged. The model is most effective when the majority of transactions "
          "are legitimate.",
          italic=True, size=10)
    doc.add_paragraph()

    _heading(doc, "3. Local Outlier Factor — LOF (Machine Learning)", level=2)
    _body(doc,
          "LOF compares each transaction to its nearest neighbours — the most similar transactions "
          "by amount, vendor, and timing. A payment may look ordinary across the full dataset but "
          "be highly anomalous among its vendor peers. For example, a $50,000 payment to a vendor "
          "whose typical invoices are around $2,000 would score very highly even if $50,000 appears "
          "elsewhere in the dataset. This context-sensitivity makes LOF particularly effective for "
          "catching inflated invoices or payments to unusual recipients.",
          size=10)
    _body(doc,
          "Caveat: Same unsupervised limitation as Isolation Forest applies.",
          italic=True, size=10)
    doc.add_paragraph()

    _heading(doc, "4. Statistical Z-Score Analysis", level=2)
    _body(doc,
          "For each vendor and each cost centre, the average payment amount and standard deviation "
          "are computed across all transactions in the dataset. Payments more than 2 standard "
          "deviations above their group average are flagged — a threshold derived from the normal "
          "distribution, where ±2 standard deviations encompasses approximately 95% of values, "
          "leaving the upper 2.5% as statistical outliers. The 2-standard-deviation threshold is a widely "
          "applied convention in quantitative analysis. This approach is consistent with the objective of analytical "
          "procedures which requires auditors to identify and investigate significant fluctuations "
          "or relationships that are inconsistent with other relevant information or that differ "
          "from expected values.",
          size=10)
    _body(doc,
          "Caveat: Auditors should apply professional judgement in assessing whether flagged amounts "
          "are significant in context, noting that payment amounts may follow a skewed rather than "
          "normal distribution, which means the proportion flagged may differ from the theoretical 2.5%.",
          italic=True, size=10)
    doc.add_paragraph()

    _heading(doc, "5. Rule-Based Flags", level=2)
    _body(doc,
          "Six binary rules derived from established forensic audit practice. Each triggers a "
          "flag (1) or not (0) per transaction line:",
          size=10)
    rules = [
        "Round number — amount divisible by 100. Round number amounts may warrant attention as "
        "fabricated or manually chosen amounts sometimes exhibit round number bias, where "
        "individuals select psychologically convenient figures rather than amounts arising from "
        "genuine invoices (Nigrini, 2012; ACFE Fraud Examiners Manual).",
        "Non-working day — invoice dated on a Saturday, Sunday, or Singapore public holiday. "
        "Payments outside business hours may bypass the normal multi-person review process.",
        "Month-end — invoice in the last 3 calendar days of the month. May indicate rushed "
        "processing to meet budget targets or period-end financial reporting cut-offs.",
        "Near approval threshold — within 5% below SGD 1K / 5K / 10K / 50K / 100K. Known as "
        "'structuring' in forensic accounting — deliberately keeping amounts below authorisation "
        "thresholds to avoid triggering higher-level approval.",
        "Individual payee — Vendor ID matches the Singapore NRIC/FIN format (one letter, 7 digits, "
        "one letter). Payments to individuals carry higher inherent risk as they bypass standard "
        "vendor registration and procurement controls.",
        "Irregular repeated amount — same amount paid to the same vendor more than twice with no "
        "detected regular monthly/quarterly/annual schedule. May indicate split or duplicate payments.",
    ]
    for rule in rules:
        _bullet(doc, rule, size=10)
    doc.add_paragraph()

    # ---- Stage 3 — Scoring formulas ----
    _heading(doc, "Stage 3 — Scoring Formulas and Weight Rationale", level=2)

    _body(doc, "Line-Level Composite Risk Score", bold=True, size=10)
    _body(doc,
          "Each of the five methods produces a score between 0 and 1, where 0 means most normal and 1 means most anomalous. "
          "These are combined into a single risk score using fixed weights:",
          size=10)
    p = _body(doc,
              "    risk_score  =  0.30 × IF  +  0.25 × LOF  +  0.25 × Z-score"
              "  +  0.15 × rule_flags  +  0.05 × Benford",
              bold=True, size=8)
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    doc.add_paragraph()
    _body(doc,
          "The rule flags score is the fraction of the 6 binary rules triggered for that line "
          "(e.g. 2 rules triggered = 2/6 = 0.33). The Benford score is normalised relative to "
          "the maximum Benford deviation in the dataset. The Z-score signal is the larger of the "
          "vendor z-score and cost centre z-score, min-max normalised to [0, 1] across all lines.",
          size=10)
    doc.add_paragraph()

    _body(doc, "Weight Rationale", bold=True, size=10)
    doc.add_paragraph()

    weight_rows = [
        ("Isolation Forest", "30%",
         "Primary ML signal; highest weight because it evaluates all features simultaneously "
         "and captures complex multi-dimensional patterns invisible to individual rules or "
         "statistics alone."),
        ("Local Outlier Factor", "25%",
         "Context-sensitive complement to Isolation Forest. Peer-group benchmarking reduces false "
         "positives by comparing each transaction to its most similar counterparts rather than "
         "the full dataset."),
        ("Z-Score Analysis", "25%",
         "Transparent and directly auditable. "
         "Higher weight because it is statistically rigorous and independently defensible."),
        ("Rule-Based Flags", "15%",
         "Directly encodes established forensic audit heuristics. Lower weight because rules are "
         "binary (on/off) and each has known limitations; their primary value is confirming and "
         "explaining signals raised by the other methods."),
        ("Benford's Law", "5%",
         "Supplementary signal only. Powerful at the dataset level but noisy at the individual "
         "transaction level. Low weight prevents Benford deviation alone from driving selection. "
         "Further suppressed when all other signals are below average (see rule below)."),
    ]

    tbl = doc.add_table(rows=1 + len(weight_rows), cols=3)
    tbl.style = 'Table Grid'
    col_widths_wt = [Inches(1.5), Inches(0.6), Inches(4.1)]
    hdr = tbl.rows[0]
    for i, label in enumerate(["Method", "Weight", "Basis for Weight Assignment"]):
        cell = hdr.cells[i]
        cell.text = label
        cell.paragraphs[0].runs[0].bold = True
        cell.paragraphs[0].runs[0].font.size = Pt(11)
        cell.paragraphs[0].runs[0].font.color.rgb = WHITE
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        _shade_cell(cell, "1F3864")
        cell.width = col_widths_wt[i]

    for row_idx, (method, weight, rationale) in enumerate(weight_rows, start=1):
        row = tbl.rows[row_idx]
        shade = "F2F6FC" if row_idx % 2 == 0 else "FFFFFF"
        for col_idx, (value, width) in enumerate(zip([method, weight, rationale], col_widths_wt)):
            cell = row.cells[col_idx]
            cell.text = value
            cell.paragraphs[0].runs[0].font.size = Pt(11)
            cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            _shade_cell(cell, shade)
            cell.width = width

    doc.add_paragraph()

    _body(doc, "Benford Suppression Rule", bold=True, size=10)
    _body(doc,
          "If a transaction's Isolation Forest, LOF, Z-score, and rule flags scores are ALL below "
          "their respective dataset medians — meaning the transaction shows no elevated risk on any "
          "other signal — its Benford contribution is zeroed out entirely. This prevents Benford "
          "deviation alone from selecting a transaction. Benford evidence is only counted when at "
          "least one other signal is also elevated.",
          size=10)
    doc.add_paragraph()

    # ---- Stage 4 — Voucher rollup ----
    _heading(doc, "Stage 4 — Voucher-Level Rollup", level=2)
    _body(doc,
          "Individual scored lines are grouped by Voucher ID — the document auditors physically "
          "pull — rather than by invoice number. The voucher score formula is:",
          size=10)
    p = _body(doc,
              "    voucher_score (multi-line)  =  0.60 × max_line_score"
              "  +  0.25 × mean_line_score  +  0.15 × flag_density",
              bold=True, size=8)
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p = _body(doc,
              "    voucher_score (single-line)  =  line risk_score  (no rollup needed)",
              bold=True, size=8)
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    doc.add_paragraph()
    _body(doc,
          "Flag density = total rule flags triggered across all lines in the voucher ÷ "
          "(6 flag types × number of lines). The 60/25/15 split reflects that audit significance "
          "is primarily driven by the worst line in the voucher, moderated by whether other lines "
          "are also elevated, and supplemented by the breadth of rule flag coverage. For multi-line "
          "vouchers, reason codes in the output are prefixed with [Account Code] so auditors can "
          "identify exactly which line triggered each flag.",
          size=10)
    doc.add_paragraph()

    # ---- ML Consensus ----
    _heading(doc, "ML Consensus Flag", level=2)
    _body(doc,
          "Each transaction line receives an ML Consensus count: the number of the three ML-based "
          "methods that independently classify that line as anomalous using each model's own "
          "boundary. Isolation Forest and LOF use sklearn's predict() method at "
          "contamination=0.05, which flags the top 5% of lines as anomalous per model. "
          "Z-score flags lines where the maximum absolute z-score exceeds 2.0 (2 standard "
          "deviations). A voucher is marked 'ML Consensus = Yes' if any of its lines is "
          "classified as anomalous by 2 or more of the three methods simultaneously.",
          size=10)
    _body(doc,
          "The ML Consensus flag does not alter the composite score — it is a corroborating "
          "indicator shown in the Excel output. When multiple independent methods agree that a "
          "transaction is anomalous, the probability of a true anomaly is materially higher than "
          "when only one method flags it.",
          size=10)
    doc.add_paragraph()

    # ---- Risk tiers and selection ----
    _heading(doc, "Risk Tier Assignment and Sample Selection", level=2)
    _body(doc,
          "After all voucher scores are computed, tiers are assigned based on where each voucher ranks within the dataset: "
          "the top 5% of scores are flagged HIGH, the next 15% MEDIUM, and the remaining 80% LOW. "
          "Percentile-based tiers ensure the tool adapts to any dataset — HIGH always covers the "
          "most anomalous 5% regardless of absolute score values, which vary by dataset size and "
          "composition.",
          size=10)
    doc.add_paragraph()
    _body(doc,
          "The final sample is drawn from all three risk tiers. All HIGH vouchers are selected first. "
          "The remaining slots are filled proportionally from MEDIUM and LOW tiers, "
          "ensuring the highest-risk vouchers are always covered.",
          size=10)
    doc.add_paragraph()

    # ---- Caveats ----
    _heading(doc, "Important Caveats", level=2)
    _body(doc,
          "The following limitations should be understood before acting on the tool's output:",
          size=10)
    caveats = [
        "Risk prioritisation, not fraud evidence — a high voucher score indicates a statistically "
        "unusual transaction that warrants examination. It does not constitute evidence of fraud "
        "or error. All selected vouchers require professional judgement to assess.",
        "Line-item scope — the tool scores individual transaction lines, not total voucher amounts. "
        "A large voucher split across many small lines of normal individual amounts may not score "
        "highly even if the total is anomalous. Auditors should review total voucher values "
        "alongside individual line scores.",
        "Unsupervised models — Isolation Forest and LOF identify outliers relative to the dataset "
        "provided. If the dataset contains pervasive irregularities, both models may treat them "
        "as normal because they resemble the majority. They are most effective when most "
        "transactions are legitimate.",
        "Benford reliability — the analysis is most meaningful for several hundred or more "
        "non-recurring transactions. Small datasets or datasets with narrow amount ranges "
        "produce less reliable Benford results.",
        "Pre-calibrated weights — component weights and rule thresholds are calibrated for typical "
        "corporate payment datasets. Unusual compositions (e.g. predominantly recurring payments, "
        "narrow amount bands) may require recalibration. Weights can be overridden by setting "
        "'sample_selector.WEIGHTS' before calling select_samples().",
        "Not a fraud detection tool — the tool has not been trained on confirmed fraud cases "
        "from this organisation. It identifies unusual patterns by learning the normal behaviour "
        "of the organisation's data. Real-world performance depends on the nature and prevalence "
        "of anomalies present in the data. Transactions not flagged by the tool should not be "
        "interpreted as confirmation that they are free from irregularities, as sophisticated "
        "anomalies that closely mimic normal payment patterns may not be detected. Auditors "
        "should not rely on the tool to detect fraud but should exercise professional judgement "
        "in investigating unusual transactions identified.",
        "Declared component weights are approximate — the five component weights describe the "
        "intended relative importance of each analytical method, not precisely isolated "
        "statistical contributions. Features shared across components — amount z-scores and "
        "rule flags appear both in their dedicated scoring components and as inputs to Isolation "
        "Forest and LOF — carry marginally more effective influence than their labelled "
        "percentage alone suggests. This does not affect the tool's output in practice: the "
        "tool produces a relative ranking of vouchers within the dataset, and transactions that "
        "are genuinely anomalous across multiple dimensions will consistently rank above those "
        "that are not, regardless of the precise effective weight of any individual feature.",
    ]
    for caveat in caveats:
        _bullet(doc, caveat, size=10)


# ---------------------------------------------------------------------------
# Page 3 — Benford's Law + Risk Distribution (landscape, side by side)
# ---------------------------------------------------------------------------

def _page3_charts(doc, df_vouchers, selected_vouchers, benford_stats):
    section = doc.add_section()
    _set_landscape(section)

    _heading(doc, "Analytical Charts", level=1)
    _body(doc,
          "Left: Benford's Law first-digit distribution (red bars = deviant digits). "
          "Right: voucher risk score distribution with the sample selection threshold.",
          size=9)
    doc.add_paragraph()

    cutoff = float(selected_vouchers['voucher_score'].min()) \
        if 'voucher_score' in selected_vouchers.columns else 0.0
    img_benford = _chart_benford(benford_stats)
    img_risk    = _chart_risk_distribution(df_vouchers, cutoff)

    tbl = doc.add_table(rows=1, cols=2)
    _remove_table_borders(tbl)
    for col_idx, img in enumerate([img_benford, img_risk]):
        cell = tbl.cell(0, col_idx)
        p    = cell.paragraphs[0]
        run  = p.add_run()
        run.add_picture(img, width=Inches(4.9))


# ---------------------------------------------------------------------------
# Page 4 — Amount Distribution + Monthly Timeline (landscape, stacked)
# ---------------------------------------------------------------------------

def _page4_distributions(doc, df):
    section = doc.add_section()
    _set_landscape(section)

    _heading(doc, "Payment Distribution & Timeline", level=1)

    _heading(doc, "Payment Amount Distribution", level=2)
    _body(doc, "Histogram of payment amounts on a log scale across all transactions.", size=9)
    _insert_image(doc, _chart_amount_distribution(df), width_inches=10.0, centre=True)

    doc.add_paragraph()

    _heading(doc, "Monthly Payment Timeline", level=2)
    _body(doc,
          "Monthly total payment value (bars, left axis) and transaction count "
          "(line, right axis) over the analysis period.",
          size=9)
    _insert_image(doc, _chart_timeline(df), width_inches=10.0, centre=True)


# ---------------------------------------------------------------------------
# Page 5 — Top Vendors (landscape)
# ---------------------------------------------------------------------------

def _page5_vendors(doc, df):
    section = doc.add_section()
    _set_landscape(section)

    _heading(doc, "Vendor Analysis", level=1)
    _body(doc,
          "Top 10 vendors ranked by transaction count (left) and by total payment value (right).",
          size=9)
    doc.add_paragraph()
    _insert_image(doc, _chart_top_vendors(df), width_inches=10.0, centre=True)


# ---------------------------------------------------------------------------
# Page 6 — Feature Reference Table (landscape)
# ---------------------------------------------------------------------------

# Columns: Feature | What It Measures | Threshold for Flagging | ML Models | Why It Matters
# ML Models shows which of the three ML scoring components each feature feeds into:
#   IF  = Isolation Forest   LOF = Local Outlier Factor   Z   = Statistical Z-Score
ML_FEATURE_TABLE_DATA = [
    (
        "Amount vs. vendor average",
        "How much the payment amount differs from what this vendor is typically paid.",
        "Z-score > 2.0",
        "IF, LOF, Z-score",
        "Unusually large payments to a vendor may indicate over-billing or fictitious invoices.",
    ),
    (
        "Amount vs. cost centre average",
        "How much the payment amount differs from the typical amounts processed in that cost centre.",
        "Z-score > 2.0",
        "IF, LOF, Z-score",
        "Helps detect amounts that are out of place for the department, suggesting possible miscoding or inflated claims.",
    ),
    (
        "Round number",
        "Whether the payment amount ends in 00, 000, or 0,000.",
        "Exactly divisible by 100",
        "IF, LOF",
        "Genuine invoice amounts rarely end in round numbers; manually chosen or fictitious amounts often do.",
    ),
    (
        "Non-working day",
        "Whether the invoice is dated on a Saturday, Sunday, or Singapore public holiday.",
        "Sat, Sun, or SG public holiday (holidays library)",
        "IF, LOF",
        "Payments authorised outside business hours may bypass the normal multi-person review and approval process.",
    ),
    (
        "Month-end",
        "Whether the invoice is dated in the last 3 calendar days of the month.",
        "Last 3 calendar days of month",
        "IF, LOF",
        "May indicate rushed processing to meet budget targets or period-end financial reporting cut-offs.",
    ),
    (
        "Near approval threshold",
        "Whether the amount falls within 5% below a common approval limit",
        "Within 5% below SGD 1K / 5K / 10K / 50K / 100K",
        "IF, LOF",
        "A known technique ('structuring') to avoid triggering higher-level approval requirements.",
    ),
    (
        "Individual payee",
        "Whether the Vendor ID matches the Singapore NRIC/FIN format (one letter, 7 digits, one letter).",
        "Regex: ^[A-Za-z][0-9]{7}[A-Za-z]$",
        "IF, LOF",
        "Payments to individuals carry higher inherent risk versus registered businesses.",
    ),
    (
        "Processing time",
        "Number of calendar days between Invoice Date and Voucher Accounting Date.",
        "Absolute z-score > 2.5",
        "IF, LOF",
        "Very fast processing may indicate bypassed controls; unusually long delays may indicate backdating.",
    ),
    (
        "Description length",
        "Character length of the Voucher Line Description field.",
        "Absolute z-score > 2.5",
        "IF, LOF",
        "Very short descriptions may indicate incomplete entries; very long ones may indicate unusual or fabricated narrative.",
    ),
    (
        "Irregular repeated amount",
        "Same vendor paid the same amount more than twice, with no regular monthly/quarterly/annual schedule.",
        "> 2 occurrences with no detected recurring cycle",
        "IF, LOF",
        "May indicate duplicated or split payments that were structured to avoid detection.",
    ),
]

# Features that contribute to scoring but do not feed into any ML model.
BENFORD_FEATURE_TABLE_DATA = [
    (
        "Benford's Law first digit",
        "Whether the payment amount's first digit deviates significantly from Benford's expected frequency.",
        "First digit among the top-3 most deviant digits; non-recurring payments only.",
        "None — Benford's Law analysis only (5% of composite score)",
        "Systematic deviation may indicate manually constructed or manipulated amounts.",
    ),
]


def _render_feature_table(doc, data, col_widths):
    """Render a 5-column feature table (Feature | Measures | Threshold | ML Models | Why)."""
    headers = ["Feature", "What It Measures", "Threshold for Flagging", "ML Models", "Why It Matters"]
    tbl = doc.add_table(rows=1 + len(data), cols=5)
    tbl.style = 'Table Grid'

    hdr = tbl.rows[0]
    for i, (header, width) in enumerate(zip(headers, col_widths)):
        cell = hdr.cells[i]
        cell.text = header
        cell.paragraphs[0].runs[0].bold = True
        cell.paragraphs[0].runs[0].font.size = Pt(10)
        cell.paragraphs[0].runs[0].font.color.rgb = WHITE
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        _shade_cell(cell, "1F3864")
        cell.width = width

    for row_idx, row_data in enumerate(data, start=1):
        row   = tbl.rows[row_idx]
        shade = "F2F6FC" if row_idx % 2 == 0 else "FFFFFF"
        for col_idx, (value, width) in enumerate(zip(row_data, col_widths)):
            cell = row.cells[col_idx]
            cell.text = value
            cell.paragraphs[0].runs[0].font.size = Pt(9.5)
            cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            _shade_cell(cell, shade)
            cell.width = width


def _page6_feature_table(doc):
    section = doc.add_section()
    _set_landscape(section)

    _heading(doc, "Feature Reference Table", level=1)
    _body(doc,
          "The tables below list each analytical feature, the threshold that determines whether a "
          "transaction is flagged, which ML scoring models the feature feeds into, and the audit "
          "rationale. ML Models: IF = Isolation Forest, LOF = Local Outlier Factor, "
          "Z-score = Statistical Z-Score Analysis.",
          size=9)
    doc.add_paragraph()

    col_widths = [Inches(1.6), Inches(2.1), Inches(1.8), Inches(1.2), Inches(3.8)]

    _heading(doc, "Features Used in Machine Learning Models", level=2)
    _body(doc,
          "All ten features below are normalised via RobustScaler and fed into the ML models "
          "before scoring. Amount z-scores additionally drive the Statistical Z-Score component "
          "directly.",
          size=9)
    doc.add_paragraph()
    _render_feature_table(doc, ML_FEATURE_TABLE_DATA, col_widths)

    doc.add_paragraph()

    _heading(doc, "Features Outside Machine Learning Models", level=2)
    _body(doc,
          "The feature below is computed by an independent method and contributes 5% of the "
          "composite risk score separately from the ML models.",
          size=9)
    doc.add_paragraph()
    _render_feature_table(doc, BENFORD_FEATURE_TABLE_DATA, col_widths)

    doc.add_paragraph()
    _body(doc,
          "References: Nigrini, M.J. (2012). Benford's Law: Applications for Forensic Accounting, "
          "Auditing, and Fraud Detection. ACFE Fraud Examiners Manual (current edition).",
          italic=True, size=7.5)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def export_word_report(df, df_vouchers, selected_vouchers, benford_stats, output_path):
    print("  Building Word report...")
    doc = Document()

    section0 = doc.sections[0]
    section0.page_width    = Cm(21.0)
    section0.page_height   = Cm(29.7)
    section0.left_margin   = Cm(2.5)
    section0.right_margin  = Cm(2.5)
    section0.top_margin    = Cm(2.5)
    section0.bottom_margin = Cm(2.5)

    style = doc.styles['Normal']
    style.font.size = Pt(12)
    style.font.name = 'Calibri'

    print("    Page 1 — Executive Summary")
    _page1(doc, df, df_vouchers, selected_vouchers, benford_stats)
    doc.add_page_break()

    print("    Page 2 — Methodology")
    _page2(doc)

    print("    Page 3 — Analytical Charts")
    _page3_charts(doc, df_vouchers, selected_vouchers, benford_stats)

    print("    Page 4 — Payment Distribution & Timeline")
    _page4_distributions(doc, df)

    print("    Page 5 — Vendor Analysis")
    _page5_vendors(doc, df)

    print("    Page 6 — Feature Reference Table")
    _page6_feature_table(doc)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    doc.save(output_path)
    print(f"  Word report saved: {output_path}")
