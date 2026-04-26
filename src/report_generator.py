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
    run = p.add_run(text)
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    run.font.color.rgb = GREY
    return p


def _bullet(doc, text, size=10):
    p = doc.add_paragraph(style='List Bullet')
    run = p.add_run(text)
    run.font.size = Pt(size)
    run.font.color.rgb = GREY
    return p


def _coloured_para(doc, label, value, colour=NAVY, size=11):
    p = doc.add_paragraph()
    r1 = p.add_run(label + ": ")
    r1.bold = True
    r1.font.size = Pt(size)
    r1.font.color.rgb = NAVY
    r2 = p.add_run(str(value))
    r2.font.size = Pt(size)
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


def _chart_risk_distribution(df_invoices, cutoff_score):
    scores = df_invoices['invoice_score'].dropna()
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.hist(scores, bins=50, color=_HEX['blue'], alpha=0.75, edgecolor='white')
    ax.axvline(cutoff_score, color=_HEX['red'], linestyle='--', linewidth=2,
               label=f'Selection threshold ({cutoff_score:.3f})')
    ax.set_xlabel('Invoice Risk Score')
    ax.set_ylabel('Number of Invoices')
    ax.set_title('Invoice Risk Score Distribution',
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

def _page1(doc, df, df_invoices, selected_invoices, benford_stats):
    _heading(doc, "Executive Summary", level=1)

    # ---- Dataset overview ----
    _heading(doc, "Dataset Overview", level=2)

    total_lines  = len(df)
    n_invoices   = len(df_invoices)
    avg_lines    = total_lines / n_invoices if n_invoices > 0 else 0
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
        ("Analysis period",                 period),
        ("Total transaction line items",    f"{total_lines:,}"),
        ("Unique invoices identified",      f"{n_invoices:,}"),
        ("Average lines per invoice",       f"{avg_lines:.1f}"),
        ("Total payments (SGD)",            f"{total_amt:,.2f}"),
        ("Unique vendors",                  f"{n_vendors:,}"),
        ("Payments to individuals",         f"{n_indiv:,} ({n_indiv/total_lines*100:.1f}%)"),
        ("Payments to companies",           f"{n_company:,} ({n_company/total_lines*100:.1f}%)"),
        ("Recurring payments identified",   f"{n_recurring:,} (excluded from Benford's analysis)"),
    ]

    tbl = doc.add_table(rows=len(stats), cols=2)
    tbl.style = 'Table Grid'
    for i, (label, value) in enumerate(stats):
        tbl.rows[i].cells[0].text = label
        tbl.rows[i].cells[1].text = value
        tbl.rows[i].cells[0].paragraphs[0].runs[0].bold = True
        for cell in tbl.rows[i].cells:
            cell.paragraphs[0].runs[0].font.size = Pt(10)
            if i % 2 == 0:
                _shade_cell(cell, "F2F6FC")

    tbl.columns[0].width = Inches(2.5)
    tbl.columns[1].width = Inches(3.5)

    doc.add_paragraph()

    # ---- Summary of findings ----
    _heading(doc, "Summary of Findings", level=2)

    n_benford  = int(df.get('benford_flag', pd.Series(0)).sum())
    n_if_high  = int((df.get('if_score',      pd.Series(0)) > 0.65).sum())
    n_lof_high = int((df.get('lof_score',     pd.Series(0)) > 0.65).sum())
    n_z_high   = int((df.get('zscore_score',  pd.Series(0)) > 0.65).sum())
    n_rule     = int((df.get('rule_flags_score', pd.Series(0)) > 0).sum())

    n_sel_lines = int(selected_invoices['invoice_line_count'].sum()) \
        if 'invoice_line_count' in selected_invoices.columns else len(selected_invoices)
    score_min = selected_invoices['invoice_score'].min() \
        if 'invoice_score' in selected_invoices.columns else 0
    score_max = selected_invoices['invoice_score'].max() \
        if 'invoice_score' in selected_invoices.columns else 0

    findings = [
        ("Benford's Law deviations flagged",          f"{n_benford:,} transaction lines"),
        ("Isolation Forest anomalies (score > 0.65)", f"{n_if_high:,} transaction lines"),
        ("Local outlier anomalies (score > 0.65)",    f"{n_lof_high:,} transaction lines"),
        ("Statistical z-score outliers",              f"{n_z_high:,} transaction lines"),
        ("Rule-based flags triggered",                f"{n_rule:,} transaction lines"),
        ("Final invoices selected for audit",         f"{len(selected_invoices)} invoices ({n_sel_lines:,} line items)"),
        ("Invoice risk score range (selected)",       f"{score_min:.3f} – {score_max:.3f}"),
    ]

    for label, value in findings:
        p = doc.add_paragraph(style='List Bullet')
        r1 = p.add_run(f"{label}: ")
        r1.bold = True
        r1.font.size = Pt(10)
        r2 = p.add_run(value)
        r2.font.size = Pt(10)

    doc.add_paragraph()
    _body(doc,
          f"The {len(selected_invoices)} invoices selected represent those with the highest composite "
          "risk scores across all analytical methods. Scoring is performed at line-item level and "
          "then rolled up to invoice level, so auditors can pull complete invoices rather than "
          "individual lines. Each selected invoice has documented reason codes "
          "(see the 'Selected Invoices' tab in the accompanying Excel workbook).",
          size=10)


# ---------------------------------------------------------------------------
# Page 2 — Methodology
# ---------------------------------------------------------------------------

def _page2(doc):
    _heading(doc, "Methodology — How the Tool Works", level=1)

    _body(doc,
          "This tool uses a combination of established statistical tests and machine learning "
          "algorithms to identify payment transactions that are unusual and therefore warrant "
          "further review. Each method is described below in plain terms.",
          size=10)
    doc.add_paragraph()

    _heading(doc, "1. Benford's Law Analysis", level=2)
    _body(doc,
          "In any large collection of naturally occurring financial amounts — such as supplier "
          "invoices or expense claims — approximately 30% of amounts start with the digit 1, "
          "17% start with 2, 12% start with 3, and so on, declining to just 5% for the digit 9. "
          "This pattern, known as Benford's Law, holds because it reflects how numbers grow "
          "proportionally in the real world.",
          size=10)
    _body(doc,
          "When a dataset deviates significantly from this expected pattern, it may indicate that "
          "amounts were manually entered, rounded, or constructed — rather than arising naturally "
          "from business transactions. The tool measures this deviation using the Mean Absolute "
          "Deviation (MAD) and a chi-square statistical test.",
          size=10)
    _body(doc,
          "Important caveat: Fixed recurring payments (e.g. monthly retainers, annual licence fees) "
          "naturally repeat the same amounts and will always deviate from Benford's distribution "
          "without being suspicious. These payments are automatically excluded from the Benford "
          "analysis and are identified separately.",
          italic=True, size=10)
    doc.add_paragraph()

    _heading(doc, "2. Isolation Forest", level=2)
    _body(doc,
          "Isolation Forest works by repeatedly splitting the payment data using random rules — "
          "for example, 'is the amount greater than $5,000?' or 'was this processed in fewer than "
          "2 days?' — until each transaction is isolated on its own. Transactions that are "
          "genuinely unusual are easier to isolate because they are different from the rest in "
          "many ways at once; they require fewer splits to separate out.",
          size=10)
    _body(doc,
          "The model examines all engineered features of each payment simultaneously — including "
          "the amount relative to the vendor's usual payments, the processing time, whether it "
          "was dated on a non-working day, and whether the payee is an individual or a company. "
          "Payments that are hardest to group with similar transactions receive the highest "
          "anomaly scores.",
          size=10)
    doc.add_paragraph()

    _heading(doc, "3. Local Outlier Factor (LOF)", level=2)
    _body(doc,
          "The Local Outlier Factor identifies payments that are unusual compared to their "
          "closest neighbours in the dataset — the most similar transactions based on amount, "
          "vendor, and timing. A payment may look ordinary in the overall dataset but be "
          "completely out of place among transactions from the same vendor.",
          size=10)
    _body(doc,
          "For example: a $50,000 payment to a vendor whose typical invoices are around $2,000 "
          "would score very highly, even if $50,000 is not an unusual amount across the whole "
          "dataset. This context-sensitivity makes LOF particularly effective for catching "
          "inflated invoices or payments to unusual recipients.",
          size=10)
    doc.add_paragraph()

    _heading(doc, "4. Statistical Z-Score Analysis", level=2)
    _body(doc,
          "For each Cost Centre and each Vendor, the tool calculates the average payment amount "
          "and how much payments typically vary from that average (standard deviation). Any "
          "payment that falls more than 2 standard deviations above its group average is "
          "flagged — a threshold that captures the top 2.5% of a normal distribution.",
          size=10)
    _body(doc,
          "This is the most direct method for identifying unusually large payments within a "
          "specific context. It is transparent and easy to explain to stakeholders, and is "
          "supported by standard statistical practice.",
          size=10)
    doc.add_paragraph()

    _heading(doc, "5. Rule-Based Flags", level=2)
    _body(doc,
          "In addition to the statistical and machine learning methods, the tool applies a set "
          "of specific rules derived from established audit and forensic accounting practice:",
          size=10)
    rules = [
        "Round number amounts (divisible by 100, 500, or 1,000) — fraudulent amounts are commonly "
        "chosen as round numbers rather than arising from genuine invoices.",
        "Transactions dated on weekends or Singapore public holidays — payments authorised "
        "outside business hours may bypass normal review controls.",
        "Month-end transactions (last 3 days of the month) — may indicate rushed processing "
        "to meet budget targets or period-end cut-off manipulation.",
        "Amounts just below a common approval threshold (e.g. $9,800 when the limit is $10,000) "
        "— a recognised technique to avoid triggering higher approval requirements.",
        "Payments to individuals (NRIC/FIN payees) — carry higher inherent risk than payments "
        "to registered companies, as they bypass standard vendor registration processes.",
        "Repeated identical amounts to the same vendor outside a regular schedule — may indicate "
        "split or duplicated payments.",
        "Unusually short or long processing time (Invoice Date to Voucher Accounting Date) — "
        "very fast processing may indicate bypassed controls; very long delays may suggest backdating.",
    ]
    for rule in rules:
        _bullet(doc, rule, size=10)

    doc.add_paragraph()
    _body(doc,
          "Each flag is used as a contributing signal — not a standalone conclusion. A transaction "
          "is only selected if multiple signals point to it, or if one signal is extremely strong. "
          "Benford's Law in particular is treated as a weak signal: a transaction will not be "
          "selected based on Benford deviation alone.",
          italic=True, size=10)


# ---------------------------------------------------------------------------
# Page 3 — Benford's Law + Risk Distribution (landscape, side by side)
# ---------------------------------------------------------------------------

def _page3_charts(doc, df_invoices, selected_invoices, benford_stats):
    section = doc.add_section()
    _set_landscape(section)

    _heading(doc, "Analytical Charts", level=1)
    _body(doc,
          "Left: Benford's Law first-digit distribution (red bars = deviant digits). "
          "Right: invoice risk score distribution with the sample selection threshold.",
          size=9)
    doc.add_paragraph()

    cutoff = float(selected_invoices['invoice_score'].min()) \
        if 'invoice_score' in selected_invoices.columns else 0.0
    img_benford = _chart_benford(benford_stats)
    img_risk    = _chart_risk_distribution(df_invoices, cutoff)

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

FEATURE_TABLE_DATA = [
    (
        "Amount vs. vendor average",
        "How much the payment amount differs from what this vendor is typically paid",
        "Z-score > 2.0",
        "Unusually large payments to a vendor may indicate over-billing or fictitious invoices",
        "The ±2 standard deviation (2-sigma) rule covers ~95% of normally distributed values. "
        "Widely referenced in AICPA and IIA audit guidance and GAAS analytical procedures.",
    ),
    (
        "Amount vs. cost centre average",
        "How much the payment amount differs from the typical amounts processed in that cost centre",
        "Z-score > 2.0",
        "Helps detect amounts that are out of place for the department, suggesting possible miscoding or inflated claims",
        "Same statistical basis as above (2-sigma rule). Applying it at cost centre level is consistent "
        "with GAAS group-level analytical procedure recommendations.",
    ),
    (
        "Round number",
        "Whether the payment amount ends in 00, 000, or 0,000",
        "Exactly divisible by 100",
        "Genuine invoice amounts rarely end in round numbers; manually chosen or fictitious amounts often do",
        "Heuristic supported by forensic accounting literature. Nigrini (2012) and the ACFE Fraud "
        "Examiners Manual identify 'round number bias' as a recognised indicator of constructed amounts.",
    ),
    (
        "Non-working day",
        "Whether the invoice is dated on a Saturday, Sunday, or Singapore public holiday",
        "Sat, Sun, or SG public holiday (holidays library)",
        "Payments authorised outside business hours may bypass the normal multi-person review and approval process",
        "Binary rule. Supported by the COSO Internal Control Framework and IIA Standard 2120, which "
        "require scrutiny of transactions occurring outside normal operating hours.",
    ),
    (
        "Month-end",
        "Whether the invoice is dated in the last 3 calendar days of the month",
        "Last 3 calendar days of month",
        "May indicate rushed processing to meet budget targets or period-end financial reporting cut-offs",
        "Recognised audit heuristic (ACFE and IIA guidance). The last-3-days window is a commonly "
        "applied cut-off in expenditure analytics.",
    ),
    (
        "Near approval threshold",
        "Whether the amount falls within 5% below a common approval limit",
        "Within 5% below SGD 1K / 5K / 10K / 50K / 100K",
        "A well-documented technique ('structuring') to avoid triggering higher-level approval requirements",
        "Known as structuring or threshold avoidance in forensic accounting. Referenced in the ACFE "
        "Fraud Examiners Manual. The 5% margin is a standard audit convention.",
    ),
    (
        "Individual payee",
        "Whether the Vendor ID matches the Singapore NRIC/FIN format (one letter, 7 digits, one letter)",
        "Regex: ^[A-Z][0-9]{7}[A-Z]$",
        "Payments to individuals carry higher inherent risk; they bypass standard vendor vetting and procurement controls",
        "Binary classification based on the Singapore NRIC/FIN ID format. IRAS and MAS regulatory "
        "guidance distinguishes individual from corporate payees.",
    ),
    (
        "Processing time",
        "Number of calendar days between Invoice Date and Voucher Accounting Date",
        "Outside 5th–95th percentile of dataset",
        "Very fast processing may indicate bypassed controls; unusually long delays may indicate backdating",
        "Percentile-based bounds are a standard non-parametric outlier method. The 5th–95th range "
        "(90% central interval) is analogous to a 90% confidence interval, consistent with GAAS "
        "analytical procedure timing analysis.",
    ),
    (
        "Description length",
        "Character length of the Voucher Line Description field",
        "Absolute z-score > 2.5",
        "Very short descriptions may indicate incomplete entries; very long ones may indicate unusual or fabricated narrative",
        "Z-score method applied to text length. A 2.5-sigma threshold (stricter than 2.0 for amounts) "
        "accounts for higher natural variability in text. Z-score thresholds of 2.0–3.0 are standard "
        "in audit data analytics.",
    ),
    (
        "Irregular repeated amount",
        "Same vendor paid the same amount more than twice, with no regular monthly/quarterly/annual schedule",
        "> 2 occurrences with no detected recurring cycle",
        "May indicate duplicated or split payments that were structured to avoid detection",
        "Heuristic aligned with ACFE guidance on duplicate payment detection. The recurrence-cycle "
        "exclusion ensures legitimate fixed payments (rent, retainers) are not falsely flagged.",
    ),
    (
        "Benford's Law first digit",
        "Whether the payment amount's first digit deviates significantly from Benford's expected frequency",
        "First digit among the top-3 most deviant digits; non-recurring payments only",
        "Systematic deviation may indicate manually constructed or manipulated amounts",
        "Based on Newcomb (1881) and Benford (1938). MAD thresholds from Nigrini (2012), the "
        "leading academic reference for forensic application of Benford's Law, recognised by the AICPA.",
    ),
]


def _page6_feature_table(doc):
    section = doc.add_section()
    _set_landscape(section)

    _heading(doc, "Feature Reference Table", level=1)
    _body(doc,
          "The table below lists each analytical feature used by the tool, the threshold that "
          "determines whether a transaction is flagged, the audit rationale, and the statistical "
          "or professional basis for the threshold.",
          size=9)
    doc.add_paragraph()

    headers    = ["Feature", "What It Measures", "Threshold for Flagging",
                  "Why It Matters", "Basis & Statistical Support"]
    col_widths = [Inches(1.6), Inches(2.0), Inches(1.8), Inches(2.2), Inches(3.2)]

    tbl = doc.add_table(rows=1 + len(FEATURE_TABLE_DATA), cols=5)
    tbl.style = 'Table Grid'

    hdr = tbl.rows[0]
    for i, (header, width) in enumerate(zip(headers, col_widths)):
        cell = hdr.cells[i]
        cell.text = header
        cell.paragraphs[0].runs[0].bold = True
        cell.paragraphs[0].runs[0].font.size = Pt(8)
        cell.paragraphs[0].runs[0].font.color.rgb = WHITE
        _shade_cell(cell, "1F3864")
        cell.width = width

    for row_idx, row_data in enumerate(FEATURE_TABLE_DATA, start=1):
        row   = tbl.rows[row_idx]
        shade = "F2F6FC" if row_idx % 2 == 0 else "FFFFFF"
        for col_idx, (value, width) in enumerate(zip(row_data, col_widths)):
            cell = row.cells[col_idx]
            cell.text = value
            cell.paragraphs[0].runs[0].font.size = Pt(7.5)
            _shade_cell(cell, shade)
            cell.width = width

    doc.add_paragraph()
    _body(doc,
          "References: Nigrini, M.J. (2012). Benford's Law: Applications for Forensic Accounting, "
          "Auditing, and Fraud Detection. ACFE Fraud Examiners Manual (current edition). "
          "AICPA Audit and Accounting Guide: Analytical Procedures. IIA Standards 2120 (Risk Management). "
          "COSO Internal Control — Integrated Framework.",
          italic=True, size=7.5)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def export_word_report(df, df_invoices, selected_invoices, benford_stats, output_path):
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
    style.font.size = Pt(10)
    style.font.name = 'Calibri'

    print("    Page 1 — Executive Summary")
    _page1(doc, df, df_invoices, selected_invoices, benford_stats)
    doc.add_page_break()

    print("    Page 2 — Methodology")
    _page2(doc)

    print("    Page 3 — Analytical Charts")
    _page3_charts(doc, df_invoices, selected_invoices, benford_stats)

    print("    Page 4 — Payment Distribution & Timeline")
    _page4_distributions(doc, df)

    print("    Page 5 — Vendor Analysis")
    _page5_vendors(doc, df)

    print("    Page 6 — Feature Reference Table")
    _page6_feature_table(doc)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    doc.save(output_path)
    print(f"  Word report saved: {output_path}")
