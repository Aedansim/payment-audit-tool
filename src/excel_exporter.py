from pathlib import Path
import pandas as pd
import numpy as np
from openpyxl import Workbook
from openpyxl.styles import (
    PatternFill, Font, Alignment, Border, Side
)
from openpyxl.utils import get_column_letter
from openpyxl.formatting.rule import ColorScaleRule

AMOUNT_COL = 'Payment Voucher Amount (SGD, Excluding GST)'

# Colour palette
HEADER_FILL  = PatternFill("solid", fgColor="1F3864")   # dark navy
HEADER_FONT  = Font(color="FFFFFF", bold=True, size=10)
HIGH_FILL    = PatternFill("solid", fgColor="FFB3B3")   # light red
MED_FILL     = PatternFill("solid", fgColor="FFDDB3")   # light orange
LOW_FILL     = PatternFill("solid", fgColor="FFFAB3")   # light yellow
ALT_FILL     = PatternFill("solid", fgColor="F2F2F2")   # light grey
ALT2_FILL    = PatternFill("solid", fgColor="E8EFF8")   # soft blue-grey (voucher alternation)
SECTION_FILL = PatternFill("solid", fgColor="D9E1F2")   # soft blue

THIN = Side(style="thin", color="BBBBBB")
THIN_BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)

_TIER_FILL = {'HIGH': HIGH_FILL, 'MEDIUM': MED_FILL, 'LOW': LOW_FILL}


def _auto_width(ws, min_w=8, max_w=50):
    for col_cells in ws.columns:
        width = max(
            len(str(cell.value)) if cell.value is not None else 0
            for cell in col_cells
        )
        col_letter = get_column_letter(col_cells[0].column)
        ws.column_dimensions[col_letter].width = min(max(width + 2, min_w), max_w)


def _write_header_row(ws, headers, row=1):
    for col_idx, header in enumerate(headers, start=1):
        cell = ws.cell(row=row, column=col_idx, value=header)
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
        cell.border = THIN_BORDER


def _safe_value(value):
    """Convert numpy types and NaN for Excel compatibility."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


# ---------------------------------------------------------------------------
# Sheet 1 — Selected Vouchers
# ---------------------------------------------------------------------------

_VCH_DISPLAY_COLS = [
    'Voucher ID', 'Vendor ID', 'Vendor Name', 'Invoice Number(s)',
    'voucher_line_count', 'voucher_score', 'voucher_risk_tier',
    'voucher_flag_count', 'voucher_any_ml_consensus',
    'voucher_reason_codes',
]

_VCH_HEADERS = [
    'Voucher ID', 'Vendor ID', 'Vendor Name', 'Invoice Number(s)',
    'Line Count', 'Voucher Score', 'Risk Tier',
    'Flag Count', 'ML Consensus', 'Reason Codes',
]


def _sheet_selected_vouchers(wb, selected_vouchers):
    ws = wb.active
    ws.title = "Selected Vouchers"
    ws.freeze_panes = "B2"

    headers = ['Sample #'] + _VCH_HEADERS
    _write_header_row(ws, headers)

    cols = ['Sample #'] + _VCH_DISPLAY_COLS
    present = [c for c in cols if c in selected_vouchers.columns]

    for r_idx, row_data in enumerate(
            selected_vouchers[present].itertuples(index=False), start=2):
        tier_idx = present.index('voucher_risk_tier') if 'voucher_risk_tier' in present else -1
        tier = row_data[tier_idx] if tier_idx >= 0 else 'LOW'
        row_fill = _TIER_FILL.get(tier, LOW_FILL)

        for c_idx, value in enumerate(row_data, start=1):
            cell = ws.cell(row=r_idx, column=c_idx)
            cell.value = _safe_value(value)
            cell.border = THIN_BORDER
            cell.alignment = Alignment(vertical='top', wrap_text=True)
            cell.fill = row_fill

            col_name = present[c_idx - 1]
            if col_name == 'voucher_score':
                cell.number_format = '0.0000'

    ws.row_dimensions[1].height = 30
    _auto_width(ws)
    n = len(headers)
    ws.column_dimensions[get_column_letter(n)].width = 60       # Reason Codes


# ---------------------------------------------------------------------------
# Sheet 2 — Voucher Line Detail
# ---------------------------------------------------------------------------

_ORIG_COLS = [
    'Vendor Name', 'Vendor ID', 'Cost Centre', 'Account Code',
    'Invoice Date', 'Voucher Accounting Date',
    'Invoice Number', 'Voucher ID', 'Voucher Line Description',
    AMOUNT_COL,
]

_LINE_SCORE_COLS = {
    'risk_score':        'Risk Score',
    'if_score':          'Isolation Forest',
    'lof_score':         'Local Outlier',
    'zscore_score':      'Z-Score Signal',
    'benford_score':     "Benford Score",
    'rule_flags_score':  'Rule Flags Score',
    'ML_Consensus_Flag': 'ML Consensus Count',
}

_LINE_FLAG_COLS = [
    'is_round_number', 'is_sg_nonworkday', 'is_month_end',
    'near_threshold', 'is_individual_payee',
    'same_amount_vendor_irregular', 'is_recurring_payment',
    'benford_flag', 'processing_days',
]


def _sheet_voucher_line_detail(wb, df_scored, selected_vouchers):
    ws = wb.create_sheet("Voucher Line Detail")
    ws.freeze_panes = "B2"

    selected_vids = set(selected_vouchers['Voucher ID']) \
        if 'Voucher ID' in selected_vouchers.columns else set()

    if selected_vids and 'Voucher ID' in df_scored.columns:
        lines = df_scored[df_scored['Voucher ID'].isin(selected_vids)].copy()
    else:
        lines = df_scored.copy()

    lines = lines.sort_values(
        ['Voucher ID', 'risk_score'], ascending=[True, False]
    ).reset_index(drop=True)

    orig_present  = [c for c in _ORIG_COLS if c in lines.columns]
    score_present = [c for c in _LINE_SCORE_COLS if c in lines.columns]
    flag_present  = [c for c in _LINE_FLAG_COLS if c in lines.columns]
    reason_col    = ['_line_reason'] if '_line_reason' in lines.columns else []

    display_cols = orig_present + score_present + flag_present + reason_col
    rename_map = {**_LINE_SCORE_COLS, '_line_reason': 'Line Reason Codes'}
    sub = lines[display_cols].rename(columns=rename_map)

    _write_header_row(ws, list(sub.columns))

    date_fmt = {'Invoice Date': 'DD/MM/YYYY', 'Voucher Accounting Date': 'DD/MM/YYYY'}

    # Alternate shading by Voucher ID group
    fills = [ALT_FILL, ALT2_FILL]
    fill_idx = 0
    prev_vid = None

    for r_idx, orig_row in lines[display_cols].iterrows():
        cur_vid = orig_row.get('Voucher ID', None) if 'Voucher ID' in lines.columns else None
        if cur_vid != prev_vid:
            fill_idx = 1 - fill_idx
            prev_vid = cur_vid
        row_fill = fills[fill_idx]

        for c_idx, (col_name, value) in enumerate(
                zip(sub.columns, orig_row[display_cols].values), start=1):
            cell = ws.cell(row=r_idx + 2, column=c_idx)
            cell.value = _safe_value(value)
            cell.border = THIN_BORDER
            cell.alignment = Alignment(vertical='top', wrap_text=False)
            cell.fill = row_fill
            if col_name in date_fmt:
                cell.number_format = date_fmt[col_name]

    ws.row_dimensions[1].height = 30
    _auto_width(ws)
    if 'Line Reason Codes' in sub.columns:
        reason_idx = list(sub.columns).index('Line Reason Codes') + 1
        ws.column_dimensions[get_column_letter(reason_idx)].width = 55


# ---------------------------------------------------------------------------
# Sheet 3 — All Vouchers Scored
# ---------------------------------------------------------------------------

_ALL_VCH_COLS = [
    'Voucher ID', 'Vendor ID', 'Vendor Name', 'Invoice Number(s)',
    'voucher_line_count', 'voucher_score', 'voucher_risk_tier',
    'voucher_max_score', 'voucher_mean_score',
    'voucher_flag_count', 'voucher_any_ml_consensus',
    'voucher_reason_codes',
]

_ALL_VCH_HEADERS = [
    'Voucher ID', 'Vendor ID', 'Vendor Name', 'Invoice Number(s)',
    'Line Count', 'Voucher Score', 'Risk Tier',
    'Max Line Score', 'Mean Line Score',
    'Flag Count', 'ML Consensus', 'Reason Codes',
]


def _sheet_all_vouchers(wb, df_vouchers):
    ws = wb.create_sheet("All Vouchers Scored")
    ws.freeze_panes = "B2"

    present = [c for c in _ALL_VCH_COLS if c in df_vouchers.columns]
    headers = [_ALL_VCH_HEADERS[_ALL_VCH_COLS.index(c)] for c in present]
    _write_header_row(ws, headers)

    for r_idx, row_data in enumerate(
            df_vouchers[present].itertuples(index=False), start=2):
        fill = ALT_FILL if r_idx % 2 == 0 else PatternFill("solid", fgColor="FFFFFF")

        for c_idx, value in enumerate(row_data, start=1):
            cell = ws.cell(row=r_idx, column=c_idx)
            cell.value = _safe_value(value)
            cell.border = THIN_BORDER
            cell.alignment = Alignment(vertical='top', wrap_text=False)
            cell.fill = fill

            col_name = present[c_idx - 1]
            if col_name in ('voucher_score', 'voucher_max_score', 'voucher_mean_score'):
                cell.number_format = '0.0000'

    ws.row_dimensions[1].height = 30
    _auto_width(ws)

    if 'voucher_score' in present:
        sc_letter = get_column_letter(present.index('voucher_score') + 1)
        ws.conditional_formatting.add(
            f"{sc_letter}2:{sc_letter}{len(df_vouchers) + 1}",
            ColorScaleRule(
                start_type='min', start_color='63BE7B',
                mid_type='percentile', mid_value=50, mid_color='FFEB84',
                end_type='max', end_color='F8696B',
            )
        )

    if 'voucher_reason_codes' in present:
        rc_letter = get_column_letter(present.index('voucher_reason_codes') + 1)
        ws.column_dimensions[rc_letter].width = 60


# ---------------------------------------------------------------------------
# Sheet 4 — All Lines Scored  (full row-level dataset, reference)
# ---------------------------------------------------------------------------

_SCORE_COLS_DISPLAY = {
    'risk_score':       'Risk Score',
    'if_score':         'Isolation Forest Score',
    'lof_score':        'Local Outlier Score',
    'zscore_score':     'Z-Score Signal',
    'benford_score':    "Benford's Score",
    'rule_flags_score': 'Rule Flags Score',
}


def _sheet_all_lines(wb, df_scored):
    ws = wb.create_sheet("All Lines Scored")
    ws.freeze_panes = "B2"

    score_cols = [c for c in _SCORE_COLS_DISPLAY if c in df_scored.columns]
    flag_cols  = [c for c in [
        'is_round_number', 'is_sg_nonworkday', 'is_month_end',
        'near_threshold', 'is_individual_payee',
        'same_amount_vendor_irregular', 'is_recurring_payment',
        'benford_flag', 'processing_days',
    ] if c in df_scored.columns]

    orig_present = [c for c in _ORIG_COLS if c in df_scored.columns]
    display_cols = orig_present + score_cols + flag_cols
    score_rename = {k: v for k, v in _SCORE_COLS_DISPLAY.items() if k in score_cols}
    sub = df_scored[display_cols].rename(columns=score_rename)

    _write_header_row(ws, list(sub.columns))

    date_fmt = {'Invoice Date': 'DD/MM/YYYY', 'Voucher Accounting Date': 'DD/MM/YYYY'}

    for r_idx, row_data in enumerate(sub.itertuples(index=False), start=2):
        fill = ALT_FILL if r_idx % 2 == 0 else PatternFill("solid", fgColor="FFFFFF")
        for c_idx, (col_name, value) in enumerate(zip(sub.columns, row_data), start=1):
            cell = ws.cell(row=r_idx, column=c_idx)
            cell.value = _safe_value(value)
            cell.border = THIN_BORDER
            cell.alignment = Alignment(vertical='top', wrap_text=False)
            cell.fill = fill
            if col_name in date_fmt:
                cell.number_format = date_fmt[col_name]

    ws.row_dimensions[1].height = 30
    _auto_width(ws)

    risk_col_idx = next(
        (i + 1 for i, c in enumerate(sub.columns) if c == 'Risk Score'), None
    )
    if risk_col_idx:
        rc = get_column_letter(risk_col_idx)
        ws.conditional_formatting.add(
            f"{rc}2:{rc}{len(sub) + 1}",
            ColorScaleRule(
                start_type='min', start_color='63BE7B',
                mid_type='percentile', mid_value=50, mid_color='FFEB84',
                end_type='max', end_color='F8696B',
            )
        )


# ---------------------------------------------------------------------------
# Sheet 5 — Benford's Law (unchanged)
# ---------------------------------------------------------------------------

def _sheet_benford(wb, benford_summary, stats):
    ws = wb.create_sheet("Benford's Law")

    ws['A1'] = "Benford's Law Analysis"
    ws['A1'].font = Font(bold=True, size=14, color="1F3864")
    ws.merge_cells('A1:G1')

    ws['A2'] = (
        f"Analysed {stats['n_analyzed']:,} non-recurring transactions  |  "
        f"Excluded {stats['n_excluded_recurring']:,} recurring payments"
    )
    ws['A2'].font = Font(italic=True, size=10, color="444444")
    ws.merge_cells('A2:G2')
    ws.row_dimensions[2].height = 18

    stats_data = [
        ("MAD (Mean Absolute Deviation)", f"{stats['mad']:.4f}"),
        ("Conformity Verdict", stats['conformity']),
        ("Chi-Square Statistic", f"{stats['chi2']:.4f}"),
        ("Chi-Square p-value", f"{stats['p_value']:.4f}"),
        ("Most Deviant Digits", ", ".join(str(d) for d in stats['deviant_digits'])),
    ]

    for row_offset, (label, value) in enumerate(stats_data, start=4):
        ws.cell(row=row_offset, column=1, value=label).font = Font(bold=True)
        ws.cell(row=row_offset, column=2, value=value)

    verdict_cell = ws.cell(row=5, column=2)
    conformity_colors = {
        "Conformity": "00B050",
        "Acceptable": "70AD47",
        "Marginally Acceptable": "FFC000",
        "Non-Conformity": "FF0000",
    }
    c = conformity_colors.get(stats['conformity'], "000000")
    verdict_cell.font = Font(bold=True, color=c)

    # Explanation block for MAD, Chi-Square, and Conformity Verdict
    expl_hdr = 9
    ws.cell(row=expl_hdr, column=1,
            value="Understanding These Metrics").font = Font(bold=True, size=11, color="1F3864")
    ws.cell(row=expl_hdr, column=1).fill = PatternFill("solid", fgColor="D9E1F2")
    ws.merge_cells(f'A{expl_hdr}:G{expl_hdr}')
    ws.row_dimensions[expl_hdr].height = 18

    _explanations = [
        (10, "MAD (Mean Absolute Deviation)",
         "Measures the average absolute difference between observed and Benford-expected first-digit "
         "frequencies. Thresholds (Nigrini, 2012): < 0.006 = Close Conformity; "
         "0.006–0.012 = Acceptable Conformity; 0.012–0.015 = Marginally Acceptable; "
         "> 0.015 = Non-Conformity. A lower MAD means the data more closely follows Benford's Law. "
         "MAD is the primary practical measure for audit interpretation."),
        (11, "Chi-Square Statistic & p-value",
         "Tests whether the observed digit frequencies are statistically significantly different from "
         "Benford's expected values. A p-value < 0.05 indicates the difference is statistically "
         "significant. Important caveat: for large datasets (> 1,000 transactions), chi-square is very "
         "sensitive and will often flag minor deviations as significant even when they are not "
         "practically meaningful. Always read chi-square alongside MAD — a significant p-value with "
         "a small MAD (< 0.012) may not warrant audit action."),
        (12, "Conformity Verdict",
         "Summarises the overall finding based on the MAD threshold. Non-Conformity does not mean "
         "fraud — it means the first-digit distribution is unusual and warrants investigation of the "
         "most deviant digits. The tool assigns Benford's Law only a 5% weight in the composite risk "
         "score and further suppresses it when all other risk signals are below average, so a "
         "Non-Conformity verdict will not on its own cause any voucher to be selected for audit."),
    ]
    for _rn, _lbl, _txt in _explanations:
        ws.cell(row=_rn, column=1, value=_lbl).font = Font(bold=True, size=9, color="1F3864")
        _ec = ws.cell(row=_rn, column=2, value=_txt)
        _ec.font = Font(size=9)
        _ec.alignment = Alignment(wrap_text=True, vertical='top')
        ws.merge_cells(f'B{_rn}:G{_rn}')
        ws.row_dimensions[_rn].height = 52

    _key_row = 14
    _key_msg = (
        "Key Takeaway: Read MAD and Chi-Square together. MAD quantifies the size of the deviation; "
        "Chi-Square (p-value) tests whether it is statistically significant for the sample size. "
        "In large datasets, a significant p-value paired with a small MAD (< 0.012) may not be "
        "practically meaningful for audit purposes. The strongest audit signal is a Non-Conformity "
        "MAD (> 0.015) combined with a low p-value — this warrants investigation of the most deviant "
        "digits (highlighted in orange in the frequency table below). Transactions whose first digit "
        "falls among the most deviant are identified by the 'Most Deviant Digits' field above."
    )
    _kc = ws.cell(row=_key_row, column=1, value=_key_msg)
    _kc.font = Font(bold=True, size=9, color="1F3864")
    _kc.alignment = Alignment(wrap_text=True, vertical='top')
    _kc.fill = PatternFill("solid", fgColor="FFF2CC")
    ws.merge_cells(f'A{_key_row}:G{_key_row}')
    ws.row_dimensions[_key_row].height = 72

    tbl_start = 16
    _write_header_row(ws, list(benford_summary.columns), row=tbl_start)
    for r_idx, row_data in enumerate(benford_summary.itertuples(index=False), start=tbl_start + 1):
        digit = row_data[0]
        is_deviant = digit in stats['deviant_digits']
        for c_idx, value in enumerate(row_data, start=1):
            cell = ws.cell(row=r_idx, column=c_idx, value=value)
            cell.border = THIN_BORDER
            cell.alignment = Alignment(horizontal='center')
            if is_deviant:
                cell.fill = PatternFill("solid", fgColor="FFE0CC")

    ws.row_dimensions[tbl_start].height = 25
    _auto_width(ws)

    note_row = tbl_start + len(benford_summary) + 2
    ws.cell(row=note_row, column=1,
            value="Note: Recurring payments (monthly, quarterly, semi-annual, annual) "
                  "are excluded from this analysis as they naturally deviate from "
                  "Benford's distribution without being suspicious.").font = Font(italic=True, size=9, color="666666")
    ws.merge_cells(f'A{note_row}:G{note_row}')


# ---------------------------------------------------------------------------
# Sheet 6 — Summary
# ---------------------------------------------------------------------------

def _sheet_summary(wb, df_scored, df_vouchers, selected_vouchers, benford_stats):
    ws = wb.create_sheet("Summary")

    ws['A1'] = "Payment Audit — Summary"
    ws['A1'].font = Font(bold=True, size=14, color="1F3864")
    ws.merge_cells('A1:C1')
    ws.row_dimensions[1].height = 28

    n_lines    = len(df_scored)
    n_vouchers = len(df_vouchers)
    avg_lines  = n_lines / n_vouchers if n_vouchers > 0 else 0
    n_sel      = len(selected_vouchers)
    n_sel_high = int((selected_vouchers.get('voucher_risk_tier', pd.Series()) == 'HIGH').sum())
    n_sel_med  = int((selected_vouchers.get('voucher_risk_tier', pd.Series()) == 'MEDIUM').sum())
    n_sel_low  = int((selected_vouchers.get('voucher_risk_tier', pd.Series()) == 'LOW').sum())
    n_vch_high = int((df_vouchers.get('voucher_risk_tier', pd.Series()) == 'HIGH').sum())
    n_vch_med  = int((df_vouchers.get('voucher_risk_tier', pd.Series()) == 'MEDIUM').sum())
    n_vch_low  = int((df_vouchers.get('voucher_risk_tier', pd.Series()) == 'LOW').sum())

    rows = [
        ("DATASET", None),
        ("Total transaction line items", f"{n_lines:,}"),
        ("Unique payment vouchers", f"{n_vouchers:,}"),
        ("Average lines per voucher", f"{avg_lines:.1f}"),
        ("Recurring payments excluded from Benford's", f"{benford_stats.get('n_excluded_recurring', 0):,}"),
        ("", None),
        ("VOUCHER RISK TIERS (all vouchers)", None),
        ("HIGH risk vouchers (top 5%)", f"{n_vch_high:,}"),
        ("MEDIUM risk vouchers (next 15%)", f"{n_vch_med:,}"),
        ("LOW risk vouchers", f"{n_vch_low:,}"),
        ("", None),
        ("AUDIT SAMPLE SELECTED", None),
        ("Total vouchers selected", f"{n_sel:,}"),
        ("  — HIGH risk (mandatory)", f"{n_sel_high:,}"),
        ("  — MEDIUM risk (proportional)", f"{n_sel_med:,}"),
        ("  — LOW risk (baseline)", f"{n_sel_low:,}"),
        ("Total line items in selected vouchers",
         f"{int(selected_vouchers['voucher_line_count'].sum()):,}"
         if 'voucher_line_count' in selected_vouchers.columns else "N/A"),
    ]

    for r_offset, (label, value) in enumerate(rows, start=3):
        label_cell = ws.cell(row=r_offset, column=1, value=label)
        is_section = value is None and label != ''
        if is_section:
            label_cell.font = Font(bold=True, color="1F3864", size=10)
            label_cell.fill = SECTION_FILL
            ws.merge_cells(f'A{r_offset}:C{r_offset}')
        elif label == '':
            pass
        else:
            label_cell.font = Font(size=10)
            val_cell = ws.cell(row=r_offset, column=2, value=value)
            val_cell.font = Font(size=10)
            val_cell.alignment = Alignment(wrap_text=True, vertical='top')
            ws.merge_cells(f'B{r_offset}:C{r_offset}')

    ws.column_dimensions['A'].width = 42
    ws.column_dimensions['B'].width = 30
    ws.column_dimensions['C'].width = 30

    for r in range(3, 3 + len(rows)):
        ws.row_dimensions[r].height = 15


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def export_excel(df_scored, df_vouchers, selected_vouchers,
                 benford_summary, benford_stats, output_path):
    wb = Workbook()

    _sheet_selected_vouchers(wb, selected_vouchers)
    _sheet_voucher_line_detail(wb, df_scored, selected_vouchers)
    _sheet_all_vouchers(wb, df_vouchers)
    _sheet_all_lines(wb, df_scored)
    _sheet_benford(wb, benford_summary, benford_stats)
    _sheet_summary(wb, df_scored, df_vouchers, selected_vouchers, benford_stats)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    wb.save(output_path)
    print(f"  Excel saved: {output_path}")
