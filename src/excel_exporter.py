from pathlib import Path
import pandas as pd
import numpy as np
from openpyxl import Workbook
from openpyxl.styles import (
    PatternFill, Font, Alignment, Border, Side, GradientFill
)
from openpyxl.utils import get_column_letter
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.formatting.rule import ColorScaleRule

AMOUNT_COL = 'Payment Voucher Amount (SGD, Excluding GST)'

# Colour palette
HEADER_FILL   = PatternFill("solid", fgColor="1F3864")   # dark navy
HEADER_FONT   = Font(color="FFFFFF", bold=True, size=10)
HIGH_FILL     = PatternFill("solid", fgColor="FFB3B3")   # light red
MED_FILL      = PatternFill("solid", fgColor="FFDDB3")   # light orange
LOW_FILL      = PatternFill("solid", fgColor="FFFAB3")   # light yellow
ALT_FILL      = PatternFill("solid", fgColor="F2F2F2")   # light grey
SECTION_FILL  = PatternFill("solid", fgColor="D9E1F2")   # soft blue

THIN = Side(style="thin", color="BBBBBB")
THIN_BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)


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


def _write_dataframe(ws, df, start_row=2, number_fmt=None):
    """Write df to worksheet starting at start_row. Returns last row written."""
    number_fmt = number_fmt or {}
    col_names = list(df.columns)

    for r_idx, row_data in enumerate(df.itertuples(index=False), start=start_row):
        for c_idx, (col_name, value) in enumerate(zip(col_names, row_data), start=1):
            cell = ws.cell(row=r_idx, column=c_idx)

            # Convert numpy types for Excel compatibility
            if isinstance(value, (np.integer,)):
                value = int(value)
            elif isinstance(value, (np.floating,)):
                value = float(value)
            elif isinstance(value, float) and np.isnan(value):
                value = None

            cell.value = value
            cell.border = THIN_BORDER
            cell.alignment = Alignment(vertical='top', wrap_text=True)

            if col_name in number_fmt:
                cell.number_format = number_fmt[col_name]

    return start_row + len(df) - 1


# ---------------------------------------------------------------------------
# Sheet 1 — Selected Samples
# ---------------------------------------------------------------------------

_ORIGINAL_COLS = [
    'Vendor Name', 'Vendor ID', 'Cost Centre', 'Account Code',
    'Invoice Date', 'Voucher Accounting Date',
    'Invoice Number', 'Voucher ID', 'Voucher Line Description',
    AMOUNT_COL,
]

_SCORE_COLS_DISPLAY = {
    'risk_score':   'Risk Score',
    'if_score':     'Isolation Forest Score',
    'lof_score':    'Local Outlier Score',
    'zscore_score': 'Z-Score Signal',
    'benford_score': "Benford's Score",
    'rule_flags_score': 'Rule Flags Score',
}


def _sheet_selected(wb, selected):
    ws = wb.active
    ws.title = "Selected Samples"
    ws.freeze_panes = "B2"

    orig_present = [c for c in _ORIGINAL_COLS if c in selected.columns]
    headers = ['Sample #'] + orig_present + ['Risk Score', 'Selection Reasons']
    _write_header_row(ws, headers)

    sub = selected[['Sample #'] + orig_present + ['risk_score', 'Selection Reasons']].copy()
    sub = sub.rename(columns={'risk_score': 'Risk Score'})

    # Determine risk tier based on rank
    n = len(sub)
    top_third = n // 3

    for r_idx, row_data in enumerate(sub.itertuples(index=False), start=2):
        rank = row_data[0]  # Sample #
        for c_idx, value in enumerate(row_data, start=1):
            cell = ws.cell(row=r_idx, column=c_idx)
            if isinstance(value, (np.integer,)):
                value = int(value)
            elif isinstance(value, (np.floating,)):
                value = round(float(value), 4)
            elif isinstance(value, float) and np.isnan(value):
                value = None
            cell.value = value
            cell.border = THIN_BORDER
            cell.alignment = Alignment(vertical='top', wrap_text=True)

        # Row colour by risk tier
        if rank <= top_third:
            row_fill = HIGH_FILL
        elif rank <= top_third * 2:
            row_fill = MED_FILL
        else:
            row_fill = LOW_FILL

        for c_idx in range(1, len(headers) + 1):
            ws.cell(row=r_idx, column=c_idx).fill = row_fill

        # Format date columns
        date_cols_idx = [i + 1 for i, c in enumerate(['Sample #'] + orig_present)
                         if c in ('Invoice Date', 'Voucher Accounting Date')]
        for ci in date_cols_idx:
            ws.cell(row=r_idx, column=ci).number_format = 'DD/MM/YYYY'

    ws.row_dimensions[1].height = 30
    _auto_width(ws)

    # Fix width of long text columns
    reasons_col = get_column_letter(len(headers))
    ws.column_dimensions[reasons_col].width = 55


# ---------------------------------------------------------------------------
# Sheet 2 — All Transactions
# ---------------------------------------------------------------------------

def _sheet_all(wb, df_scored):
    ws = wb.create_sheet("All Transactions")
    ws.freeze_panes = "B2"

    score_cols = [c for c in _SCORE_COLS_DISPLAY if c in df_scored.columns]
    flag_cols  = [c for c in [
        'is_round_number', 'is_sg_nonworkday', 'is_month_end',
        'near_threshold', 'is_individual_payee',
        'same_amount_vendor_irregular', 'is_recurring_payment',
        'benford_flag', 'processing_days',
    ] if c in df_scored.columns]

    orig_present = [c for c in _ORIGINAL_COLS if c in df_scored.columns]
    display_cols = orig_present + score_cols + flag_cols

    score_rename = {k: v for k, v in _SCORE_COLS_DISPLAY.items() if k in score_cols}
    sub = df_scored[display_cols].rename(columns=score_rename)

    _write_header_row(ws, list(sub.columns))

    date_num_fmt = {
        'Invoice Date': 'DD/MM/YYYY',
        'Voucher Accounting Date': 'DD/MM/YYYY',
    }

    for r_idx, row_data in enumerate(sub.itertuples(index=False), start=2):
        fill = ALT_FILL if r_idx % 2 == 0 else PatternFill("solid", fgColor="FFFFFF")
        for c_idx, (col_name, value) in enumerate(zip(sub.columns, row_data), start=1):
            cell = ws.cell(row=r_idx, column=c_idx)
            if isinstance(value, (np.integer,)):
                value = int(value)
            elif isinstance(value, (np.floating,)):
                value = round(float(value), 4) if not np.isnan(value) else None
            elif isinstance(value, float) and np.isnan(value):
                value = None
            cell.value = value
            cell.border = THIN_BORDER
            cell.alignment = Alignment(vertical='top', wrap_text=False)
            cell.fill = fill
            if col_name in date_num_fmt:
                cell.number_format = date_num_fmt[col_name]

    ws.row_dimensions[1].height = 30
    _auto_width(ws)

    # Conditional formatting on Risk Score column
    risk_col_idx = next(
        (i + 1 for i, c in enumerate(sub.columns) if c == 'Risk Score'), None
    )
    if risk_col_idx:
        risk_col_letter = get_column_letter(risk_col_idx)
        ws.conditional_formatting.add(
            f"{risk_col_letter}2:{risk_col_letter}{len(sub) + 1}",
            ColorScaleRule(
                start_type='min', start_color='63BE7B',
                mid_type='percentile', mid_value=50, mid_color='FFEB84',
                end_type='max', end_color='F8696B',
            )
        )


# ---------------------------------------------------------------------------
# Sheet 3 — Benford's Law
# ---------------------------------------------------------------------------

def _sheet_benford(wb, benford_summary, stats):
    ws = wb.create_sheet("Benford's Law")

    # Title block
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

    # Stats summary
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

    # Conformity colour
    verdict_cell = ws.cell(row=5, column=2)
    conformity_colors = {
        "Conformity": "00B050",
        "Acceptable": "70AD47",
        "Marginally Acceptable": "FFC000",
        "Non-Conformity": "FF0000",
    }
    c = conformity_colors.get(stats['conformity'], "000000")
    verdict_cell.font = Font(bold=True, color=c)

    # Digit table
    tbl_start = 11
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

    # Note
    note_row = tbl_start + len(benford_summary) + 2
    ws.cell(row=note_row, column=1,
            value="Note: Recurring payments (monthly, quarterly, semi-annual, annual) "
                  "are excluded from this analysis as they naturally deviate from "
                  "Benford's distribution without being suspicious.").font = Font(italic=True, size=9, color="666666")
    ws.merge_cells(f'A{note_row}:G{note_row}')


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def export_excel(df_scored, selected, benford_summary, benford_stats, output_path):
    wb = Workbook()

    _sheet_selected(wb, selected)
    _sheet_all(wb, df_scored)
    _sheet_benford(wb, benford_summary, benford_stats)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    wb.save(output_path)
    print(f"  Excel saved: {output_path}")
