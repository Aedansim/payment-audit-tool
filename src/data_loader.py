import pandas as pd
import numpy as np
from collections import namedtuple
from pathlib import Path

REQUIRED_COLUMNS = [
    'Vendor Name', 'Vendor ID', 'Cost Centre', 'Account Code',
    'Invoice Date', 'Voucher Accounting Date',
    'Payment Due Date', 'Payment Date',
    'Invoice Number', 'Voucher ID', 'Voucher Line Description',
    'Payment Voucher Amount (SGD, Excluding GST)',
]
AMOUNT_COL = 'Payment Voucher Amount (SGD, Excluding GST)'

# Optional columns: loaded and carried through if present, created blank if absent.
# Display/reference only — never enter ml_features or any scoring computation.
OPTIONAL_COLUMNS = ['Account Description']

# uens: set of excluded Vendor IDs (UENs); names: uen -> entity_name (display only).
ExcludedVendors = namedtuple('ExcludedVendors', ['uens', 'names'])

_EXCLUDED_FILENAMES = {
    'excluded vendors.xlsx', 'excluded_vendors.xlsx',
}


def load_transactions(filepath):
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(
            f"\n  File not found: {filepath}"
            "\n  Please place your Excel file in the 'data/' folder "
            "and update INPUT_FILE in Step 0."
        )

    print(f"  Loading '{path.name}'...")
    # Read headers first so we can build a dtype dict that keeps date columns
    # as native types — avoids Excel date serial numbers becoming strings.
    _date_cols = {'Invoice Date', 'Voucher Accounting Date', 'Payment Due Date', 'Payment Date'}
    _header_cols = pd.read_excel(filepath, nrows=0).columns.tolist()
    _dtype_dict = {c: str for c in _header_cols if c not in _date_cols}
    df = pd.read_excel(filepath, dtype=_dtype_dict)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "\n  Missing required columns:\n    " + "\n    ".join(missing) +
            "\n\n  Columns found in your file:\n    " + "\n    ".join(df.columns.tolist())
        )

    # Optional descriptive columns — retained if present, blank if absent, so
    # downstream display code can rely on the column always existing.
    for col in OPTIONAL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna('').astype(str).str.strip()
        else:
            df[col] = ''

    for col in ['Invoice Date', 'Voucher Accounting Date', 'Payment Due Date', 'Payment Date']:
        df[col] = pd.to_datetime(df[col], dayfirst=True, errors='coerce')

    df[AMOUNT_COL] = (
        df[AMOUNT_COL].astype(str)
        .str.replace(r'[SGD$,\s]', '', regex=True)
        .str.replace(r'\(([^)]+)\)', r'-\1', regex=True)
    )
    df[AMOUNT_COL] = pd.to_numeric(df[AMOUNT_COL], errors='coerce')

    before = len(df)
    df = df[df[AMOUNT_COL].notna() & (df[AMOUNT_COL] != 0)].reset_index(drop=True)
    removed = before - len(df)
    if removed:
        print(f"  Note: {removed} rows removed (missing or zero amounts).")
    n_reversals = int((df[AMOUNT_COL] < 0).sum())
    if n_reversals:
        print(f"  Note: {n_reversals} reversal/credit note row(s) detected "
              f"(negative amounts) — retained for analysis.")

    print(f"  {len(df):,} transactions loaded successfully.")
    return df


def load_excluded_vendors(data_folder):
    """Load the user-maintained exclusion list from the data/ folder.

    Looks for 'Excluded vendors.xlsx' (also 'Excluded_vendors.xlsx' /
    'Excluded Vendors.xlsx', matched case-insensitively). The file has a 'uen'
    column (matched against 'Vendor ID') and an optional 'entity_name' column
    (display only). Returns an ExcludedVendors(uens, names) namedtuple.
    """
    folder = Path(data_folder)
    match = None
    if folder.is_dir():
        for p in folder.iterdir():
            if p.is_file() and p.name.lower() in _EXCLUDED_FILENAMES:
                match = p
                break

    if match is None:
        print("  Note: No 'Excluded vendors.xlsx' file found in the data/ folder. "
              "No vendors will be excluded from sample selection on this basis.")
        return ExcludedVendors(set(), {})

    raw = pd.read_excel(match, dtype=str)
    col_map = {str(c).strip().lower(): c for c in raw.columns}

    if 'uen' not in col_map:
        raise ValueError(
            "\n  The 'Excluded vendors' file is missing a 'uen' column."
            "\n  Columns found in your file:\n    " +
            "\n    ".join(str(c) for c in raw.columns.tolist())
        )

    uens = set()
    names = {}
    has_names = 'entity_name' in col_map
    for _, row in raw.iterrows():
        uen = row[col_map['uen']]
        if pd.isna(uen):
            continue
        uen = str(uen).strip()
        if not uen:
            continue
        uens.add(uen)
        if has_names:
            name = row[col_map['entity_name']]
            names[uen] = '' if pd.isna(name) else str(name).strip()
        else:
            names[uen] = ''

    print(f"  Loaded {len(uens)} excluded vendor UEN(s) from 'Excluded vendors.xlsx'.")
    return ExcludedVendors(uens, names)
