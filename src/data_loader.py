import pandas as pd
import numpy as np
from pathlib import Path

REQUIRED_COLUMNS = [
    'Vendor Name', 'Vendor ID', 'Cost Centre', 'Account Code',
    'Invoice Date', 'Voucher Accounting Date',
    'Invoice Number', 'Voucher ID', 'Voucher Line Description',
    'Payment Voucher Amount (SGD, Excluding GST)',
]
AMOUNT_COL = 'Payment Voucher Amount (SGD, Excluding GST)'


def load_transactions(filepath):
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(
            f"\n  File not found: {filepath}"
            "\n  Please place your Excel file in the 'data/' folder "
            "and update INPUT_FILE in Step 1."
        )

    print(f"  Loading '{path.name}'...")
    df = pd.read_excel(filepath, dtype=str)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "\n  Missing required columns:\n    " + "\n    ".join(missing) +
            "\n\n  Columns found in your file:\n    " + "\n    ".join(df.columns.tolist())
        )

    for col in ['Invoice Date', 'Voucher Accounting Date']:
        df[col] = pd.to_datetime(df[col], dayfirst=True, errors='coerce')

    df[AMOUNT_COL] = (
        df[AMOUNT_COL].astype(str)
        .str.replace(r'[SGD$,\s]', '', regex=True)
        .str.replace(r'\(([^)]+)\)', r'-\1', regex=True)
    )
    df[AMOUNT_COL] = pd.to_numeric(df[AMOUNT_COL], errors='coerce')

    before = len(df)
    df = df[df[AMOUNT_COL].notna() & (df[AMOUNT_COL] > 0)].reset_index(drop=True)
    removed = before - len(df)
    if removed:
        print(f"  Note: {removed} rows removed (missing or non-positive amounts).")

    print(f"  {len(df):,} transactions loaded successfully.")
    return df
