import re
import numpy as np
import pandas as pd
import holidays
from scipy import stats

AMOUNT_COL = 'Payment Voucher Amount (SGD, Excluding GST)'
_SG_HOLIDAYS = holidays.Singapore()

APPROVAL_THRESHOLDS = [1_000, 5_000, 10_000, 50_000, 100_000]
NEAR_THRESHOLD_PCT = 0.05

# Cycle specs: (name, min_days, max_days)
_CYCLES = [
    ('monthly',    21,  40),
    ('quarterly',  80, 100),
    ('semiannual', 170, 195),
    ('annual',     350, 380),
]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _group_zscore(df, value_col, group_col):
    def _z(x):
        s = x.std(ddof=1)
        return (x - x.mean()) / s if s > 0 else pd.Series(0.0, index=x.index)
    return df.groupby(group_col)[value_col].transform(_z).fillna(0.0)


def _sg_nonworkday(date):
    if pd.isna(date):
        return 0
    d = date.date()
    return 1 if (d.weekday() >= 5 or d in _SG_HOLIDAYS) else 0


def _round_number(amount):
    if pd.isna(amount) or amount <= 0:
        return 0
    return 1 if (amount % 100 == 0) else 0


def _near_threshold(amount):
    if pd.isna(amount):
        return 0
    for t in APPROVAL_THRESHOLDS:
        if t * (1 - NEAR_THRESHOLD_PCT) <= amount < t:
            return 1
    return 0


def _individual_payee(vendor_id):
    if pd.isna(vendor_id):
        return 0
    return 1 if re.match(r'^[A-Za-z][0-9]{7}[A-Za-z]$', str(vendor_id).strip()) else 0


def _detect_recurring(df):
    """
    Tag transactions where the same vendor-amount pair appears on a regular
    monthly, quarterly, semi-annual, or annual schedule.
    """
    is_recurring = pd.Series(False, index=df.index)

    for (vendor, amount), grp in df.groupby(['Vendor ID', AMOUNT_COL], sort=False):
        if len(grp) < 2:
            continue
        dates = grp['Invoice Date'].dropna().sort_values()
        if len(dates) < 2:
            continue
        gaps = dates.diff().dt.days.dropna().abs().tolist()
        for _, low, high in _CYCLES:
            if all(low <= g <= high for g in gaps):
                is_recurring[grp.index] = True
                break

    return is_recurring


def _prune_correlated(df, features, threshold=0.85):
    """Drop one of any feature pair with Spearman |corr| > threshold."""
    corr = df[features].fillna(0).corr(method='spearman').abs()
    dropped = set()
    messages = []
    for i, f1 in enumerate(features):
        for f2 in features[i + 1:]:
            if f1 in dropped or f2 in dropped:
                continue
            if corr.loc[f1, f2] > threshold:
                dropped.add(f2)
                messages.append(
                    f"  [Feature pruning] Dropped '{f2}' "
                    f"(Spearman corr = {corr.loc[f1, f2]:.2f} with '{f1}')"
                )
    for m in messages:
        print(m)
    if not messages:
        print("  [Feature pruning] No highly correlated features found.")
    return [f for f in features if f not in dropped]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def engineer_features(df):
    """
    Compute all risk features.
    Returns (df_with_features, ml_feature_names_after_pruning).
    """
    print("  Computing amount z-scores...")
    df['amount_log'] = np.log1p(df[AMOUNT_COL])
    df['amount_zscore_vendor'] = _group_zscore(df, AMOUNT_COL, 'Vendor ID')
    df['amount_zscore_costcentre'] = _group_zscore(df, AMOUNT_COL, 'Cost Centre')

    print("  Computing rule-based flags...")
    df['is_round_number'] = df[AMOUNT_COL].apply(_round_number)
    df['is_sg_nonworkday'] = df['Invoice Date'].apply(_sg_nonworkday)
    df['is_month_end'] = df['Invoice Date'].apply(
        lambda d: 0 if pd.isna(d) else (1 if d.day >= (d.days_in_month - 2) else 0)
    )
    df['near_threshold'] = df[AMOUNT_COL].apply(_near_threshold)
    df['is_individual_payee'] = df['Vendor ID'].apply(_individual_payee)
    df['vendor_txn_count'] = df.groupby('Vendor ID')['Voucher ID'].transform('count')

    print("  Computing processing time features...")
    df['processing_days'] = (
        df['Voucher Accounting Date'] - df['Invoice Date']
    ).dt.days
    proc_mean = df['processing_days'].mean()
    proc_std = df['processing_days'].std(ddof=1)
    df['processing_days_zscore'] = (
        ((df['processing_days'].fillna(proc_mean) - proc_mean) / proc_std).abs()
        if proc_std > 0 else 0.0
    )

    print("  Computing description length features...")
    df['desc_length'] = df['Voucher Line Description'].astype(str).str.len()
    desc_mean = df['desc_length'].mean()
    desc_std = df['desc_length'].std(ddof=1)
    df['desc_length_zscore'] = (
        ((df['desc_length'] - desc_mean) / desc_std).abs()
        if desc_std > 0 else 0.0
    )

    print("  Detecting recurring payment schedules (this may take a moment)...")
    df['is_recurring_payment'] = _detect_recurring(df).astype(int)
    n_rec = df['is_recurring_payment'].sum()
    print(f"  Found {n_rec:,} transactions on recurring schedules "
          f"(excluded from Benford's Law).")

    counts = df.groupby(['Vendor ID', AMOUNT_COL])['Voucher ID'].transform('count')
    df['same_amount_vendor_irregular'] = (
        (counts > 2) & (df['is_recurring_payment'] == 0)
    ).astype(int)

    ml_features = [
        'amount_log',
        'amount_zscore_vendor',
        'amount_zscore_costcentre',
        'vendor_txn_count',
        'processing_days_zscore',
        'desc_length_zscore',
        'is_round_number',
        'is_sg_nonworkday',
        'is_month_end',
        'is_individual_payee',
        'near_threshold',
        'same_amount_vendor_irregular',
    ]

    print("  Checking feature correlations...")
    ml_features = _prune_correlated(df, ml_features)
    print(f"  Using {len(ml_features)} features for ML models.")
    return df, ml_features
