import re
import numpy as np
import pandas as pd
from scipy import stats

AMOUNT_COL = 'Payment Voucher Amount (SGD, Excluding GST)'

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


def _is_weekend_payment(date):
    if pd.isna(date):
        return 0
    return 1 if date.date().weekday() >= 5 else 0


def _detect_duplicates(df):
    """Flag rows where the same invoice (vendor + invoice number + amount) appears in
    more than one distinct Voucher ID — a potential double payment.
    Returns (is_duplicate Series, duplicate_matched_invoice Series)."""
    is_dup = pd.Series(0, index=df.index)
    matched_inv = pd.Series('', index=df.index, dtype=object)
    has_invoice = df['Invoice Number'].notna() & (
        df['Invoice Number'].astype(str).str.strip() != ''
    )
    if not has_invoice.any():
        return is_dup, matched_inv
    relevant = df[has_invoice]
    key = ['Vendor ID', 'Invoice Number', AMOUNT_COL]
    cross_voucher = relevant.groupby(key)['Voucher ID'].transform('nunique') > 1
    flagged_idx = relevant[cross_voucher].index
    is_dup.loc[flagged_idx] = 1
    matched_inv.loc[flagged_idx] = (
        relevant.loc[flagged_idx, 'Invoice Number'].astype(str).str.strip()
    )
    return is_dup, matched_inv


def _vendor_amount_cv(df):
    """Per-vendor coefficient of variation on positive amounts only.
    Higher values indicate vendors whose billing amounts vary more month-to-month,
    widening the z-score tolerance and potentially masking overpayments."""
    def _cv(x):
        x = x.where(x > 0).dropna()
        if len(x) < 2:
            return 0.0
        m = x.mean()
        return float(x.std(ddof=1) / m) if m > 0 else 0.0
    return df.groupby('Vendor ID')[AMOUNT_COL].transform(_cv).fillna(0.0)


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


def _detect_split_purchase(df):
    """Flag transactions where the same vendor has 2+ invoices on the same date
    with alphanumerically sequential numeric suffixes — possible split purchase."""
    result = pd.Series(0, index=df.index)

    suffixes = df['Invoice Number'].astype(str).str.strip().str.extract(r'(\d+)$', expand=False)

    work = pd.DataFrame({
        'vid':        df['Vendor ID'],
        'idate':      df['Invoice Date'],
        'has_suffix': suffixes.notna().astype(int),
        'suffix':     suffixes,
    }, index=df.index).dropna(subset=['idate'])

    if work.empty:
        return result

    g = work.groupby(['vid', 'idate'])['has_suffix']
    work['grp_total']    = g.transform('count')
    work['grp_with_suf'] = g.transform('sum')

    # Keep only groups where every row has a suffix and group size >= 2
    valid = work[(work['grp_total'] >= 2) & (work['grp_total'] == work['grp_with_suf'])].copy()
    if valid.empty:
        return result

    valid['suf_int'] = valid['suffix'].astype(int)
    gs = valid.groupby(['vid', 'idate'])['suf_int']
    cnt = gs.transform('count')
    mn  = gs.transform('min')
    mx  = gs.transform('max')
    nu  = gs.transform('nunique')

    # Consecutive integer range: max − min == count − 1 with no duplicates
    is_split = (mx - mn == cnt - 1) & (cnt == nu)
    result.loc[valid[is_split].index] = 1
    return result


def _is_digit_transposition(a, b):
    """Return True if two positive amounts differ by exactly one digit-position swap.
    Operates on cent-integer strings so decimal places are included in the comparison."""
    a_str = str(int(round(a * 100)))
    b_str = str(int(round(b * 100)))
    if len(a_str) != len(b_str):
        return False
    diffs = [(ca, cb) for ca, cb in zip(a_str, b_str) if ca != cb]
    return len(diffs) == 2 and diffs[0][0] == diffs[1][1] and diffs[0][1] == diffs[1][0]


def _detect_transposed_amounts(df):
    """Flag transactions where same vendor and description have digit-transposed amounts —
    exactly two digit positions swapped in the cent-integer representation, suggesting a keying error.
    Returns (is_transposed Series, transposed_matched_invoice Series)."""
    result = pd.Series(0, index=df.index)
    matched_inv = pd.Series('', index=df.index, dtype=object)
    pos_mask = df[AMOUNT_COL] > 0
    if not pos_mask.any():
        return result, matched_inv
    pos = df[pos_mask].copy()
    pos['_desc_key'] = pos['Voucher Line Description'].astype(str).str.strip().str.lower()
    inv_series = df['Invoice Number'].astype(str).str.strip()
    for (vid, desc), grp in pos.groupby(['Vendor ID', '_desc_key'], sort=False):
        if len(grp) < 2:
            continue
        idxs = grp.index.tolist()
        amounts = grp[AMOUNT_COL].tolist()
        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                if _is_digit_transposition(amounts[i], amounts[j]):
                    result.loc[idxs[i]] = 1
                    result.loc[idxs[j]] = 1
                    if not matched_inv.loc[idxs[i]]:
                        matched_inv.loc[idxs[i]] = inv_series.loc[idxs[j]]
                    if not matched_inv.loc[idxs[j]]:
                        matched_inv.loc[idxs[j]] = inv_series.loc[idxs[i]]
    return result, matched_inv


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
    df['amount_log'] = np.log1p(df[AMOUNT_COL].abs())
    df['amount_zscore_vendor'] = _group_zscore(df, AMOUNT_COL, 'Vendor ID')
    df['amount_zscore_costcentre'] = _group_zscore(df, AMOUNT_COL, 'Cost Centre')

    print("  Computing rule-based flags...")
    df['is_reversal'] = (df[AMOUNT_COL] < 0).astype(int)
    df['is_duplicate'], df['duplicate_matched_invoice'] = _detect_duplicates(df)
    df['is_round_number'] = df[AMOUNT_COL].apply(_round_number)
    df['is_weekend_payment'] = df['Invoice Date'].apply(_is_weekend_payment)
    df['is_month_end'] = df['Voucher Accounting Date'].apply(
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

    print("  Computing vendor billing consistency (coefficient of variation)...")
    df['vendor_amount_cv'] = _vendor_amount_cv(df)

    print("  Detecting recurring payment schedules (this may take a moment)...")
    df['is_recurring_payment'] = _detect_recurring(df).astype(int)
    n_rec = df['is_recurring_payment'].sum()
    print(f"  Found {n_rec:,} transactions on recurring schedules "
          f"(excluded from Benford's Law).")

    counts = df.groupby(['Vendor ID', AMOUNT_COL])['Voucher ID'].transform('count')
    df['same_amount_vendor_irregular'] = (
        (counts > 2) & (df['is_recurring_payment'] == 0)
    ).astype(int)

    print("  Detecting split purchase risk (same vendor, same date, sequential invoice numbers)...")
    df['is_split_purchase_risk'] = _detect_split_purchase(df)
    n_split = int(df['is_split_purchase_risk'].sum())
    print(f"  Found {n_split:,} transactions with split purchase risk.")

    print("  Detecting transposed amounts (same vendor and description, digit-transposed value)...")
    df['is_transposed_amount'], df['transposed_matched_invoice'] = _detect_transposed_amounts(df)
    n_trans = int(df['is_transposed_amount'].sum())
    print(f"  Found {n_trans:,} transactions with possible transposed amounts.")

    ml_features = [
        'amount_log',
        'amount_zscore_vendor',
        'amount_zscore_costcentre',
        'vendor_txn_count',
        'vendor_amount_cv',
        'processing_days_zscore',
        'desc_length_zscore',
        'is_round_number',
        'is_weekend_payment',
        'is_month_end',
        'is_individual_payee',
        'near_threshold',
        'same_amount_vendor_irregular',
        'is_duplicate',
        'is_reversal',
        'is_split_purchase_risk',
        'is_transposed_amount',
    ]

    print("  Checking feature correlations...")
    ml_features = _prune_correlated(df, ml_features)
    print(f"  Using {len(ml_features)} features for ML models.")
    return df, ml_features
