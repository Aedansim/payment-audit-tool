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

def _group_zscore(values, group):
    """Within-group z-score of `values` (a Series) grouped by `group` (a Series).
    Callers pass df[AMOUNT_COL].abs() so reversals are assessed by magnitude, not sign."""
    def _z(x):
        s = x.std(ddof=1)
        return (x - x.mean()) / s if s > 0 else pd.Series(0.0, index=x.index)
    return values.groupby(group).transform(_z).fillna(0.0)


def _is_weekend_payment(date):
    if pd.isna(date):
        return 0
    return 1 if date.date().weekday() >= 5 else 0


def _detect_duplicates(df):
    """Flag rows where the same invoice (vendor + invoice number + invoice date + amount)
    appears in more than one distinct Voucher ID — a potential double payment.
    Returns (is_duplicate Series, duplicate_matched_voucher Series) where
    duplicate_matched_voucher holds the counterpart Voucher ID(s), comma-separated.
    Rows with a missing Invoice Date cannot match on the date key and are not flagged."""
    is_dup = pd.Series(0, index=df.index)
    matched_vch = pd.Series('', index=df.index, dtype=object)
    has_invoice = df['Invoice Number'].notna() & (
        df['Invoice Number'].astype(str).str.strip() != ''
    )
    if not has_invoice.any():
        return is_dup, matched_vch
    relevant = df[has_invoice]
    key = ['Vendor ID', 'Invoice Number', 'Invoice Date', AMOUNT_COL]
    cross_voucher = relevant.groupby(key)['Voucher ID'].transform('nunique') > 1
    flagged_idx = relevant[cross_voucher].index
    is_dup.loc[flagged_idx] = 1
    # For each flagged group, record the OTHER Voucher ID(s) for each row
    flagged_relevant = relevant.loc[flagged_idx]
    for _grp_key, grp in flagged_relevant.groupby(key):
        all_vids = sorted(grp['Voucher ID'].astype(str).unique())
        for idx, row in grp.iterrows():
            own = str(row['Voucher ID'])
            others = [v for v in all_vids if v != own]
            matched_vch.loc[idx] = ', '.join(others)
    return is_dup, matched_vch


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


_SPLIT_LO_BAND = (5_700, 6_000)
_SPLIT_HI_BAND = (85_500, 90_000)


def _detect_split_purchase(df):
    """Flag transactions where the same vendor has 2+ invoices on the same date with
    alphanumerically sequential numeric suffixes AND the group total falls within 5%
    below $6,000 or $90,000 (threshold bands $5,700–<$6,000 or $85,500–<$90,000)."""
    result = pd.Series(0, index=df.index)

    suffixes = df['Invoice Number'].astype(str).str.strip().str.extract(r'(\d+)$', expand=False)
    # 19+ digit suffixes exceed int64 and are not plausible sequential counters — treat as absent
    suffixes = suffixes.where(suffixes.str.len() <= 18)

    work = pd.DataFrame({
        'vid':        df['Vendor ID'],
        'idate':      df['Invoice Date'],
        'has_suffix': suffixes.notna().astype(int),
        'suffix':     suffixes,
        'amount':     df[AMOUNT_COL],
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

    valid['suf_int'] = valid['suffix'].astype(np.int64)
    gs = valid.groupby(['vid', 'idate'])['suf_int']
    cnt = gs.transform('count')
    mn  = gs.transform('min')
    mx  = gs.transform('max')
    nu  = gs.transform('nunique')

    # Condition 4: consecutive integer range — max − min == count − 1 with no duplicates
    is_split = (mx - mn == cnt - 1) & (cnt == nu)

    # Condition 5: group total falls within 5% below $6,000 or $90,000
    valid['grp_sum'] = valid.groupby(['vid', 'idate'])['amount'].transform('sum')
    is_in_band = (
        ((valid['grp_sum'] >= _SPLIT_LO_BAND[0]) & (valid['grp_sum'] < _SPLIT_LO_BAND[1])) |
        ((valid['grp_sum'] >= _SPLIT_HI_BAND[0]) & (valid['grp_sum'] < _SPLIT_HI_BAND[1]))
    )

    result.loc[valid[is_split & is_in_band].index] = 1
    return result


# ---------------------------------------------------------------------------
# FY split purchase detection
# ---------------------------------------------------------------------------
# A fiscal year runs 1 April YYYY – 31 March YYYY+1. A potential split purchase is
# 2+ similar payments to the same vendor within one fiscal year whose positive total
# exceeds the small-value purchase approval threshold. Token lists are module-level so
# they can be extended without changing the logic.

_FY_SPLIT_THRESHOLD = 6_000

_FY_MONTH_TOKENS = frozenset({
    'january', 'jan', 'february', 'feb', 'march', 'mar', 'april', 'apr', 'may',
    'june', 'jun', 'july', 'jul', 'august', 'aug', 'september', 'sep', 'sept',
    'october', 'oct', 'november', 'nov', 'december', 'dec',
})

_FY_FILLER_TOKENS = frozenset({
    'invoice', 'inv', 'no', 'ref', 'reference', 'qty', 'quantity', 'unit', 'units',
    'period', 'dated', 'date', 'for', 'the', 'and', 'of', 'to', 'payment', 'pmt', 'voucher',
})


def _fy_label(date):
    """FY label from a Voucher Accounting Date: month >= 4 -> FY{year}, else FY{year-1}."""
    if pd.isna(date):
        return np.nan
    return f"FY{date.year if date.month >= 4 else date.year - 1}"


def _normalise_fy_desc(desc):
    """Reduce a Voucher Line Description to a meaningful-words key for FY-split grouping.
    Lowercase+strip, drop digits, punctuation->space, drop whole-token month names and
    generic filler words, collapse spaces.

    Fallback (g): when the cleaned key carries no real descriptive content — i.e. it is blank
    (e.g. a purely numeric reference like '12/2024/356/22' -> '') OR only short residual
    fragments survive (e.g. an alphanumeric reference 'PO-4471-22' -> 'po') — fall back to the
    original description cleaned only by strip+lowercase+whitespace-collapse, preserving the
    raw reference (digits and punctuation kept) as the grouping key. This keeps distinct
    reference codes in separate groups instead of merging them (e.g. 'PO-4471-22' vs
    'PO-9999-88'). A token of >= 3 letters is treated as real descriptive content.

    Limitation: a description that is a unique reference per payment forms its own single-item
    group and is never flagged (a group needs 2+ payments) — surfaced for separate manual review."""
    if pd.isna(desc):
        return np.nan
    raw = str(desc)
    s = re.sub(r'[0-9]', '', raw.strip().lower())     # (a)(b)
    s = re.sub(r'[^a-z\s]', ' ', s)                    # (c) punctuation -> space
    tokens = [t for t in s.split()                     # (d)(e) whole-token removal
              if t not in _FY_MONTH_TOKENS and t not in _FY_FILLER_TOKENS]
    key = ' '.join(tokens).strip()                     # (f)
    if key and any(len(t) >= 3 for t in tokens):
        return key
    fallback = re.sub(r'\s+', ' ', raw.strip().lower())  # (g) keep digits/punctuation
    return fallback if fallback else np.nan


def _detect_fy_split_purchase(df, excluded_uens=None):
    """Flag positive-amount payments belonging to a (Vendor ID, fiscal year, normalised
    description) group of 2+ payments whose positive total exceeds SGD 6,000.
    Returns (is_fy_split_purchase, fy_split_group_total, fy_split_group_count,
    fy_split_fy_label) Series aligned to df.index. Deterministic exact-match grouping
    on the cleaned key — not the pairwise similarity used during sample selection.

    Transactions whose Vendor ID is in excluded_uens (UENs from the Excluded vendors
    file, trimmed full-identifier match) are excluded: never flagged and never
    contributing to any group's total or count."""
    excluded_set = {str(u).strip() for u in excluded_uens} if excluded_uens else set()
    is_split    = pd.Series(0, index=df.index)
    grp_total   = pd.Series(0.0, index=df.index)
    grp_count   = pd.Series(0, index=df.index)
    fy_label    = pd.Series('', index=df.index, dtype=object)

    work = pd.DataFrame({
        '_vid':       df['Vendor ID'],
        '_fy':        df['Voucher Accounting Date'].apply(_fy_label),
        '_desc_norm': df['Voucher Line Description'].apply(_normalise_fy_desc),
        '_amount':    df[AMOUNT_COL],
    }, index=df.index)

    eligible = work[
        work['_fy'].notna() & work['_desc_norm'].notna() & (work['_amount'] > 0)
    ]
    if excluded_set:
        is_excl = eligible['_vid'].astype(str).str.strip().isin(excluded_set)
        n_excl = int(is_excl.sum())
        eligible = eligible[~is_excl]
        print(f"  Excluded {n_excl} transaction(s) from FY split purchase detection "
              f"(vendors in Excluded vendors file).")
    n_flagged = n_groups = 0
    if not eligible.empty:
        for (_vid, fy, _desc), grp in eligible.groupby(['_vid', '_fy', '_desc_norm'], sort=False):
            if len(grp) < 2:
                continue
            total = grp['_amount'].sum()
            if total <= _FY_SPLIT_THRESHOLD:
                continue
            idx = grp.index
            is_split.loc[idx]  = 1
            grp_total.loc[idx] = round(float(total), 2)
            grp_count.loc[idx] = len(grp)
            fy_label.loc[idx]  = fy
            n_groups  += 1
            n_flagged += len(grp)

    print(f"  Found {n_flagged:,} transactions across {n_groups:,} vendor-FY-description "
          f"groups with potential FY split purchase (group total > SGD 6,000).")
    return is_split, grp_total, grp_count, fy_label


def _is_digit_transposition(a, b):
    """Return True if two positive amounts differ by exactly one digit-position swap
    in their whole-dollar (integer) portion. Cents are discarded so the swap cannot
    cross the decimal point — e.g. $348.23 vs $328.43 (a trivial $19.80 difference) is
    NOT a transposition, whereas $4,800 vs $8,400 is."""
    a_str = str(int(abs(a)))   # whole dollars only, cents dropped
    b_str = str(int(abs(b)))
    if len(a_str) != len(b_str):
        return False
    diffs = [(ca, cb) for ca, cb in zip(a_str, b_str) if ca != cb]
    return len(diffs) == 2 and diffs[0][0] == diffs[1][1] and diffs[0][1] == diffs[1][0]


def _detect_transposed_amounts(df):
    """Flag transactions where same vendor and description have digit-transposed amounts —
    exactly two digit positions swapped in the whole-dollar portion, suggesting a keying error.
    Returns (is_transposed Series, transposed_matched_invoice Series)."""
    result = pd.Series(0, index=df.index)
    matched_inv = pd.Series('', index=df.index, dtype=object)
    pos_mask = df[AMOUNT_COL] > 0
    if not pos_mask.any():
        return result, matched_inv
    pos = df[pos_mask].copy()
    # Description key is a strict near-exact match (lowercase + strip only — no digit,
    # punctuation, month-name, or filler-word removal) BY DESIGN. Amounts are only compared
    # for transposition within a (Vendor ID, description) group, so a strict key avoids false
    # transposition matches between unrelated payments that merely share a similar description.
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

def engineer_features(df, excluded_uens=None):
    """
    Compute all risk features.
    Returns (df_with_features, ml_feature_names_after_pruning).

    excluded_uens : set of str, optional
        Vendor IDs (UENs) from the 'Excluded vendors.xlsx' file. Their transactions
        are excluded from FY split purchase detection. None → no exclusions.
    """
    print("  Computing amount z-scores...")
    df['amount_log'] = np.log1p(df[AMOUNT_COL].abs())
    df['amount_zscore_vendor'] = _group_zscore(df[AMOUNT_COL].abs(), df['Vendor ID'])
    df['amount_zscore_costcentre'] = _group_zscore(df[AMOUNT_COL].abs(), df['Cost Centre'])

    print("  Computing rule-based flags...")
    df['is_reversal'] = (df[AMOUNT_COL] < 0).astype(int)
    df['is_duplicate'], df['duplicate_matched_voucher'] = _detect_duplicates(df)
    df['is_round_number'] = df[AMOUNT_COL].apply(_round_number)
    df['is_weekend_payment'] = df['Voucher Accounting Date'].apply(_is_weekend_payment)
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

    print("  Detecting potential FY split purchases (same vendor, similar description, same fiscal year)...")
    (df['is_fy_split_purchase'], df['fy_split_group_total'],
     df['fy_split_group_count'], df['fy_split_fy_label']) = _detect_fy_split_purchase(df, excluded_uens)

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
