import pandas as pd
import numpy as np

AMOUNT_COL = 'Payment Voucher Amount (SGD, Excluding GST)'

WEIGHTS = {
    'if_score':          0.30,
    'lof_score':         0.25,
    'zscore_score':      0.25,
    'rule_flags_score':  0.15,
    'benford_score':     0.05,
}

FLAG_COLS = [
    'is_round_number',
    'is_sg_nonworkday',
    'is_month_end',
    'near_threshold',
    'is_individual_payee',
    'same_amount_vendor_irregular',
]


def _rule_flags_score(df):
    present = [c for c in FLAG_COLS if c in df.columns]
    if not present:
        return pd.Series(0.0, index=df.index)
    return df[present].clip(0, 1).sum(axis=1) / len(present)


def _benford_score_normalised(df):
    if 'benford_deviation_score' not in df.columns:
        return pd.Series(0.0, index=df.index)
    bmax = df['benford_deviation_score'].max()
    if bmax <= 0:
        return pd.Series(0.0, index=df.index)
    return (df['benford_deviation_score'] / bmax).clip(0, 1)


def compute_risk_scores(df):
    """Add rule_flags_score, benford_score, and composite risk_score to df."""
    df['rule_flags_score'] = _rule_flags_score(df)
    df['benford_score'] = _benford_score_normalised(df)

    # Benford-only suppression: if all other signals are below their 50th percentile,
    # zero out Benford contribution so it cannot single-handedly select a transaction.
    other_cols = ['if_score', 'lof_score', 'zscore_score', 'rule_flags_score']
    medians = df[other_cols].median()
    all_weak = (df[other_cols] < medians).all(axis=1)
    df.loc[all_weak, 'benford_score'] = 0.0

    df['risk_score'] = (
        df.get('if_score',         pd.Series(0.0, index=df.index)) * WEIGHTS['if_score'] +
        df.get('lof_score',        pd.Series(0.0, index=df.index)) * WEIGHTS['lof_score'] +
        df.get('zscore_score',     pd.Series(0.0, index=df.index)) * WEIGHTS['zscore_score'] +
        df['rule_flags_score']                                       * WEIGHTS['rule_flags_score'] +
        df['benford_score']                                          * WEIGHTS['benford_score']
    )
    return df


def _build_reason(row):
    parts = []

    if row.get('is_individual_payee', 0):
        parts.append("Payment to individual (NRIC/FIN payee)")

    az_v = abs(row.get('amount_zscore_vendor', 0))
    if az_v > 2.0:
        parts.append(f"Amount {az_v:.1f} std devs from vendor average")

    az_c = abs(row.get('amount_zscore_costcentre', 0))
    if az_c > 2.0:
        parts.append(f"Amount {az_c:.1f} std devs from cost centre average")

    if row.get('is_round_number', 0):
        amt = row.get(AMOUNT_COL, '')
        parts.append(f"Round number amount (SGD {amt:,.0f})" if isinstance(amt, (int, float)) else "Round number amount")

    if row.get('is_sg_nonworkday', 0):
        parts.append("Transaction on non-working day (weekend/public holiday)")

    if row.get('is_month_end', 0):
        parts.append("Month-end transaction (last 3 days of month)")

    if row.get('near_threshold', 0):
        parts.append("Amount just below approval threshold")

    if row.get('same_amount_vendor_irregular', 0):
        parts.append("Repeated amount for same vendor (irregular schedule)")

    pt = row.get('processing_days_zscore', 0)
    if pt > 2.5:
        days = row.get('processing_days', None)
        day_str = f" ({int(days)} days)" if days is not None and not pd.isna(days) else ""
        parts.append(f"Unusual processing time{day_str}")

    if row.get('desc_length_zscore', 0) > 2.5:
        parts.append("Unusual description length")

    if row.get('benford_flag', 0) and row.get('benford_score', 0) > 0:
        fd = row.get('benford_first_digit', None)
        d_str = f" (first digit: {int(fd)})" if fd is not None and not pd.isna(fd) else ""
        parts.append(f"Benford's Law deviation{d_str}")

    if not parts:
        # Fallback: high ML scores
        if row.get('if_score', 0) > 0.65:
            parts.append("High Isolation Forest anomaly score")
        if row.get('lof_score', 0) > 0.65:
            parts.append("High local outlier score")
        if not parts:
            parts.append("Elevated composite risk score")

    return "; ".join(parts)


def select_samples(df, n_samples=25):
    """
    Score all transactions and return the top n_samples.

    Returns
    -------
    df_scored   : full dataframe with risk scores (sorted descending)
    selected    : top n_samples rows with 'Selection Reasons' column
    """
    df = compute_risk_scores(df)
    df = df.sort_values('risk_score', ascending=False).reset_index(drop=True)

    n = min(n_samples, len(df))
    selected = df.head(n).copy()
    selected['Selection Reasons'] = selected.apply(_build_reason, axis=1)
    selected.insert(0, 'Sample #', range(1, n + 1))

    print(f"  Selected {n} transactions (risk scores: "
          f"{selected['risk_score'].max():.3f} – {selected['risk_score'].min():.3f}).")
    return df, selected
