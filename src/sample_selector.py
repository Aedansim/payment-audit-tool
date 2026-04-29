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

# Weights for voucher-level rollup
_VCH_W_MAX   = 0.60
_VCH_W_MEAN  = 0.25
_VCH_W_FLAGS = 0.15

FLAG_COLS = [
    'is_round_number',
    'is_sg_nonworkday',
    'is_month_end',
    'near_threshold',
    'is_individual_payee',
    'same_amount_vendor_irregular',
]


# ---------------------------------------------------------------------------
# Line-level scoring (unchanged)
# ---------------------------------------------------------------------------

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
        if row.get('if_anomaly', 0):
            parts.append("High Isolation Forest anomaly score")
        if row.get('lof_anomaly', 0):
            parts.append("High local outlier score")
        if not parts:
            parts.append("Elevated composite risk score")

    return "; ".join(parts)


# ---------------------------------------------------------------------------
# Voucher-level rollup
# ---------------------------------------------------------------------------

def _ml_consensus_flag(df):
    """Count how many of the 3 ML models classify each row as anomalous via predict()."""
    flags = pd.DataFrame({
        'if':  df.get('if_anomaly',     pd.Series(0, index=df.index)),
        'lof': df.get('lof_anomaly',    pd.Series(0, index=df.index)),
        'z':   df.get('zscore_anomaly', pd.Series(0, index=df.index)),
    })
    return flags.sum(axis=1)


def _rollup_vouchers(df):
    """
    Group line-scored rows by Voucher ID and compute voucher-level fields.
    Returns (df_vouchers, df_with_helpers) where df_with_helpers has
    _line_reason and ML_Consensus_Flag columns added.
    """
    df = df.copy()
    df['_line_reason'] = df.apply(_build_reason, axis=1)
    df['ML_Consensus_Flag'] = _ml_consensus_flag(df)

    flag_present = [c for c in FLAG_COLS if c in df.columns]
    n_flags = len(flag_present)

    records = []
    for voucher_id, grp in df.groupby('Voucher ID', sort=False):
        line_count = len(grp)
        flag_count = int(grp[flag_present].clip(0, 1).values.sum()) if flag_present else 0
        total_possible = n_flags * line_count
        flag_density = flag_count / total_possible if total_possible > 0 else 0.0

        max_score  = float(grp['risk_score'].max())
        mean_score = float(grp['risk_score'].mean())

        # Single-line voucher: score equals the line score exactly
        if line_count == 1:
            vch_score = max_score
        else:
            vch_score = (
                _VCH_W_MAX   * max_score +
                _VCH_W_MEAN  * mean_score +
                _VCH_W_FLAGS * flag_density
            )

        # Reason codes: no prefix for single-line vouchers;
        # prefix with Account Code for multi-line so auditor knows which line triggered each reason
        reasons = []
        seen = set()
        for _, row in grp.iterrows():
            for part in row['_line_reason'].split('; '):
                part = part.strip()
                if not part:
                    continue
                if line_count > 1:
                    acct = str(row.get('Account Code', ''))
                    entry = f"[{acct}] {part}" if acct else part
                else:
                    entry = part
                if entry not in seen:
                    seen.add(entry)
                    reasons.append(entry)

        top_line = grp.loc[grp['risk_score'].idxmax()]

        # Collect all distinct, non-blank Invoice Numbers linked to this voucher
        inv_nums = (
            grp['Invoice Number'].astype(str).str.strip()
            .pipe(lambda s: s[~s.isin(['', 'nan', 'NaN', 'None'])])
            .unique().tolist()
        )

        records.append({
            'Voucher ID':               str(voucher_id),
            'Vendor ID':                top_line.get('Vendor ID', ''),
            'Vendor Name':              top_line.get('Vendor Name', ''),
            'Invoice Number(s)':        ', '.join(inv_nums),
            'voucher_line_count':       line_count,
            'voucher_max_score':        round(max_score, 4),
            'voucher_mean_score':       round(mean_score, 4),
            'voucher_flag_count':       flag_count,
            'voucher_any_ml_consensus': int((grp['ML_Consensus_Flag'] >= 2).any()),
            'voucher_score':            round(vch_score, 4),
            'voucher_reason_codes':     ' | '.join(reasons),
        })

    df_vch = pd.DataFrame(records)
    df_vch = df_vch.sort_values('voucher_score', ascending=False).reset_index(drop=True)
    return df_vch, df


def _assign_risk_tier(df_vch):
    """Assign HIGH / MEDIUM / LOW: top 5% = HIGH, next 15% = MEDIUM, rest = LOW."""
    scores = df_vch['voucher_score']
    high_cut = scores.quantile(0.95)
    med_cut  = scores.quantile(0.80)

    def _tier(s):
        if s >= high_cut:
            return 'HIGH'
        elif s >= med_cut:
            return 'MEDIUM'
        return 'LOW'

    df_vch = df_vch.copy()
    df_vch['voucher_risk_tier'] = scores.apply(_tier)
    return df_vch


def _stratified_sample(df_vch, n_samples):
    """All HIGH mandatory (capped at n_samples); proportional MEDIUM (~75% of remainder); random LOW baseline."""
    high   = df_vch[df_vch['voucher_risk_tier'] == 'HIGH']
    medium = df_vch[df_vch['voucher_risk_tier'] == 'MEDIUM']
    low    = df_vch[df_vch['voucher_risk_tier'] == 'LOW']

    # If HIGH alone fills or exceeds the quota, return only the top n_samples from HIGH
    if len(high) >= n_samples:
        return high.head(n_samples).reset_index(drop=True)

    selected = [high]
    remaining = n_samples - len(high)

    if remaining > 0 and len(medium) > 0:
        n_med = min(len(medium), max(1, int(remaining * 0.75)))
        selected.append(medium.head(n_med))
        remaining -= n_med

    if remaining > 0 and len(low) > 0:
        n_low = min(len(low), remaining)
        selected.append(low.sample(n=n_low, random_state=42))

    result = pd.concat(selected, ignore_index=True)
    return result.sort_values('voucher_score', ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def select_samples(df, n_samples=25):
    """
    Score all rows, roll up to payment voucher level, and select the top n_samples vouchers.

    Parameters
    ----------
    n_samples : int, default 25
        Number of vouchers to include in the audit sample. Must be a positive integer.
        Change SAMPLE_SIZE in the notebook's Step 1 cell to override.

    Returns
    -------
    df_scored        : full row-level dataframe with risk_score and helper columns
    df_vouchers      : voucher-level rollup, all vouchers sorted by voucher_score desc
    selected_vouchers: top n_samples vouchers with Sample #, Sample_Rationale columns
    """
    if not isinstance(n_samples, int) or n_samples < 1:
        raise ValueError(
            f"n_samples must be a positive integer (got {n_samples!r}). "
            "Set SAMPLE_SIZE to a whole number ≥ 1 in the notebook's Step 1 cell."
        )
    df = compute_risk_scores(df)
    df = df.sort_values('risk_score', ascending=False).reset_index(drop=True)

    print("  Rolling up to payment voucher level...")
    df_vouchers, df = _rollup_vouchers(df)
    df_vouchers = _assign_risk_tier(df_vouchers)

    n = min(n_samples, len(df_vouchers))
    selected = _stratified_sample(df_vouchers, n)
    selected = selected.copy()
    selected.insert(0, 'Sample #', range(1, len(selected) + 1))
    selected['Sample_Rationale'] = selected['voucher_risk_tier'].map({
        'HIGH':   'Mandatory — top 5% voucher risk score',
        'MEDIUM': 'Proportional selection — elevated risk tier',
        'LOW':    'Baseline — random selection from lower-risk vouchers',
    })

    n_high = int((df_vouchers['voucher_risk_tier'] == 'HIGH').sum())
    n_med  = int((df_vouchers['voucher_risk_tier'] == 'MEDIUM').sum())
    n_low  = int((df_vouchers['voucher_risk_tier'] == 'LOW').sum())
    print(f"  {len(df_vouchers):,} payment vouchers from {len(df):,} line items "
          f"(HIGH: {n_high}, MEDIUM: {n_med}, LOW: {n_low}).")
    print(f"  Selected {len(selected)} vouchers "
          f"(scores: {selected['voucher_score'].max():.3f} – "
          f"{selected['voucher_score'].min():.3f}).")

    return df, df_vouchers, selected
