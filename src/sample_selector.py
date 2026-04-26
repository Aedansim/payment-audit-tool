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

# Weights for invoice-level rollup
_INV_W_MAX   = 0.60
_INV_W_MEAN  = 0.25
_INV_W_FLAGS = 0.15

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
        if row.get('if_score', 0) > 0.65:
            parts.append("High Isolation Forest anomaly score")
        if row.get('lof_score', 0) > 0.65:
            parts.append("High local outlier score")
        if not parts:
            parts.append("Elevated composite risk score")

    return "; ".join(parts)


# ---------------------------------------------------------------------------
# Invoice-level rollup
# ---------------------------------------------------------------------------

def _ml_consensus_flag(df):
    """Count how many of the 3 ML models flag each row as anomalous (score > 0.65)."""
    flags = pd.DataFrame({
        'if':  (df.get('if_score',     pd.Series(0.0, index=df.index)) > 0.65).astype(int),
        'lof': (df.get('lof_score',    pd.Series(0.0, index=df.index)) > 0.65).astype(int),
        'z':   (df.get('zscore_score', pd.Series(0.0, index=df.index)) > 0.65).astype(int),
    })
    return flags.sum(axis=1)


def _make_invoice_key(df):
    """Vendor ID + Invoice Number; fall back to Voucher ID when Invoice Number is blank."""
    inv_num = df['Invoice Number'].astype(str).str.strip()
    blank = inv_num.isin(['', 'nan', 'NaN', 'None'])
    key = df['Vendor ID'].astype(str) + '||' + inv_num
    key[blank] = '__VOUCHER__' + df.loc[blank, 'Voucher ID'].astype(str)
    return key


def _rollup_invoices(df):
    """
    Group line-scored rows by invoice and compute invoice-level fields.
    Returns (df_invoices, df_with_keys) where df_with_keys has _invoice_key,
    _line_reason, and ML_Consensus_Flag columns added.
    """
    df = df.copy()
    df['_invoice_key'] = _make_invoice_key(df)
    df['_line_reason'] = df.apply(_build_reason, axis=1)
    df['ML_Consensus_Flag'] = _ml_consensus_flag(df)

    flag_present = [c for c in FLAG_COLS if c in df.columns]
    n_flags = len(flag_present)

    records = []
    for key, grp in df.groupby('_invoice_key', sort=False):
        line_count = len(grp)
        flag_count = int(grp[flag_present].clip(0, 1).values.sum()) if flag_present else 0
        total_possible = n_flags * line_count
        flag_density = flag_count / total_possible if total_possible > 0 else 0.0

        max_score  = float(grp['risk_score'].max())
        mean_score = float(grp['risk_score'].mean())

        # Single-line invoice: score equals the line score exactly
        if line_count == 1:
            inv_score = max_score
        else:
            inv_score = (
                _INV_W_MAX   * max_score +
                _INV_W_MEAN  * mean_score +
                _INV_W_FLAGS * flag_density
            )

        # Deduplicated reason codes prefixed with Voucher ID so auditor knows which line triggered each
        reasons = []
        seen = set()
        for _, row in grp.iterrows():
            voucher = str(row.get('Voucher ID', ''))
            for part in row['_line_reason'].split('; '):
                part = part.strip()
                if part:
                    entry = f"[{voucher}] {part}"
                    if entry not in seen:
                        seen.add(entry)
                        reasons.append(entry)

        top_line = grp.loc[grp['risk_score'].idxmax()]

        # Use Voucher ID as the invoice identifier when Invoice Number is blank
        inv_num = str(top_line.get('Invoice Number', ''))
        if inv_num.strip() in ('', 'nan', 'NaN', 'None'):
            inv_num = str(top_line.get('Voucher ID', ''))

        records.append({
            '_invoice_key':             key,
            'Vendor ID':                top_line.get('Vendor ID', ''),
            'Vendor Name':              top_line.get('Vendor Name', ''),
            'Invoice Number':           inv_num,
            'invoice_line_count':       line_count,
            'invoice_max_score':        round(max_score, 4),
            'invoice_mean_score':       round(mean_score, 4),
            'invoice_flag_count':       flag_count,
            'invoice_any_ml_consensus': int((grp['ML_Consensus_Flag'] >= 2).any()),
            'invoice_score':            round(inv_score, 4),
            'invoice_reason_codes':     ' | '.join(reasons),
        })

    df_inv = pd.DataFrame(records)
    df_inv = df_inv.sort_values('invoice_score', ascending=False).reset_index(drop=True)
    return df_inv, df


def _assign_risk_tier(df_inv):
    """Assign HIGH / MEDIUM / LOW: top 5% = HIGH, next 15% = MEDIUM, rest = LOW."""
    scores = df_inv['invoice_score']
    high_cut = scores.quantile(0.95)
    med_cut  = scores.quantile(0.80)

    def _tier(s):
        if s >= high_cut:
            return 'HIGH'
        elif s >= med_cut:
            return 'MEDIUM'
        return 'LOW'

    df_inv = df_inv.copy()
    df_inv['invoice_risk_tier'] = scores.apply(_tier)
    return df_inv


def _stratified_sample(df_inv, n_samples):
    """All HIGH mandatory; proportional MEDIUM (~75% of remainder); random LOW baseline."""
    high   = df_inv[df_inv['invoice_risk_tier'] == 'HIGH']
    medium = df_inv[df_inv['invoice_risk_tier'] == 'MEDIUM']
    low    = df_inv[df_inv['invoice_risk_tier'] == 'LOW']

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
    return result.sort_values('invoice_score', ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def select_samples(df, n_samples=25):
    """
    Score all rows, roll up to invoice level, and return the top n_samples invoices.

    Returns
    -------
    df_scored         : full row-level dataframe with risk_score and helper columns
    df_invoices       : invoice-level rollup, all invoices sorted by invoice_score desc
    selected_invoices : top n_samples invoices with Sample #, Sample_Rationale columns
    """
    df = compute_risk_scores(df)
    df = df.sort_values('risk_score', ascending=False).reset_index(drop=True)

    print("  Rolling up to invoice level...")
    df_invoices, df = _rollup_invoices(df)
    df_invoices = _assign_risk_tier(df_invoices)

    n = min(n_samples, len(df_invoices))
    selected = _stratified_sample(df_invoices, n)
    selected = selected.copy()
    selected.insert(0, 'Sample #', range(1, len(selected) + 1))
    selected['Sample_Rationale'] = selected['invoice_risk_tier'].map({
        'HIGH':   'Mandatory — top 5% invoice risk score',
        'MEDIUM': 'Proportional selection — elevated risk tier',
        'LOW':    'Baseline — random selection from lower-risk invoices',
    })

    n_high = int((df_invoices['invoice_risk_tier'] == 'HIGH').sum())
    n_med  = int((df_invoices['invoice_risk_tier'] == 'MEDIUM').sum())
    n_low  = int((df_invoices['invoice_risk_tier'] == 'LOW').sum())
    print(f"  {len(df_invoices):,} invoices from {len(df):,} line items "
          f"(HIGH: {n_high}, MEDIUM: {n_med}, LOW: {n_low}).")
    print(f"  Selected {len(selected)} invoices "
          f"(scores: {selected['invoice_score'].max():.3f} – "
          f"{selected['invoice_score'].min():.3f}).")

    return df, df_invoices, selected
