"""
Accuracy benchmark for the Payment Audit Tool.
Generates 530 synthetic transactions (500 normal + 30 injected anomalies, 5 per type)
and measures how many anomalies are recovered in the top-25 invoice selection.
"""
import sys, warnings, io
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

AMOUNT_COL = 'Payment Voucher Amount (SGD, Excluding GST)'
RNG = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

N_NORMAL   = 500
N_ANOMALY  = 5          # per anomaly type
ANOMALY_TYPES = [
    'individual_payee',
    'near_threshold',
    'round_number',
    'high_amount',
    'month_end',
    'weekend_date',
]

VENDORS = [f"VENDOR_{i:03d}" for i in range(1, 51)]
VENDOR_IDS = [f"V{i:05d}" for i in range(1, 51)]
COST_CENTRES = [f"CC{i:02d}" for i in range(1, 11)]
ACCOUNT_CODES = [f"AC{i:03d}" for i in range(1, 6)]

# Singapore public holidays 2023-2024 (a handful for realism)
SG_HOLIDAYS = pd.to_datetime([
    '2023-01-22', '2023-04-07', '2023-05-01', '2023-06-02',
    '2023-08-09', '2023-11-13', '2023-12-25',
    '2024-01-01', '2024-02-10', '2024-04-10', '2024-05-01',
    '2024-08-09', '2024-10-31', '2024-12-25',
])


def _make_normal_amount():
    """Log-normal amounts roughly in the $500–$50,000 range."""
    return round(float(np.exp(RNG.normal(8.5, 1.2))), 2)


def _workday(date):
    """Return date if it's a workday, else advance to next Monday."""
    d = pd.Timestamp(date)
    while d.weekday() >= 5 or d in SG_HOLIDAYS:
        d += pd.Timedelta(days=1)
    return d


def _random_workday(start='2023-01-01', end='2024-12-31'):
    days = (pd.Timestamp(end) - pd.Timestamp(start)).days
    d = pd.Timestamp(start) + pd.Timedelta(days=int(RNG.integers(0, days)))
    return _workday(d)


def _build_row(i, vendor_idx, amount, invoice_date, anomaly_type=None, voucher_date=None):
    inv_date = pd.Timestamp(invoice_date)
    acc_date = pd.Timestamp(voucher_date) if voucher_date is not None else inv_date + pd.Timedelta(days=int(RNG.integers(1, 15)))
    vendor_name = VENDORS[vendor_idx]
    vendor_id   = VENDOR_IDS[vendor_idx]

    if anomaly_type == 'individual_payee':
        # Valid Singapore NRIC/FIN format
        vendor_id = f"S{RNG.integers(1000000, 9999999)}A"
        vendor_name = f"INDIV_{i}"

    return {
        'Vendor Name':               vendor_name,
        'Vendor ID':                 vendor_id,
        'Cost Centre':               COST_CENTRES[vendor_idx % len(COST_CENTRES)],
        'Account Code':              ACCOUNT_CODES[vendor_idx % len(ACCOUNT_CODES)],
        'Invoice Date':              inv_date,
        'Voucher Accounting Date':   acc_date,
        'Invoice Number':            f"INV{i:05d}",
        'Voucher ID':                f"VCH{i:05d}",
        'Voucher Line Description':  f"Payment for services rendered - ref {i}",
        AMOUNT_COL:                  amount,
        '_anomaly_type':             anomaly_type or 'normal',
    }


def generate_dataset():
    rows = []

    # --- Normal transactions ---
    for i in range(N_NORMAL):
        v_idx   = int(RNG.integers(0, len(VENDORS)))
        amount  = _make_normal_amount()
        inv_date = _random_workday()
        rows.append(_build_row(i, v_idx, amount, inv_date))

    base = N_NORMAL

    # --- Anomaly 1: individual payee ---
    for j in range(N_ANOMALY):
        amount   = _make_normal_amount()
        inv_date = _random_workday()
        rows.append(_build_row(base + j, 0, amount, inv_date, 'individual_payee'))
    base += N_ANOMALY

    # --- Anomaly 2: near threshold ---
    THRESHOLDS = [1_000, 5_000, 10_000, 50_000, 100_000]
    for j in range(N_ANOMALY):
        t = THRESHOLDS[j % len(THRESHOLDS)]
        amount = round(t * RNG.uniform(0.951, 0.999), 2)
        inv_date = _random_workday()
        rows.append(_build_row(base + j, j + 1, amount, inv_date, 'near_threshold'))
    base += N_ANOMALY

    # --- Anomaly 3: round number ---
    for j in range(N_ANOMALY):
        amount = float(int(RNG.integers(10, 500)) * 100)
        inv_date = _random_workday()
        rows.append(_build_row(base + j, j + 2, amount, inv_date, 'round_number'))
    base += N_ANOMALY

    # --- Anomaly 4: high amount (>4× vendor mean to force a strong z-score) ---
    for j in range(N_ANOMALY):
        amount = round(float(RNG.uniform(80_000, 250_000)), 2)
        inv_date = _random_workday()
        rows.append(_build_row(base + j, j + 3, amount, inv_date, 'high_amount'))
    base += N_ANOMALY

    # --- Anomaly 5: month-end (last 3 days of month, voucher accounting date) ---
    for j in range(N_ANOMALY):
        month_start = pd.Timestamp('2023-01-01') + pd.DateOffset(months=j * 2)
        last_day    = (month_start + pd.DateOffset(months=1) - pd.Timedelta(days=1)).day
        day = last_day - int(RNG.integers(0, 3))
        inv_date = pd.Timestamp(month_start.year, month_start.month, day)
        amount   = _make_normal_amount()
        # Pass inv_date as voucher_date so Voucher Accounting Date is also month-end
        rows.append(_build_row(base + j, j + 4, amount, inv_date, 'month_end', voucher_date=inv_date))
    base += N_ANOMALY

    # --- Anomaly 6: weekend date ---
    for j in range(N_ANOMALY):
        # Find a Saturday in 2023-2024
        d = pd.Timestamp('2023-01-07') + pd.Timedelta(weeks=j * 12)  # guaranteed Saturday
        assert d.weekday() == 5, f"Expected Saturday, got weekday {d.weekday()}"
        amount = _make_normal_amount()
        rows.append(_build_row(base + j, j + 5, amount, d, 'weekend_date'))

    df = pd.DataFrame(rows).reset_index(drop=True)
    df['Invoice Date']            = pd.to_datetime(df['Invoice Date'])
    df['Voucher Accounting Date'] = pd.to_datetime(df['Voucher Accounting Date'])
    return df


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(df_raw):
    from src.feature_engineering import engineer_features
    from src import benfords_law
    from src.ml_models import run_ensemble
    from src.sample_selector import select_samples

    df, ml_features = engineer_features(df_raw.drop(columns=['_anomaly_type']).copy())
    df, _, _        = benfords_law.analyze(df)
    df              = run_ensemble(df, ml_features)
    df_scored, df_vouchers, selected_vouchers = select_samples(df, n_samples=25)

    # Restore anomaly labels by matching Voucher ID (each anomaly is its own single-line voucher)
    label_map = df_raw.set_index('Voucher ID')['_anomaly_type'].to_dict()
    df_vouchers['_anomaly_type'] = df_vouchers['Voucher ID'].map(label_map).fillna('normal')
    selected_vouchers['_anomaly_type'] = selected_vouchers['Voucher ID'].map(label_map).fillna('normal')

    return df_vouchers, selected_vouchers


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(df_vouchers, selected_vouchers):
    anomaly_mask = df_vouchers['_anomaly_type'] != 'normal'
    all_anomalies = df_vouchers[anomaly_mask]
    detected = selected_vouchers[selected_vouchers['_anomaly_type'] != 'normal']

    n_anomalies  = len(all_anomalies)
    n_selected   = len(selected_vouchers)
    n_detected   = len(detected)

    recall    = n_detected / n_anomalies if n_anomalies > 0 else 0
    precision = n_detected / n_selected  if n_selected  > 0 else 0

    # Cohen's d: separation between anomaly and normal voucher scores
    anom_scores   = all_anomalies['voucher_score']
    normal_scores = df_vouchers[~anomaly_mask]['voucher_score']
    pooled_std = np.sqrt(
        (anom_scores.std(ddof=1)**2 + normal_scores.std(ddof=1)**2) / 2
    )
    cohens_d = (anom_scores.mean() - normal_scores.mean()) / pooled_std \
        if pooled_std > 0 else 0

    # Per-type breakdown
    per_type = []
    for atype in ANOMALY_TYPES:
        grp = all_anomalies[all_anomalies['_anomaly_type'] == atype]
        in_top25 = len(selected_vouchers[selected_vouchers['_anomaly_type'] == atype])
        avg_score = grp['voucher_score'].mean() if len(grp) > 0 else 0

        all_scores = df_vouchers['voucher_score'].values
        pcts = [int(np.mean(all_scores <= s) * 100) for s in grp['voucher_score']]
        avg_pct = int(np.mean(pcts)) if pcts else 0

        per_type.append({
            'Anomaly Type':   atype,
            'In Top 25':      f"{in_top25}/{len(grp)}",
            'Avg Score':      round(avg_score, 3),
            'Score Pctile':   f"{avg_pct}th",
        })

    return {
        'n_anomalies':  n_anomalies,
        'n_selected':   n_selected,
        'n_detected':   n_detected,
        'recall':       recall,
        'precision':    precision,
        'cohens_d':     cohens_d,
        'per_type':     per_type,
    }


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_results(metrics, df_vouchers, selected_vouchers):
    sep = "=" * 62

    print()
    print(sep)
    print("  ACCURACY BENCHMARK  —  Payment Audit Tool")
    print(f"  Dataset: {len(df_vouchers):,} payment vouchers  "
          f"({metrics['n_anomalies']} injected anomalies, "
          f"{len(df_vouchers) - metrics['n_anomalies']} normal)")
    print(sep)

    print()
    print("  Overall Results")
    print(f"  {'Recall':<30} {metrics['n_detected']}/{metrics['n_anomalies']}  "
          f"({metrics['recall']*100:.1f}%)")
    print(f"  {'Precision':<30} {metrics['n_detected']}/{metrics['n_selected']}  "
          f"({metrics['precision']*100:.1f}%)")
    print(f"  {'Cohen\'s d (score separation)':<30} {metrics['cohens_d']:.2f}")

    print()
    print("  Per-Anomaly-Type Breakdown")
    print(f"  {'Anomaly Type':<28} {'In Top 25':<12} {'Avg Score':<12} {'Score Pctile'}")
    print("  " + "-" * 58)
    for row in metrics['per_type']:
        print(f"  {row['Anomaly Type']:<28} {row['In Top 25']:<12} "
              f"{row['Avg Score']:<12} {row['Score Pctile']}")

    print()
    print("  Voucher-Level Selection Summary")
    tier_counts = selected_vouchers['voucher_risk_tier'].value_counts() \
        if 'voucher_risk_tier' in selected_vouchers.columns else {}
    for tier in ['HIGH', 'MEDIUM', 'LOW']:
        n = tier_counts.get(tier, 0)
        print(f"  {'  ' + tier + ' tier':<30} {n} vouchers selected")

    print()
    print("  Score Distribution (all vouchers)")
    scores = df_vouchers['voucher_score']
    print(f"  {'  Mean':<30} {scores.mean():.4f}")
    print(f"  {'  Std dev':<30} {scores.std():.4f}")
    print(f"  {'  Min':<30} {scores.min():.4f}")
    print(f"  {'  Max':<30} {scores.max():.4f}")
    print(f"  {'  95th pctile (HIGH cutoff)':<30} {scores.quantile(0.95):.4f}")
    print(f"  {'  80th pctile (MED cutoff)':<30} {scores.quantile(0.80):.4f}")
    print(f"  {'  Cutoff (lowest selected)':<30} {selected_vouchers['voucher_score'].min():.4f}")

    print()
    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("Generating synthetic dataset...")
    df_raw = generate_dataset()
    print(f"  {len(df_raw)} total rows "
          f"({(df_raw['_anomaly_type'] == 'normal').sum()} normal, "
          f"{(df_raw['_anomaly_type'] != 'normal').sum()} anomalies)\n")

    # Suppress pipeline output for clean display
    print("Running pipeline (suppressing step-by-step output)...")
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        df_vouchers, selected_vouchers = run_pipeline(df_raw)
    finally:
        sys.stdout = old_stdout

    metrics = compute_metrics(df_vouchers, selected_vouchers)
    print_results(metrics, df_vouchers, selected_vouchers)
