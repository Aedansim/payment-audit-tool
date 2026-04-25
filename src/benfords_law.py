import numpy as np
import pandas as pd
from scipy.stats import chisquare

AMOUNT_COL = 'Payment Voucher Amount (SGD, Excluding GST)'

# Benford's expected probability for each first digit 1–9
BENFORD_EXPECTED = {d: np.log10(1 + 1 / d) for d in range(1, 10)}


def _first_digit(amount):
    """Extract first significant digit (1–9) from a positive number."""
    if pd.isna(amount) or amount <= 0:
        return np.nan
    s = f"{amount:.10g}".replace('.', '').replace('-', '').lstrip('0')
    return int(s[0]) if s else np.nan


def analyze(df):
    """
    Run Benford's Law analysis on non-recurring transactions.

    Returns
    -------
    df : DataFrame with 'benford_first_digit', 'benford_flag', 'benford_deviation_score' columns
    summary : DataFrame  (digit-level stats for Excel / dashboard)
    stats : dict         (chi2, p_value, mad, conformity, deviant_digits, n_analyzed)
    """
    df = df.copy()
    analysis = df[df['is_recurring_payment'] == 0].copy()

    analysis['_fd'] = analysis[AMOUNT_COL].apply(_first_digit)
    analysis = analysis.dropna(subset=['_fd'])
    analysis['_fd'] = analysis['_fd'].astype(int)

    n = len(analysis)
    digits = range(1, 10)

    observed_counts = analysis['_fd'].value_counts().reindex(digits, fill_value=0)
    expected_counts = pd.Series({d: BENFORD_EXPECTED[d] * n for d in digits})
    observed_pct = observed_counts / n
    expected_pct = pd.Series(BENFORD_EXPECTED)
    deviation = (observed_pct - expected_pct).abs()

    # Chi-square test (add small epsilon to expected to avoid div-by-zero)
    chi2_stat, p_value = chisquare(
        observed_counts.values,
        f_exp=np.maximum(expected_counts.values, 1e-10),
    )

    mad = deviation.mean()
    if mad < 0.006:
        conformity = "Conformity"
    elif mad < 0.012:
        conformity = "Acceptable"
    elif mad < 0.015:
        conformity = "Marginally Acceptable"
    else:
        conformity = "Non-Conformity"

    deviant_digits = deviation.nlargest(3).index.tolist()

    # Build summary table
    summary = pd.DataFrame({
        'First Digit': list(digits),
        'Expected Frequency': [f"{BENFORD_EXPECTED[d]*100:.1f}%" for d in digits],
        'Observed Count': observed_counts.values,
        'Expected Count': [f"{expected_counts[d]:.1f}" for d in digits],
        'Observed Frequency': [f"{observed_pct[d]*100:.1f}%" for d in digits],
        'Absolute Deviation': [f"{deviation[d]*100:.2f}%" for d in digits],
        'Note': ['Most deviant' if d == deviant_digits[0]
                 else ('2nd most deviant' if d == deviant_digits[1]
                       else ('3rd most deviant' if d == deviant_digits[2] else ''))
                 for d in digits],
    })

    # Tag individual transactions
    digit_map = _first_digit  # reuse function
    df['benford_first_digit'] = df[AMOUNT_COL].apply(digit_map)
    deviation_map = deviation.to_dict()

    df['benford_flag'] = (
        (df['is_recurring_payment'] == 0) &
        (df['benford_first_digit'].isin(deviant_digits))
    ).astype(int)

    df['benford_deviation_score'] = (
        df['benford_flag'] *
        df['benford_first_digit'].map(deviation_map).fillna(0)
    )

    stats_out = {
        'chi2': chi2_stat,
        'p_value': p_value,
        'mad': mad,
        'conformity': conformity,
        'deviant_digits': deviant_digits,
        'n_analyzed': n,
        'n_excluded_recurring': int(df['is_recurring_payment'].sum()),
        'observed_counts': observed_counts,
        'expected_pct': expected_pct,
        'observed_pct': observed_pct,
    }

    print(f"  Benford's Law: MAD = {mad:.4f} ({conformity}), "
          f"chi² p-value = {p_value:.4f}")
    print(f"  Most deviant digits: {deviant_digits}")

    return df, summary, stats_out
