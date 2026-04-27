"""
benchmark_comparison.py

Side-by-side accuracy comparison of two pipeline configurations.
No src/ files are modified — modified logic is defined inline here.

BASELINE  — current behaviour:
            · ML_Consensus_Flag uses 0.65 threshold on normalised IF/LOF scores
            · voucher_score = 0.60×max + 0.25×mean + 0.15×flag_density
            · single-line vouchers: voucher_score = risk_score exactly

MODIFIED  — proposed behaviour:
            · ML_Consensus_Flag uses sklearn predict() at contamination=0.05
              (top 5% per model); z-score anomaly uses abs > 2.0 (2σ boundary)
            · voucher_score = 0.55×max + 0.22×mean + 0.13×flag_density + 0.10×ml_consensus
            · formula applied uniformly to all vouchers (incl. single-line)

Run with:  python benchmark_comparison.py
"""
import sys, warnings, io
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler

# Reuse dataset generation and metrics from benchmark.py
from benchmark import generate_dataset, compute_metrics, ANOMALY_TYPES

AMOUNT_COL = 'Payment Voucher Amount (SGD, Excluding GST)'


# ---------------------------------------------------------------------------
# Shared silence helper
# ---------------------------------------------------------------------------

class _Silence:
    """Context manager that swallows stdout."""
    def __enter__(self):
        self._old = sys.stdout
        sys.stdout = io.StringIO()
    def __exit__(self, *_):
        sys.stdout = self._old


# ---------------------------------------------------------------------------
# BASELINE pipeline  (uses src/ modules unmodified)
# ---------------------------------------------------------------------------

def run_pipeline_baseline(df_raw):
    from src.feature_engineering import engineer_features
    from src import benfords_law
    from src.ml_models import run_ensemble
    from src.sample_selector import select_samples

    df, ml_feats    = engineer_features(df_raw.drop(columns=['_anomaly_type']).copy())
    df, _, _        = benfords_law.analyze(df)
    df              = run_ensemble(df, ml_feats)
    _, df_vch, sel  = select_samples(df, n_samples=25)

    label_map = df_raw.set_index('Voucher ID')['_anomaly_type'].to_dict()
    df_vch['_anomaly_type'] = df_vch['Voucher ID'].map(label_map).fillna('normal')
    sel['_anomaly_type']    = sel['Voucher ID'].map(label_map).fillna('normal')
    return df_vch, sel


# ---------------------------------------------------------------------------
# MODIFIED run_ensemble  (adds binary anomaly flags via predict())
# ---------------------------------------------------------------------------

def _normalise(arr):
    lo, hi = arr.min(), arr.max()
    if hi == lo:
        return np.zeros_like(arr, dtype=float)
    return (arr - lo) / (hi - lo)


def run_ensemble_modified(df, ml_features, random_state=42):
    """
    Same as src/ml_models.run_ensemble but additionally stores:
      if_anomaly      — 1 if IsolationForest.predict() == -1  (top 5%)
      lof_anomaly     — 1 if LOF.fit_predict() == -1          (top 5%)
      zscore_anomaly  — 1 if max(|vendor_z|, |cc_z|) > 2.0   (2σ boundary)
    Normalised *_score columns are unchanged so risk_score formula is unaffected.
    """
    X = df[ml_features].fillna(0).values
    X_scaled = RobustScaler().fit_transform(X)

    # Isolation Forest
    iso = IsolationForest(n_estimators=300, contamination=0.05,
                          max_samples='auto', random_state=random_state, n_jobs=-1)
    iso.fit(X_scaled)
    df['if_score']   = _normalise(-iso.score_samples(X_scaled))
    df['if_anomaly'] = (iso.predict(X_scaled) == -1).astype(int)

    # Local Outlier Factor
    n_neighbors = min(20, len(df) - 1)
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=0.05,
                             algorithm='ball_tree', n_jobs=-1)
    lof_pred = lof.fit_predict(X_scaled)
    df['lof_score']   = _normalise(-lof.negative_outlier_factor_)
    df['lof_anomaly'] = (lof_pred == -1).astype(int)

    # Z-score signal
    z_cols = [c for c in ['amount_zscore_vendor', 'amount_zscore_costcentre']
              if c in df.columns]
    if z_cols:
        z_max = df[z_cols].abs().max(axis=1)
    elif 'amount_zscore_overall' in df.columns:
        z_max = df['amount_zscore_overall'].abs()
    else:
        z_max = pd.Series(0.0, index=df.index)

    df['zscore_score']   = _normalise(z_max.values)
    df['zscore_anomaly'] = (z_max > 2.0).astype(int)

    return df


# ---------------------------------------------------------------------------
# MODIFIED sample_selector helpers
# ---------------------------------------------------------------------------

# Import unchanged helpers from src so we don't duplicate unchanged logic.
from src.sample_selector import (
    FLAG_COLS,
    compute_risk_scores,   # risk_score formula unchanged
    _assign_risk_tier,     # percentile tiers unchanged
    _stratified_sample,    # HIGH/MEDIUM/LOW selection unchanged
    _build_reason,         # reason text unchanged
)

_MOD_W_MAX   = 0.55
_MOD_W_MEAN  = 0.22
_MOD_W_FLAGS = 0.13
_MOD_W_MLCON = 0.10   # new: ML consensus bonus


def _ml_consensus_flag_modified(df):
    """Binary predict() flags — top 5% per model, 2σ for z-score."""
    flags = pd.DataFrame({
        'if':  df.get('if_anomaly',    pd.Series(0, index=df.index)),
        'lof': df.get('lof_anomaly',   pd.Series(0, index=df.index)),
        'z':   df.get('zscore_anomaly', pd.Series(0, index=df.index)),
    })
    return flags.sum(axis=1)


def _rollup_vouchers_modified(df):
    df = df.copy()
    df['_line_reason']      = df.apply(_build_reason, axis=1)
    df['ML_Consensus_Flag'] = _ml_consensus_flag_modified(df)

    flag_present = [c for c in FLAG_COLS if c in df.columns]
    n_flags = len(flag_present)

    records = []
    for voucher_id, grp in df.groupby('Voucher ID', sort=False):
        line_count = len(grp)
        flag_count = int(grp[flag_present].clip(0, 1).values.sum()) if flag_present else 0
        total_poss = n_flags * line_count
        flag_density = flag_count / total_poss if total_poss > 0 else 0.0

        max_score    = float(grp['risk_score'].max())
        mean_score   = float(grp['risk_score'].mean())
        ml_consensus = int((grp['ML_Consensus_Flag'] >= 2).any())

        # New formula applied uniformly (incl. single-line vouchers)
        vch_score = (
            _MOD_W_MAX   * max_score    +
            _MOD_W_MEAN  * mean_score   +
            _MOD_W_FLAGS * flag_density +
            _MOD_W_MLCON * ml_consensus
        )

        reasons, seen = [], set()
        for _, row in grp.iterrows():
            for part in row['_line_reason'].split('; '):
                part = part.strip()
                if not part:
                    continue
                entry = (f"[{row.get('Account Code', '')}] {part}"
                         if line_count > 1 else part)
                if entry not in seen:
                    seen.add(entry)
                    reasons.append(entry)

        top_line = grp.loc[grp['risk_score'].idxmax()]
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
            'voucher_any_ml_consensus': ml_consensus,
            'voucher_score':            round(vch_score, 4),
            'voucher_reason_codes':     ' | '.join(reasons),
        })

    df_vch = pd.DataFrame(records).sort_values('voucher_score', ascending=False).reset_index(drop=True)
    return df_vch, df


def select_samples_modified(df, n_samples=25):
    df = compute_risk_scores(df)   # risk_score formula unchanged
    df = df.sort_values('risk_score', ascending=False).reset_index(drop=True)
    df_vch, df = _rollup_vouchers_modified(df)
    df_vch = _assign_risk_tier(df_vch)

    n = min(n_samples, len(df_vch))
    selected = _stratified_sample(df_vch, n).copy()
    selected.insert(0, 'Sample #', range(1, len(selected) + 1))
    selected['Sample_Rationale'] = selected['voucher_risk_tier'].map({
        'HIGH':   'Mandatory — top 5% voucher risk score',
        'MEDIUM': 'Proportional selection — elevated risk tier',
        'LOW':    'Baseline — random selection from lower-risk vouchers',
    })
    return df, df_vch, selected


# ---------------------------------------------------------------------------
# MODIFIED pipeline
# ---------------------------------------------------------------------------

def run_pipeline_modified(df_raw):
    from src.feature_engineering import engineer_features
    from src import benfords_law

    df, ml_feats    = engineer_features(df_raw.drop(columns=['_anomaly_type']).copy())
    df, _, _        = benfords_law.analyze(df)
    df              = run_ensemble_modified(df, ml_feats)
    _, df_vch, sel  = select_samples_modified(df, n_samples=25)

    label_map = df_raw.set_index('Voucher ID')['_anomaly_type'].to_dict()
    df_vch['_anomaly_type'] = df_vch['Voucher ID'].map(label_map).fillna('normal')
    sel['_anomaly_type']    = sel['Voucher ID'].map(label_map).fillna('normal')
    return df_vch, sel


# ---------------------------------------------------------------------------
# Comparison display
# ---------------------------------------------------------------------------

def print_comparison(m_b, m_m, dv_b, dv_m, sel_b, sel_m):
    W = 72
    sep = "=" * W

    print()
    print(sep)
    print("  BENCHMARK COMPARISON  —  Payment Audit Tool")
    print(f"  Dataset: {len(dv_b):,} vouchers, {m_b['n_anomalies']} injected anomalies")
    print(sep)

    print()
    print("  Configurations")
    print("  BASELINE : 0.65 threshold on normalised IF/LOF; no ML consensus in voucher_score")
    print("  MODIFIED : predict() top-5% flags; +0.10 ML consensus weight in voucher_score")

    # Overall metrics
    print()
    print(f"  {'Metric':<36} {'BASELINE':>16} {'MODIFIED':>16}")
    print("  " + "-" * 68)

    def _row(label, b, m):
        print(f"  {label:<36} {str(b):>16} {str(m):>16}")

    _row("Recall",
         f"{m_b['n_detected']}/{m_b['n_anomalies']} ({m_b['recall']*100:.1f}%)",
         f"{m_m['n_detected']}/{m_m['n_anomalies']} ({m_m['recall']*100:.1f}%)")
    _row("Precision",
         f"{m_b['n_detected']}/{m_b['n_selected']} ({m_b['precision']*100:.1f}%)",
         f"{m_m['n_detected']}/{m_m['n_selected']} ({m_m['precision']*100:.1f}%)")
    _row("Cohen's d (score separation)",
         f"{m_b['cohens_d']:.3f}",
         f"{m_m['cohens_d']:.3f}")

    # Per-type
    print()
    print(f"  {'Anomaly Type':<24} {'BASELINE':>10} {'MODIFIED':>10}  "
          f"{'Avg Score':>16}  {'Score Pctile':>14}")
    print("  " + "-" * 76)
    for b_r, m_r in zip(m_b['per_type'], m_m['per_type']):
        delta_score = m_r['Avg Score'] - b_r['Avg Score']
        sign = "+" if delta_score >= 0 else ""
        print(f"  {b_r['Anomaly Type']:<24} {b_r['In Top 25']:>10} {m_r['In Top 25']:>10}  "
              f"  {b_r['Avg Score']:.3f}->{m_r['Avg Score']:.3f} ({sign}{delta_score:.3f})  "
              f"{b_r['Score Pctile']:>6}->{m_r['Score Pctile']:>6}")

    # ML consensus flag distribution
    print()
    print("  ML Consensus Flag  (vouchers where 2+ models independently agree)")
    for label, dv in [("BASELINE", dv_b), ("MODIFIED", dv_m)]:
        if 'voucher_any_ml_consensus' in dv.columns:
            n   = int(dv['voucher_any_ml_consensus'].sum())
            pct = n / len(dv) * 100
            print(f"  {label:<12} {n:>4} / {len(dv)} vouchers flagged  ({pct:.1f}%)")

    # Score distribution
    print()
    print(f"  {'Score statistic':<36} {'BASELINE':>16} {'MODIFIED':>16}")
    print("  " + "-" * 68)
    stats = [
        ("Max voucher_score",          dv_b['voucher_score'].max(),            dv_m['voucher_score'].max()),
        ("Mean voucher_score",         dv_b['voucher_score'].mean(),           dv_m['voucher_score'].mean()),
        ("95th pctile (HIGH cutoff)",  dv_b['voucher_score'].quantile(0.95),   dv_m['voucher_score'].quantile(0.95)),
        ("80th pctile (MED cutoff)",   dv_b['voucher_score'].quantile(0.80),   dv_m['voucher_score'].quantile(0.80)),
        ("Lowest selected score",      sel_b['voucher_score'].min(),           sel_m['voucher_score'].min()),
    ]
    for label, bv, mv in stats:
        print(f"  {label:<36} {bv:>16.4f} {mv:>16.4f}")

    # Risk tier breakdown in selection
    print()
    print("  Tier breakdown in selected 25 vouchers")
    for tier in ['HIGH', 'MEDIUM', 'LOW']:
        nb = (sel_b['voucher_risk_tier'] == tier).sum() if 'voucher_risk_tier' in sel_b.columns else 0
        nm = (sel_m['voucher_risk_tier'] == tier).sum() if 'voucher_risk_tier' in sel_m.columns else 0
        print(f"  {tier:<12} BASELINE {nb:>3}   MODIFIED {nm:>3}")

    print()
    print(sep)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("Generating synthetic dataset...")
    df_raw = generate_dataset()
    n_norm = (df_raw['_anomaly_type'] == 'normal').sum()
    n_anom = (df_raw['_anomaly_type'] != 'normal').sum()
    print(f"  {len(df_raw)} rows  ({n_norm} normal, {n_anom} anomalies)\n")

    print("Running BASELINE pipeline...")
    with _Silence():
        dv_b, sel_b = run_pipeline_baseline(df_raw)
    m_b = compute_metrics(dv_b, sel_b)
    print("  Done.\n")

    print("Running MODIFIED pipeline...")
    with _Silence():
        dv_m, sel_m = run_pipeline_modified(df_raw)
    m_m = compute_metrics(dv_m, sel_m)
    print("  Done.")

    print_comparison(m_b, m_m, dv_b, dv_m, sel_b, sel_m)
