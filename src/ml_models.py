import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler


def _normalise(arr):
    """Scale array to [0, 1]. Handles constant arrays."""
    lo, hi = arr.min(), arr.max()
    if hi == lo:
        return np.zeros_like(arr, dtype=float)
    return (arr - lo) / (hi - lo)


def run_ensemble(df, ml_features, random_state=42):
    """
    Fit Isolation Forest, LOF, and statistical z-score on the feature matrix.

    Adds columns to df:
      if_score        — Isolation Forest anomaly score (0=normal, 1=anomalous)
      if_anomaly      — 1 if IsolationForest.predict() == -1  (top 5% boundary)
      lof_score       — Local Outlier Factor score (0=normal, 1=anomalous)
      lof_anomaly     — 1 if LOF.fit_predict() == -1          (top 5% boundary)
      zscore_score    — Statistical z-score signal (0=normal, 1=anomalous)
      zscore_anomaly  — 1 if max(|vendor_z|, |cc_z|) > 2.0   (2σ boundary)

    Returns df with score columns added.
    """
    X = df[ml_features].fillna(0).values
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    # --- Isolation Forest ---
    print("  Running Isolation Forest...")
    iso = IsolationForest(
        n_estimators=100,
        contamination=0.05,
        max_samples='auto',
        random_state=random_state,
        n_jobs=1,
    )
    iso.fit(X_scaled)
    # score_samples returns negative values; more negative = more anomalous
    iso_raw = iso.score_samples(X_scaled)
    df['if_score']   = _normalise(-iso_raw)  # flip sign so higher = more anomalous
    df['if_anomaly'] = (iso.predict(X_scaled) == -1).astype(int)

    # --- Local Outlier Factor ---
    print("  Running Local Outlier Factor...")
    n_neighbors = min(20, len(df) - 1)
    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=0.05,
        algorithm='ball_tree',
        n_jobs=1,
    )
    lof_pred = lof.fit_predict(X_scaled)
    # negative_outlier_factor_ is negative; more negative = more anomalous
    lof_raw = -lof.negative_outlier_factor_
    df['lof_score']   = _normalise(lof_raw)
    df['lof_anomaly'] = (lof_pred == -1).astype(int)

    # --- Statistical z-score signal ---
    # Take the maximum absolute z-score across vendor and cost-centre dimensions
    z_cols = [c for c in ['amount_zscore_vendor', 'amount_zscore_costcentre']
              if c in df.columns]
    if z_cols:
        z_max = df[z_cols].abs().max(axis=1)
    else:
        z_max = pd.Series(0.0, index=df.index)
    df['zscore_score']   = _normalise(z_max.values)
    df['zscore_anomaly'] = (z_max > 2.0).astype(int)

    print("  ML scoring complete.")
    return df
