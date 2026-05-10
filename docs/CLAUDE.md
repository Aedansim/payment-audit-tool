# docs/ — Benchmark Results & Accuracy History

Read this when running `benchmark.py`, interpreting results, or evaluating scoring changes.

## Running the benchmark

```bash
python benchmark.py                     # current pipeline
python benchmark_comparison.py          # compare two pipeline configs side-by-side
```

`benchmark_comparison.py`: runs both pipelines against the same synthetic dataset, prints recall, precision, Cohen's d, ML consensus flag distribution, score statistics. No `src/` files modified — all modified logic defined inline. Use before adopting proposed changes to scoring formula or ML thresholds.

## Benchmark setup

525 synthetic transactions (500 normal + 25 injected anomalies, 5 per anomaly type), each as its own single-line voucher. Scores at voucher level. `is_month_end` removed — benchmark no longer injects month-end anomalies. `weekend_date` anomalies use `Voucher Accounting Date` on Saturday (matches production flag).

## Current results — May 2026 (14 amendments applied)

| Anomaly Type | In Top 25 | Avg Score | Score Percentile |
|---|---|---|---|
| individual_payee | 5/5 | 0.336 | 97th |
| high_amount | 4/5 | 0.362 | 97th |
| weekend_date | 5/5 | 0.427 | 98th |
| round_number | 2/5 | 0.309 | 95th |
| near_threshold | 1/5 | 0.278 | 94th |

**Recall 68.0% (17/25) · Precision 81.0% (17/21) · Cohen's d = 3.40 (very strong separation)**

## Previous results — May 2026 (pre-amendment)

| Anomaly Type | In Top 25 | Avg Score | Score Percentile |
|---|---|---|---|
| individual_payee | 5/5 | 0.546 | 99th |
| high_amount | 5/5 | 0.381 | 96th |
| month_end | 2/5 | 0.355 | 93rd |
| round_number | 2/5 | 0.349 | 94th |
| near_threshold | 2/5 | 0.333 | 92nd |
| weekend_date | 3/5 | 0.322 | 93rd |

**Recall 63.3% (19/30) · Precision 76.0% (19/25) · Cohen's d = 2.89**

## Previous results — April 2026

| Anomaly Type | In Top 25 | Avg Score | Score Percentile |
|---|---|---|---|
| individual_payee | 5/5 | 0.546 | 99th |
| round_number | 3/5 | 0.448 | 96th |
| high_amount | 2/5 | 0.410 | 94th |
| near_threshold | 2/5 | 0.383 | 91st |
| weekend_date | 1/5 | 0.391 | 93rd |
| month_end | 1/5 | 0.317 | 86th |

**Recall 46.7% (14/30) · Precision 56.0% (14/25) · Cohen's d = 2.83**

## Previous results — April 2025 (line-level selection)

| Anomaly Type | In Top 25 | Avg Score | Score Percentile |
|---|---|---|---|
| individual_payee | 5/5 | 0.358 | 98th |
| near_threshold | 4/5 | 0.454 | 97th |
| round_number | 4/5 | 0.348 | 96th |
| high_amount | 3/5 | 0.521 | 97th |
| month_end | 1/5 | 0.274 | 92nd |
| weekend_date | 0/5 | 0.301 | 94th |

**Recall 56.7% (17/30) · Precision 68% (17/25) · Cohen's d = 2.46**

## Interpreting benchmark vs production

**Cohen's d is the meaningful metric.** Benchmark recall figures are synthetic artefacts.

The benchmark uses single-line vouchers, so line-level and voucher-level selection are equivalent. Gaps across runs reflect different random dataset characteristics rather than real regressions.

**Cohen's d = 3.40 (current)** = very strong separation. In real multi-line-voucher data, recall is expected to be higher: any flagged line elevates the whole voucher.

**Benchmark limitation — similarity deduplication & vendor cap:** all synthetic descriptions share the same template ("Payment for services rendered - ref N") → pairwise Jaccard ≈ 0.75 > 0.70 threshold → filter fires aggressively, some HIGH-tier vouchers dropped without replacement (vendor cap slots unfilled when no dissimilar candidate exists). In production data with varied descriptions, both filters fire only for genuine near-duplicates and the full sample quota is always filled.

**Single-signal anomalies** (weekend date, near threshold) score 94th–98th percentile but may be displaced by stronger multi-signal anomalies. Tool performs best when multiple flags stack on the same transaction.

## Key prior evaluation (documented in benchmark_comparison.py)

Switching `predict()` binary flags into the voucher *scoring formula* (adding 0.10 ML consensus weight to `voucher_score`) worsened Cohen's d 2.834→2.551. This change was rejected. The current implementation uses `predict()` for the ML Consensus Flag display column only — the `risk_score` and `voucher_score` formulas are unchanged.
