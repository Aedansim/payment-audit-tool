# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common commands

```bash
# Install dependencies (run once, or after requirements.txt changes)
pip install -r requirements.txt

# Syntax-check all source modules
python -c "
import py_compile
for f in ['src/data_loader.py','src/feature_engineering.py','src/benfords_law.py',
          'src/ml_models.py','src/sample_selector.py','src/excel_exporter.py',
          'src/report_generator.py']:
    py_compile.compile(f, doraise=True); print('OK', f)
"

# Run the full pipeline against a test file (from the payment_audit_tool/ root)
python -c "
import sys, warnings; sys.path.insert(0,'.'); warnings.filterwarnings('ignore')
from src.data_loader import load_transactions
from src.feature_engineering import engineer_features
from src import benfords_law
from src.ml_models import run_ensemble
from src.sample_selector import select_samples
df = load_transactions('data/<your_file>.xlsx')
df, feats = engineer_features(df)
df, summary, stats = benfords_law.analyze(df)
df = run_ensemble(df, feats)
df_scored, df_vouchers, selected_vouchers = select_samples(df)
print(selected_vouchers[['Sample #','Vendor Name','voucher_score','voucher_reason_codes']].head())
"

# Git workflow
git add src/<changed_file>.py
git commit -m "short imperative description"
git push
```

## Git discipline

Commit and push to GitHub after every meaningful unit of work — a completed feature, a bug fix, a refactor, an output module change. Do not batch multiple unrelated changes into one commit.

**Commit message format:** short imperative subject line (≤ 72 chars), e.g.:
- `fix: handle negative processing_days in zscore calculation`
- `feat: add weekly recurrence cycle to recurring detection`
- `refactor: extract threshold logic into shared constant`

After every commit, always run `git push` immediately so the remote is never behind. The project owner uses GitHub as the authoritative backup and rollback point.

## Architecture

The pipeline is strictly linear — each stage adds columns to the same DataFrame and passes it to the next. All modules live in `src/` and are orchestrated by `Payment_Audit_Tool.ipynb`.

```
load_transactions()      → df (raw)
engineer_features()      → df + feature columns, ml_feature_names[]
benfords_law.analyze()   → df + benford_* columns, summary DataFrame, stats dict
run_ensemble()           → df + if_score, lof_score, zscore_score columns
select_samples()         → df_scored (line-level), df_vouchers (voucher rollup), selected_vouchers (top-N)
    ↓
export_excel()           — 6-tab openpyxl workbook (voucher-level selection)
export_word_report()     — 6-page python-docx report with embedded matplotlib charts
```

**Note:** The HTML dashboard (`src/dashboard.py`) was removed in April 2025. Charts are now embedded directly in the Word report using matplotlib. `plotly` remains in `requirements.txt` but is no longer used by the pipeline.

### Key design decisions to know before editing

**`AMOUNT_COL`** is a long string constant (`'Payment Voucher Amount (SGD, Excluding GST)'`) defined at the top of every module that uses it. Always reference the constant, never the literal.

**Feature correlation pruning** (`feature_engineering._prune_correlated`): runs Spearman correlation on the candidate ML feature list and drops one of any pair with |corr| > 0.85. This runs at the end of `engineer_features()` and determines which columns reach `run_ensemble()`. The z-score columns (`amount_zscore_vendor`, `amount_zscore_costcentre`) often get pruned here but remain in the DataFrame — they are still used by `ml_models.zscore_score` and `sample_selector._build_reason`.

**Benford suppression rule** (`sample_selector.compute_risk_scores`): if a transaction's IF, LOF, z-score, and rule-flag scores are all below their dataset medians, its Benford contribution is zeroed out so it cannot be selected on Benford evidence alone.

**Recurring payment detection** (`feature_engineering._detect_recurring`): groups by `(Vendor ID, amount)` and checks whether all inter-date gaps fit one cycle (monthly 21–40 days, quarterly 80–100, semi-annual 170–195, annual 350–380 ±7 days). Transactions tagged `is_recurring_payment=1` have their `benford_deviation_score` zeroed out in `benfords_law.analyze()`.

**Composite score weights** are defined in `sample_selector.WEIGHTS` and can be overridden from the notebook before calling `select_samples()`.

**Individual payee detection** uses the Singapore NRIC/FIN regex `^[A-Za-z][0-9]{7}[A-Za-z]$` on `Vendor ID`.

**Word report — 6-page structure** (`report_generator`):
- Page 1 (portrait): Executive Summary — dataset overview table + findings bullets
- Page 2 (portrait): Methodology — plain-English description of each analytical method
- Page 3 (landscape): Analytical Charts — Benford's Law distribution + risk score histogram, side by side in a borderless 2-column table
- Page 4 (landscape): Payment Distribution & Timeline — amount distribution (log scale) + monthly timeline (dual-axis bar/line), stacked full-width
- Page 5 (landscape): Vendor Analysis — top 10 vendors by transaction count and by total amount
- Page 6 (landscape): Feature Reference Table — 11-row reference table with thresholds and audit rationale

Each landscape section is created by `_set_landscape()` via `doc.add_section()`. Charts are generated as in-memory PNG `BytesIO` objects using matplotlib (Agg backend) and embedded with `run.add_picture()`. Helper `_remove_table_borders()` is used for side-by-side chart layout on page 3.

### Module responsibilities

| Module | Key exports |
|---|---|
| `data_loader` | `load_transactions(filepath) → df` |
| `feature_engineering` | `engineer_features(df) → (df, ml_features[])` |
| `benfords_law` | `analyze(df) → (df, summary_df, stats_dict)` |
| `ml_models` | `run_ensemble(df, ml_features) → df` |
| `sample_selector` | `select_samples(df, n_samples) → (df_scored, df_vouchers, selected_vouchers)` |
| `excel_exporter` | `export_excel(df_scored, df_vouchers, selected_vouchers, summary, stats, path)` |
| `report_generator` | `export_word_report(df_scored, df_vouchers, selected_vouchers, stats, path)` |

### Required input columns

The tool validates exactly these 10 column names on load (raises `ValueError` if any are missing):

`Vendor Name`, `Vendor ID`, `Cost Centre`, `Account Code`, `Invoice Date`, `Voucher Accounting Date`, `Invoice Number`, `Voucher ID`, `Voucher Line Description`, `Payment Voucher Amount (SGD, Excluding GST)`

### What not to commit

`data/` — contains user transaction files (gitignored by `data/*.xlsx`, `data/*.csv`, etc.).  
`output/` — generated artefacts (gitignored by `output/*`). Only `data/.gitkeep` and `output/.gitkeep` are tracked.

## Accuracy benchmark (synthetic test, April 2025)

Tested against 530 synthetic transactions (500 normal + 30 injected anomalies, 5 per type).
No ground-truth data exists; this is the only benchmark on record.

| Anomaly Type | In Top 25 | Avg Score | Score Percentile |
|---|---|---|---|
| individual_payee | 5/5 | 0.358 | 98th |
| near_threshold | 4/5 | 0.454 | 97th |
| round_number | 4/5 | 0.348 | 96th |
| high_amount | 3/5 | 0.521 | 97th |
| month_end | 1/5 | 0.274 | 92nd |
| weekend_date | 0/5 | 0.301 | 94th |

**Overall: Recall 56.7% (17/30), Precision 68% (17/25), Cohen's d = 2.46 (strong separation)**

Interpretation: Single-signal anomalies (weekend date, month-end) score in the 92nd–94th percentile but are displaced from the top 25 by stronger multi-signal anomalies. The tool performs best when multiple flags stack on the same transaction.
