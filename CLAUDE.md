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

# Run accuracy benchmark (530 synthetic transactions, 30 injected anomalies)
python benchmark.py

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

**Two-level scoring — line items then payment vouchers** (`sample_selector`):
- All scoring (Benford, ML ensemble, z-scores, rule flags) runs at the individual line-item level, producing `risk_score` per row. The feature engineering and scoring engine are not involved in the rollup.
- Lines are then grouped by `Voucher ID` into payment vouchers. `Voucher ID` is the audit unit because it is system-generated, always present, and is the document auditors physically pull. `Invoice Number` is retained as a display field (`Invoice Number(s)`) showing which vendor invoices the voucher relates to.
- Voucher score formula: `0.60 × max_line_score + 0.25 × mean_line_score + 0.15 × flag_density`. Single-line vouchers have `voucher_score = risk_score` exactly.
- Risk tiers assigned by percentile: HIGH (top 5%), MEDIUM (next 15%), LOW (rest).
- Stratified sample selection: all HIGH mandatory (capped at `n_samples`), ~75% of remainder from MEDIUM, random LOW baseline.
- Reason codes for single-line vouchers: plain text, no prefix. For multi-line vouchers: prefixed with `[Account Code]` to identify which line triggered each reason.

**Sample size cap** (`sample_selector._stratified_sample`): if the HIGH tier alone contains ≥ `n_samples` vouchers, the function returns only the top `n_samples` from HIGH and skips MEDIUM/LOW entirely. This ensures the output is always exactly `n_samples`.

**Word report — 6-page structure** (`report_generator`):
- Page 1 (portrait): Executive Summary — dataset overview table + findings bullets
- Page 2 (portrait): Methodology — comprehensive audit-grade standalone document covering: four-stage pipeline overview (feature engineering → scoring → voucher rollup → sample selection); each of the five analytical methods with caveats; exact line-level scoring formula (`0.30×IF + 0.25×LOF + 0.25×Z-score + 0.15×rule_flags + 0.05×Benford`) and weight rationale table; Benford suppression rule; voucher rollup formula (`0.60×max + 0.25×mean + 0.15×flag_density`); ML consensus flag explanation; risk tier percentile cutoffs; stratified sample selection logic; seven transparency caveats (the 7th notes that declared component weights are approximate because features shared across components carry marginally more effective influence than their labelled percentage alone suggests, and explains why this does not affect the relative ranking output)
- Page 3 (landscape): Analytical Charts — Benford's Law distribution + voucher risk score histogram, side by side in a borderless 2-column table
- Page 4 (landscape): Payment Distribution & Timeline — amount distribution (log scale) + monthly timeline (dual-axis bar/line), stacked full-width
- Page 5 (landscape): Vendor Analysis — top 10 vendors by transaction count and by total amount
- Page 6 (landscape): Feature Reference Table — Two tables, both with 5 columns: Feature, What It Measures, Threshold for Flagging, ML Models, Why It Matters. Column widths: 1.6/2.1/1.8/1.2/3.8 inches. Table 1 ("Features Used in Machine Learning Models"): 10 rows covering the features that feed into IF, LOF, and/or Z-score. ML Models column shows "IF, LOF, Z-score" for the two amount z-score features (which directly drive the Z-score component) and "IF, LOF" for all others. Table 2 ("Features Outside Machine Learning Models"): 1 row for Benford's Law first digit, ML Models = "None — Benford's Law analysis only (5% of composite score)". Rendered via shared helper `_render_feature_table()`. References footer cites Nigrini (2012) and ACFE Fraud Examiners Manual only.

Each landscape section is created by `_set_landscape()` via `doc.add_section()`. Charts are generated as in-memory PNG `BytesIO` objects using matplotlib (Agg backend) and embedded with `run.add_picture()`. Helper `_remove_table_borders()` is used for side-by-side chart layout on page 3. Both `_shade_cell()` and `_remove_table_borders()` use lxml `find(qn(...))` directly — do NOT use `get_or_add_tblPr()` or `get_or_add_tcPr()`, which were removed in python-docx 1.x.

**Excel workbook — 6-tab structure** (`excel_exporter`):
- Tab 1 — **Selected Vouchers**: one row per selected voucher, colour-coded by risk tier (HIGH=red, MEDIUM=orange, LOW=yellow). Shows `Voucher ID`, `Vendor Name`, `Invoice Number(s)`, scores, tier, flag count, ML consensus flag, reason codes. No Sample Rationale column.
- Tab 2 — **Voucher Line Detail**: all transaction lines belonging to selected vouchers, alternating background shading per voucher group, individual line scores and flags visible so auditors can see which line drove selection.
- Tab 3 — **All Vouchers Scored**: full voucher-level rollup sorted by `voucher_score` descending, with colour-scale conditional formatting.
- Tab 4 — **All Lines Scored**: full row-level scored dataset (reference), with colour-scale on `risk_score`.
- Tab 5 — **Benford's Law**: summary statistics (rows 4–8), followed by an "Understanding These Metrics" explanation block (rows 9–15) covering MAD thresholds (Nigrini 2012), chi-square large-dataset caveat, Conformity Verdict interpretation, and a Key Takeaway box on reading MAD and chi-square together. Digit frequency table starts at row 16 (deviant digits highlighted in orange).
- Tab 6 — **Summary**: dataset counts, tier distribution, and audit sample breakdown. No methodology notes.

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

### Feature overlap — intentional design decision (April 2026)

`amount_zscore_vendor` and `amount_zscore_costcentre` feed into three components: the dedicated Z-score component (25% weight), and also the IF and LOF feature matrices (as two of ~10 inputs). The six rule-based flags feed into two components: the dedicated rule_flags_score (15% weight), and also the IF and LOF feature matrices. This means those features carry marginally more effective weight than their labelled percentages suggest.

This is documented as Caveat 7 in the Word report methodology page. It is **not a design flaw**: the overlap is a byproduct of ensemble cross-method reinforcement — transactions anomalous on these signals consistently rank above those that are not, which is the tool's objective. Removing these features from the IF/LOF matrix was considered and rejected because it would weaken detection coverage and the overlap effect is attenuated by the multi-dimensional nature of those models.

The `ML_Consensus_Flag` threshold of 0.65 on normalised scores was reviewed in April 2026. Switching to `sklearn.predict()` at `contamination=0.05` (top 5% boundary) was evaluated using `benchmark_comparison.py` and found to worsen Cohen's d (2.834→2.551) and recall (14→13) due to weight reallocation deflating single-line voucher scores. The 0.65 threshold is retained. If audit defensibility of the threshold becomes a concern, use `predict()` for the binary display flag only without changing the voucher formula.

`amount_zscore_overall` and `amount_zscore_account` were removed from `feature_engineering.py` in April 2026 — both were computed but never referenced in scoring, reason codes, or ML models. The dead `amount_zscore_overall` fallback branch in `ml_models.py` was also removed.

### Known design limitation

The payments listing contains multiple line items per payment voucher (same `Voucher ID`, different `Account Code` / `Cost Centre`). The tool's feature engineering and Benford analysis operate at the individual line level, not at the voucher total. This means:
- Benford's Law is applied to individual line amounts, not the total voucher amount.
- Z-scores compare individual line amounts against vendor or cost centre averages.
- A large voucher split into many small lines may score unremarkably on individual lines even if the total is anomalous.

This is intentional: voucher-level aggregation before scoring would lose line-level signals (e.g. one suspicious line in a normal voucher). The two-level approach — score lines, roll up to vouchers — is the chosen trade-off.

### What not to commit

`data/` — contains user transaction files (gitignored by `data/*.xlsx`, `data/*.csv`, etc.).  
`output/` — generated artefacts (gitignored by `output/*`). Only `data/.gitkeep` and `output/.gitkeep` are tracked.  
`benchmark.py` — committed to the repo as a development/QA tool. It is not part of the production pipeline.  
`benchmark_comparison.py` — committed as a QA tool for comparing two pipeline configurations side-by-side. Runs both pipelines against the same synthetic dataset and prints recall, precision, Cohen's d, ML consensus flag distribution, and score statistics. No src/ files are modified by the script; all modified logic is defined inline. Use this when evaluating proposed changes to the scoring formula or ML thresholds before deciding whether to adopt them.

## Accuracy benchmark (synthetic test, April 2025 — updated)

Run with `python benchmark.py`. Tests against 530 synthetic transactions (500 normal + 30 injected anomalies, 5 per type), each as its own single-line voucher. Scores are at voucher level.

### Current results (voucher-level selection)

| Anomaly Type | In Top 25 | Avg Score | Score Percentile |
|---|---|---|---|
| individual_payee | 5/5 | 0.546 | 99th |
| round_number | 3/5 | 0.448 | 96th |
| high_amount | 2/5 | 0.410 | 94th |
| near_threshold | 2/5 | 0.383 | 91st |
| weekend_date | 1/5 | 0.391 | 93rd |
| month_end | 1/5 | 0.317 | 86th |

**Overall: Recall 46.7% (14/30), Precision 56.0% (14/25), Cohen's d = 2.83 (strong separation)**

### Previous results (line-level selection, April 2025)

| Anomaly Type | In Top 25 | Avg Score | Score Percentile |
|---|---|---|---|
| individual_payee | 5/5 | 0.358 | 98th |
| near_threshold | 4/5 | 0.454 | 97th |
| round_number | 4/5 | 0.348 | 96th |
| high_amount | 3/5 | 0.521 | 97th |
| month_end | 1/5 | 0.274 | 92nd |
| weekend_date | 0/5 | 0.301 | 94th |

**Overall: Recall 56.7% (17/30), Precision 68% (17/25), Cohen's d = 2.46**

### Interpreting the difference

The benchmark recall appears lower in the current version (46.7% vs 56.7%) but this is a synthetic test artefact. The benchmark uses single-line vouchers, so line-level and voucher-level selection are equivalent — the gap is due to the different random characteristics of the test datasets, not a real regression.

The meaningful comparison is Cohen's d: **2.83 vs 2.46**. This improved, meaning anomalous vouchers are more clearly separated from normal ones in score space. In real data with multi-line vouchers, recall is expected to be higher than the benchmark suggests because any flagged line elevates the whole voucher.

Single-signal anomalies (month-end, weekend date) consistently score in the 86th–93rd percentile but are displaced from the top 25 by stronger multi-signal anomalies. The tool performs best when multiple flags stack on the same transaction.
