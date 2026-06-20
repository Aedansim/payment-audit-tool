# Payment Audit Tool — CLAUDE.md

Python-based payment audit pipeline that scores transactions using ML ensemble + Benford's Law + rule flags, selects a stratified audit sample, and exports to Excel + Word.

## Quick commands

```bash
pip install -r requirements.txt
python benchmark.py                  # accuracy benchmark (525 synthetic transactions)
python make_scoring_reference.py     # generates output/Scoring_Methodology.xlsx
git add src/<file>.py && git commit -m "short imperative" && git push
```

## Pipeline (strictly linear — each stage adds columns to the same DataFrame)

```
load_transactions()     → df (raw)
load_excluded_vendors() → ExcludedVendors(uens, names)  (reads data/Excluded vendors.xlsx; optional)
engineer_features(df, excluded_uens)  → df + feature columns, ml_feature_names[]
benfords_law.analyze()  → df + benford_* columns, summary_df, stats_dict
run_ensemble()          → df + if_score, lof_score, zscore_score, if_anomaly, lof_anomaly, zscore_anomaly
select_samples(df, n_samples, excluded_uens)  → df_scored, df_vouchers, selected_vouchers
    ↓
export_excel()          — 10-tab openpyxl workbook (charts embedded as matplotlib images)
export_word_report()    — 4-page python-docx methodology report (no charts; data moved to Excel)
```

Orchestrated by `Payment_Audit_Tool.ipynb`. Notebook step order: STEP 0 = config (INPUT_FILE, SAMPLE_SIZE, WEIGHTS), STEP 1 = package install.

## Module responsibilities

| Module | Key export |
|---|---|
| `data_loader` | `load_transactions(filepath) → df` ; `load_excluded_vendors(data_folder) → ExcludedVendors(uens, names)` |
| `feature_engineering` | `engineer_features(df, excluded_uens=None) → (df, ml_features[])` |
| `benfords_law` | `analyze(df) → (df, summary_df, stats_dict)` |
| `ml_models` | `run_ensemble(df, ml_features) → df` |
| `sample_selector` | `select_samples(df, n_samples, excluded_uens=None) → (df_scored, df_vouchers, selected_vouchers)` |
| `excel_exporter` | `export_excel(df_scored, df_vouchers, selected_vouchers, summary, stats, path, excluded=None)` |
| `report_generator` | `export_word_report(df_scored, df_vouchers, selected_vouchers, stats, path, excluded_count=0)` |

## Required input columns (12 — raises ValueError if any missing)

`Vendor Name`, `Vendor ID`, `Cost Centre`, `Account Code`, `Invoice Date`, `Voucher Accounting Date`, `Payment Due Date`, `Payment Date`, `Invoice Number`, `Voucher ID`, `Voucher Line Description`, `Payment Voucher Amount (SGD, Excluding GST)`

`Payment Date` is the actual date of payment and drives `processing_days`, the FY-split fiscal-year label, and the Summary tab's Payment Period. `Payment Due Date` drives the `is_late_payment` rule flag (Payment Date > Payment Due Date). `Voucher Accounting Date` is still required because it drives `is_weekend_payment` (the date the voucher was raised). Any other new columns in the input file are ignored.

## Critical constants

- **`AMOUNT_COL`** = `'Payment Voucher Amount (SGD, Excluding GST)'` — always use the constant, never the literal string.
- **`FLAG_COLS`** = 10 rule-based flags. Denominator in `_rule_flags_score` and `flag_density` is always `len(present)` — never hardcoded.
- **`WEIGHTS`** defined in `sample_selector.WEIGHTS` — overridable from notebook before calling `select_samples()`.

## Git discipline

Commit after every meaningful unit of work. Never batch unrelated changes.

```
fix: handle negative processing_days in zscore calculation
feat: add weekly recurrence cycle to recurring detection
refactor: extract threshold logic into shared constant
```

Always `git push` immediately after every commit.

## What not to commit

- `data/` — user transaction files (gitignored)
- `output/` — generated artefacts (gitignored); only `.gitkeep` files tracked
- `benchmark.py`, `benchmark_comparison.py` — QA tools, committed but not production pipeline
- `make_scoring_reference.py` — documentation utility, committed

## Sub-memory files (read these when working on specific areas)

| File | When to read |
|---|---|
| `src/CLAUDE.md` | Editing any `src/` module — scoring logic, feature engineering, design decisions |
| `src/outputs/CLAUDE.md` | Editing Excel or Word report output — structure, formatting, typography |
| `docs/CLAUDE.md` | Benchmark results, accuracy history, interpreting benchmark vs production |
