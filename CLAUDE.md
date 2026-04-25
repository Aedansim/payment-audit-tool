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
          'src/dashboard.py','src/report_generator.py']:
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
df_scored, selected = select_samples(df)
print(selected[['Sample #','Vendor Name','risk_score','Selection Reasons']].head())
"

# Git workflow
git add src/<changed_file>.py
git commit -m "short imperative description"
git push
```

## Architecture

The pipeline is strictly linear — each stage adds columns to the same DataFrame and passes it to the next. All modules live in `src/` and are orchestrated by `Payment_Audit_Tool.ipynb`.

```
load_transactions()      → df (raw)
engineer_features()      → df + feature columns, ml_feature_names[]
benfords_law.analyze()   → df + benford_* columns, summary DataFrame, stats dict
run_ensemble()           → df + if_score, lof_score, zscore_score columns
select_samples()         → df_scored (full, sorted), selected (top-N with reasons)
    ↓
export_excel()           — 3-tab openpyxl workbook
export_dashboard()       — self-contained Plotly HTML
export_word_report()     — 3-page python-docx report
```

### Key design decisions to know before editing

**`AMOUNT_COL`** is a long string constant (`'Payment Voucher Amount (SGD, Excluding GST)'`) defined at the top of every module that uses it. Always reference the constant, never the literal.

**Feature correlation pruning** (`feature_engineering._prune_correlated`): runs Spearman correlation on the candidate ML feature list and drops one of any pair with |corr| > 0.85. This runs at the end of `engineer_features()` and determines which columns reach `run_ensemble()`. The z-score columns (`amount_zscore_vendor`, `amount_zscore_costcentre`) often get pruned here but remain in the DataFrame — they are still used by `ml_models.zscore_score` and `sample_selector._build_reason`.

**Benford suppression rule** (`sample_selector.compute_risk_scores`): if a transaction's IF, LOF, z-score, and rule-flag scores are all below their dataset medians, its Benford contribution is zeroed out so it cannot be selected on Benford evidence alone.

**Recurring payment detection** (`feature_engineering._detect_recurring`): groups by `(Vendor ID, amount)` and checks whether all inter-date gaps fit one cycle (monthly 21–40 days, quarterly 80–100, semi-annual 170–195, annual 350–380 ±7 days). Transactions tagged `is_recurring_payment=1` have their `benford_deviation_score` zeroed out in `benfords_law.analyze()`.

**Composite score weights** are defined in `sample_selector.WEIGHTS` and can be overridden from the notebook before calling `select_samples()`.

**Individual payee detection** uses the Singapore NRIC/FIN regex `^[A-Za-z][0-9]{7}[A-Za-z]$` on `Vendor ID`.

**Word report pagination**: `report_generator` uses two `Document.sections` — Section 1 is portrait (pages 1–2: Executive Summary + Methodology), Section 2 is landscape (page 3: Feature Reference Table). The landscape section is created by `_set_landscape()` which swaps `page_width`/`page_height` and tightens margins to 1.5 cm.

**Dashboard HTML** is self-contained: the first Plotly figure is exported with `include_plotlyjs='cdn'` (loads from CDN) and its `<script>` tag is extracted and moved to `<head>`. All subsequent figures use `include_plotlyjs=False`.

### Module responsibilities

| Module | Key exports |
|---|---|
| `data_loader` | `load_transactions(filepath) → df` |
| `feature_engineering` | `engineer_features(df) → (df, ml_features[])` |
| `benfords_law` | `analyze(df) → (df, summary_df, stats_dict)` |
| `ml_models` | `run_ensemble(df, ml_features) → df` |
| `sample_selector` | `select_samples(df, n_samples) → (df_scored, selected_df)` |
| `excel_exporter` | `export_excel(df_scored, selected, summary, stats, path)` |
| `dashboard` | `export_dashboard(df_scored, selected, stats, path)` |
| `report_generator` | `export_word_report(df_scored, selected, stats, path)` |

### Required input columns

The tool validates exactly these 10 column names on load (raises `ValueError` if any are missing):

`Vendor Name`, `Vendor ID`, `Cost Centre`, `Account Code`, `Invoice Date`, `Voucher Accounting Date`, `Invoice Number`, `Voucher ID`, `Voucher Line Description`, `Payment Voucher Amount (SGD, Excluding GST)`

### What not to commit

`data/` — contains user transaction files (gitignored by `data/*.xlsx`, `data/*.csv`, etc.).  
`output/` — generated artefacts (gitignored by `output/*`). Only `data/.gitkeep` and `output/.gitkeep` are tracked.
