# src/ — Scoring Logic & Design Decisions

Read this when editing any module in `src/`. See root `CLAUDE.md` for pipeline overview and module responsibilities.

## Scoring formulas

**Line-level composite score:**
`risk_score = 0.30×IF + 0.25×LOF + 0.25×Z-score + 0.15×rule_flags + 0.05×Benford`

**Voucher rollup:**
`voucher_score = 0.60×max_line_score + 0.25×mean_line_score + 0.15×flag_density`
Single-line vouchers: `voucher_score = risk_score` exactly.

**Risk tiers (percentile-based):** HIGH = top 5%, MEDIUM = next 15%, LOW = rest.

## Key design decisions

### Feature engineering (`feature_engineering.py`)

**Correlation pruning** (`_prune_correlated`): Spearman correlation on ML feature candidates; drops one of any pair with |corr| > 0.85. Runs at end of `engineer_features()`. `amount_zscore_vendor` and `amount_zscore_costcentre` often get pruned here but remain in the DataFrame — still used by `ml_models.zscore_score` and `sample_selector._build_reason`.

**Recurring payment detection** (`_detect_recurring`): groups by `(Vendor ID, amount)`; checks inter-date gaps fit one cycle (monthly 21–40d, quarterly 80–100d, semi-annual 170–195d, annual 350–380d ±7d). Tagged `is_recurring_payment=1` → `benford_deviation_score` zeroed in `benfords_law.analyze()`.

**Individual payee detection:** Singapore NRIC/FIN regex `^[A-Za-z][0-9]{7}[A-Za-z]$` on `Vendor ID`. (Case-insensitive — both upper and lower accepted.)

**Split purchase risk** (`_detect_split_purchase`): flags groups where same vendor + same `Invoice Date` + alphanumerically sequential invoice suffixes AND group total within 5% below $6,000 or $90,000 (bands: $5,700–<$6,000 or $85,500–<$90,000). Fully vectorised — no Python for-loop over groups. Suffixes cast to `np.int64` (not `int`/`np.int_`, which is 32-bit on Windows); suffixes >18 digits excluded via `Series.where(suffixes.str.len() <= 18)`.

**Transposed amount detection** (`_detect_transposed_amounts`): flags same-vendor + same-lowercase-description pairs where two positive amounts differ by exactly one digit-position swap in their cent-integer strings. Returns tuple `(is_transposed Series, transposed_matched_invoice Series)`. Unpacked in `engineer_features()` as: `df['is_transposed_amount'], df['transposed_matched_invoice'] = _detect_transposed_amounts(df)`.

**Duplicate detection** (`_detect_duplicates`): returns tuple `(is_duplicate Series, duplicate_matched_voucher Series)` — Voucher ID(s) of counterpart(s), comma-separated. Unpacked: `df['is_duplicate'], df['duplicate_matched_voucher'] = _detect_duplicates(df)`. Old `duplicate_matched_invoice` column removed.

**Removed features (do not restore):**
- `amount_zscore_overall`, `amount_zscore_account` — removed April 2026, never used in scoring, reason codes, or ML.
- `is_month_end` — removed; benchmark no longer injects month-end anomalies.
- Dead `amount_zscore_overall` fallback branch in `ml_models.py` — removed April 2026.

**Date loading** (`data_loader.load_transactions`): two-pass Excel read — header-only pass then full read with `dtype=str` for all columns *except* `Invoice Date` and `Voucher Accounting Date` (excluded so openpyxl returns proper `datetime` objects). `pd.to_datetime(dayfirst=True, errors='coerce')` loop still runs after for text-stored dates. Do NOT revert to single `pd.read_excel(dtype=str)` — reintroduces serial-number NaT bug.

**Period display (STEP 2 notebook):** creates `_preview` copy with `.dt.strftime('%d/%m/%Y').fillna('')` for display only. Underlying `df_raw` retains `datetime64` throughout for feature arithmetic.

### ML models (`ml_models.py`)

**`n_jobs=1`** on both `IsolationForest` and `LocalOutlierFactor` — shared notebook servers fork one worker per CPU core under `n_jobs=-1`, exceeding per-user memory limits. Quality unaffected. `IsolationForest` uses `n_estimators=100` (reduced from 300 — no measurable quality gain).

**Binary anomaly flags:** `if_anomaly` (IsolationForest.predict() == -1, top 5%), `lof_anomaly` (LOF.fit_predict() == -1, top 5%), `zscore_anomaly` (max absolute z-score > 2.0). Used for `ML_Consensus_Flag` display only — not part of `risk_score` formula.

**Feature overlap (intentional):** `amount_zscore_vendor` and `amount_zscore_costcentre` feed both the Z-score component (25%) and the IF/LOF feature matrices. The 9 rule flags feed both `rule_flags_score` (15%) and the IF/LOF matrices. This is Caveat 7 in the Word report — not a design flaw.

### Sample selector (`sample_selector.py`)

**Two-level scoring:** all scoring runs at line level → lines grouped by `Voucher ID` for audit unit. `Voucher ID` is system-generated and always present. `Invoice Number` → display field `Invoice Number(s)`. `Voucher Line Description` → `Voucher Line Description(s)` (unique non-blank values, pipe-separated).

**Voucher Line Description rollup** (`_rollup_vouchers`): use list comprehension with `pd.notna()` guard. Do NOT use `.astype(str).str.strip().pipe(...isin...)` — `Series.unique()` can return float NaN that bypasses `.isin(['nan'])` and breaks `str.join()`.

**Benford suppression rule** (`compute_risk_scores`): if IF, LOF, z-score, and rule-flag scores are all below dataset medians → Benford contribution zeroed (cannot be selected on Benford evidence alone).

**T08 de-prioritisation:** vouchers where `Vendor ID` starts with `T08` (case-insensitive) → marked `is_t08_vendor=True`. Temporary copy `df_for_sampling` overrides their tier to LOW before sampling; real tier restored in `selected_vouchers` after. `df_vouchers` never modified. Both `_similarity_filter()` and `_vendor_cap()` build replacement candidate pools non-T08 first (score desc), T08 appended last.

**Sample size cap** (`_stratified_sample`): if HIGH tier alone ≥ `n_samples`, return only top `n_samples` from HIGH, skip MEDIUM/LOW.

**Execution order:** `_stratified_sample → _similarity_filter → _vendor_cap → return`

**Similarity deduplication** (`_similarity_filter`): for each vendor with ≥2 selected vouchers, Jaccard token-overlap > 0.70 → drop lower-scoring, replace with next-best unselected across all vendors passing the threshold. Marks `similarity_deduplicated=True` on replacement rows. When building replacement row use `pd.DataFrame([replacement.to_dict()])` — NOT `replacement.to_frame().T` (transposing mixed-type Series converts to object dtype, breaks `.round()` on `voucher_score`).

**Vendor cap** (`_vendor_cap`): max 2 vouchers per Vendor ID. Excess replaced subject to: (1) Jaccard ≤ 0.70 vs all retained, (2) vendor count guard (skip candidates from vendors already holding 2 slots). Uses `voucher_to_vendor` dict + per-iteration `vendor_retained_counts` dict. Marks `vendor_capped=True`. Slot left unfilled if no candidate passes both checks.

**Reason codes:** single-line vouchers = plain text. Multi-line vouchers = prefixed with `[Account Code]`. IF and LOF anomaly reasons appended unconditionally when `if_anomaly==1` or `lof_anomaly==1`. Final fallback `"Elevated composite risk score"` only when no signal triggered. Vendor-capped vouchers get `" | NOTE FOR AUDITOR: ..."` suffix.

## Rule flags (9 total — FLAG_COLS)

`is_round_number`, `is_weekend_payment`, `near_threshold`, `is_individual_payee`, `same_amount_vendor_irregular`, `is_duplicate`, `is_reversal`, `is_split_purchase_risk`, `is_transposed_amount`

Both `_rule_flags_score()` and `flag_density` in `_rollup_vouchers()` divide by `len(present)` — never a hardcoded number. All references in `report_generator.py` and `make_scoring_reference.py` use "9 rules / ÷9". The Methodology page worked example reads `"2 rules triggered = 2/9 = 0.22"` — do not revert to 2/10 or 2/8.

## ML Consensus Flag

Set from binary `if_anomaly`, `lof_anomaly`, `zscore_anomaly` flags (predict()-based, not score threshold). Old `> 0.65` threshold references fully removed — both in `report_generator._page1()` bullet labels and `sample_selector._build_reason()` fallback. Bullet labels now read `(top 5% boundary)`.
