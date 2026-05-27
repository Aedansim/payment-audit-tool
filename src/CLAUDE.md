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

**Transposed amount detection** (`_detect_transposed_amounts`): flags same-vendor + same-lowercase-description pairs where two positive amounts differ by exactly one digit-position swap in their **whole-dollar integer** strings (`str(int(abs(amount)))` — cents dropped, so swaps cannot cross the decimal point; e.g. `$348.23`/`$328.43` is NOT a transposition, `$4,800`/`$8,400` is). Returns tuple `(is_transposed Series, transposed_matched_invoice Series)`. Unpacked in `engineer_features()` as: `df['is_transposed_amount'], df['transposed_matched_invoice'] = _detect_transposed_amounts(df)`. The description key is a strict near-exact match (lowercase + strip only — no digit/punctuation/month/filler removal) **by design**, to avoid false matches between unrelated payments; this is deliberately NOT the FY-split normalisation.

**Duplicate detection** (`_detect_duplicates`): returns tuple `(is_duplicate Series, duplicate_matched_voucher Series)` — Voucher ID(s) of counterpart(s), comma-separated. Unpacked: `df['is_duplicate'], df['duplicate_matched_voucher'] = _detect_duplicates(df)`. Old `duplicate_matched_invoice` column removed.

**Amount z-scores on absolute value:** `amount_zscore_vendor` and `amount_zscore_costcentre` are computed via `_group_zscore(df[AMOUNT_COL].abs(), df['<group>'])` — the helper now takes value/group **Series** (not column names). Reversals are assessed by magnitude, not flagged for sign. Displayed `AMOUNT_COL` stays signed everywhere; only the z-score transform uses `.abs()` (as `amount_log` already does).

**FY split purchase detection** (`_detect_fy_split_purchase`): flags positive-amount rows in a `(Vendor ID, fiscal-year, normalised-description)` group of 2+ payments whose positive total > `_FY_SPLIT_THRESHOLD` (SGD 6,000). FY = 1 Apr–31 Mar from **Voucher Accounting Date** (`_fy_label`). Description key (`_normalise_fy_desc`): lowercase/strip → drop digits → punctuation→space → drop whole-token month names (`_FY_MONTH_TOKENS`) and filler words (`_FY_FILLER_TOKENS`) → collapse. Fallback (g): when the cleaned key is blank OR has no token ≥ 3 letters (e.g. `PO-4471-22` → `po`), use the full original reference (digits/punctuation kept) so distinct reference codes are NOT merged. Returns 4 columns `(is_fy_split_purchase, fy_split_group_total, fy_split_group_count, fy_split_fy_label)`. **Deliberately excluded from `ml_features` AND `FLAG_COLS`** — review aid only, does not influence the composite score (recurring legitimate payments would otherwise false-positive). Surfaced via reason code, the Excel "FY Split Purchase" tab, and a Word report subsection. Accepts `excluded_uens` (from `engineer_features(df, excluded_uens)`): rows whose `Vendor ID` is in the Excluded vendors set are dropped before grouping — never flagged, never contributing to a group total/count.

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

**Excluded-vendor de-prioritisation:** file-driven via `data_loader.load_excluded_vendors(data_folder)`, which reads `Excluded vendors.xlsx` (column `uen` = Vendor ID; optional `entity_name` display-only) and returns an `ExcludedVendors(uens, names)` namedtuple. `select_samples(df, n_samples, excluded_uens)` marks vouchers whose `Vendor ID` (trimmed, full-identifier exact match — **not** a prefix) is in `excluded_uens` as `is_excluded_vendor=True`. Temporary copy `df_for_sampling` overrides their tier to LOW before sampling; real tier restored in `selected_vouchers` after. `df_vouchers` never modified. Both `_similarity_filter()` and `_vendor_cap()` build replacement candidate pools non-excluded first (score desc), excluded appended last. The previous hardcoded vendor-ID-prefix rule for government agencies was fully removed — all exclusions are now file-driven. The same `excluded_uens` set is also passed to `engineer_features()` → `_detect_fy_split_purchase()` to drop excluded vendors from FY split detection.

**Sample size cap** (`_stratified_sample`): if HIGH tier alone ≥ `n_samples`, return only top `n_samples` from HIGH, skip MEDIUM/LOW.

**Execution order:** `_stratified_sample → _similarity_filter → _vendor_cap → return`

**Similarity deduplication** (`_similarity_filter`): for each vendor with ≥2 selected vouchers, Jaccard token-overlap > 0.70 → drop lower-scoring, replace with next-best unselected across all vendors passing the threshold. Marks `similarity_deduplicated=True` on replacement rows. When building replacement row use `pd.DataFrame([replacement.to_dict()])` — NOT `replacement.to_frame().T` (transposing mixed-type Series converts to object dtype, breaks `.round()` on `voucher_score`).

**Why the similarity filter can miss near-duplicate descriptions:** `_get_voucher_desc()` lowercases and whitespace-splits the raw description — it does not strip leading reference-number prefixes (e.g. `"ABC5:Invitation to Project"`). When there is no space after the colon, the prefix and the following word fuse into a single token (`"abc5:invitation"`), which has zero overlap with `"abc6:invitation"`. Example: descriptions `"ABC5:Invitation to Project (ITP)"` and `"ABC6:Invitation to Project (ITP)"` tokenise to Jaccard = 3/5 = 0.60, below the 0.70 threshold → both are retained in the sample.

**Why the threshold was not lowered and prefixes were not auto-stripped:** lowering the threshold risks over-suppressing legitimate vouchers that share common boilerplate. Stripping prefixes risks false deduplication when the prefix IS a meaningful business code (e.g. a project or cost-centre reference) that genuinely distinguishes two transactions — this cannot be determined from data alone without business confirmation. The chosen approach is to leave the threshold and tokeniser unchanged and rely on the auditor to investigate pairs that appear similar in the output. If a future dataset confirms that a specific prefix pattern is always a sequential label (never a meaningful code), Option B (regex strip in `_get_voucher_desc`) or a threshold reduction to 0.60 can be revisited then.

**Similarity vs. duplicate detection:** `_detect_duplicates()` in `feature_engineering.py` matches on Vendor ID + Invoice Number + Invoice Date + Amount (rows with a missing Invoice Date are not flagged) — it never compares description text. Near-duplicate descriptions with different invoice numbers will not be caught there; only `_similarity_filter()` addresses them.

**Vendor cap** (`_vendor_cap`): max 2 vouchers per Vendor ID. Excess replaced subject to: (1) Jaccard ≤ 0.70 vs all retained, (2) vendor count guard (skip candidates from vendors already holding 2 slots). Uses `voucher_to_vendor` dict + per-iteration `vendor_retained_counts` dict. Marks `vendor_capped=True`. Slot left unfilled if no candidate passes both checks.

**Reason codes:** single-line vouchers = plain text. Multi-line vouchers = prefixed with `[Account Code]`. IF and LOF anomaly reasons appended unconditionally when `if_anomaly==1` or `lof_anomaly==1`. Final fallback `"Elevated composite risk score"` only when no signal triggered. Vendor-capped vouchers get `" | NOTE FOR AUDITOR: ..."` suffix.

## Rule flags (9 total — FLAG_COLS)

`is_round_number`, `is_weekend_payment`, `near_threshold`, `is_individual_payee`, `same_amount_vendor_irregular`, `is_duplicate`, `is_reversal`, `is_split_purchase_risk`, `is_transposed_amount`

Both `_rule_flags_score()` and `flag_density` in `_rollup_vouchers()` divide by `len(present)` — never a hardcoded number. All references in `report_generator.py` and `make_scoring_reference.py` use "9 rules / ÷9". The Methodology page worked example reads `"2 rules triggered = 2/9 = 0.22"` — do not revert to 2/10 or 2/8.

## ML Consensus Flag

Set from binary `if_anomaly`, `lof_anomaly`, `zscore_anomaly` flags (predict()-based, not score threshold). Old `> 0.65` threshold references fully removed — both in `report_generator._page1()` bullet labels and `sample_selector._build_reason()` fallback. Bullet labels now read `(top 5% boundary)`.
