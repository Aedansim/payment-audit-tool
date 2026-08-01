# src/outputs/ — Excel, Word & PDF Output Reference

Read this when editing `excel_exporter.py`, `report_generator.py`, or `pdf_dashboard_generator.py`.

## Excel workbook — 10-tab structure (`excel_exporter.py`)

Sheet order in `export_excel`: Selected Vouchers → Voucher Line Detail → All Vouchers Scored → All Lines Scored → **Reversals Review** → **FY Split Purchase** → Benford's Law → **Analytical Charts** → **Excluded Vendors** → Summary.

| Tab | Content |
|---|---|
| 1 — **Selected Vouchers** | One row per selected voucher, colour-coded by risk tier (HIGH=red, MEDIUM=orange, LOW=yellow). Columns: `Voucher ID`, `Vendor Name`, `Invoice Number(s)`, `Voucher Line Description(s)` (pipe-separated, col width 50), `Total Amount (SGD)` (`#,##0.00`), scores, tier, flag count, ML Consensus Flag, `Vendor Capped` (True/False), reason codes. No Sample Rationale column. |
| 2 — **Voucher Line Detail** | All lines for selected vouchers, alternating background shading per voucher group. Includes all 10 rule flags + `if_anomaly`, `lof_anomaly`, `zscore_anomaly` (binary 0/1) via `_LINE_FLAG_COLS`. Payment Due Date and Payment Date columns are shown alongside Invoice Date / Voucher Accounting Date, all formatted DD/MM/YYYY. |
| 3 — **All Vouchers Scored** | Full voucher-level rollup sorted by `voucher_score` desc, colour-scale conditional formatting. Includes `Total Amount (SGD)`. Carries `is_excluded_vendor` (excluded vendors keep their true scores/tier here). |
| 4 — **All Lines Scored** | Full row-level scored dataset, colour-scale on `risk_score`. Flag columns: all 10 rule flags + `if_anomaly`, `lof_anomaly`, `zscore_anomaly` via inline `flag_cols` list. |
| 5 — **Reversals Review** | `df_scored` rows where `is_reversal == 1`, sorted Vendor ID → Voucher Accounting Date. Amber note on top, per-vendor alternating shading, dates DD/MM/YYYY, amount `#,##0.00`. Header row includes Invoice Date, Voucher Accounting Date, Payment Due Date, Payment Date. Matched original payment found via a **three-tier cascade** (Tier 1: Vendor ID + Invoice Number + abs amount; Tier 2: Vendor ID + Invoice Number; Tier 3: Vendor ID + abs amount; candidate must be positive). IDs go in "Matched Original Payment (Voucher ID)"; the tier annotation goes in a separate **"Match Basis"** column. Summary block (count, distinct vendors, total abs SGD, no-match count). Empty-safe message row. Built by `_sheet_reversals_review`. |
| 6 — **FY Split Purchase** | `df_scored` rows where `is_fy_split_purchase == 1`, sorted Vendor ID → Fiscal Year → Payment Date. Amber note states fiscal year is determined by **Payment Date** (incl. reference-code limitation + "review aid only, does not influence the risk score"). Columns incl. Fiscal Year, Payment Due Date, Payment Date, "No. of Similar Payments in FY", "Group Total (SGD)". Summary block (flagged count, distinct vendors, distinct vendor-FY-desc groups, total SGD). Empty-safe message row. Built by `_sheet_fy_split_purchase`. |
| 7 — **Benford's Law** | Rows 4–8: summary stats. Row 10: digit frequency table header. Rows 11–19: digit data, deviant digits highlighted orange. Below: recurring payment exclusions note → "Understanding These Metrics" section (MAD, Chi-Square, Conformity Verdict at font 10) → "Key Takeaway" section (soft blue header, warm yellow body, dynamic text based on `stats['mad']` and `stats['p_value']`). |
| 8 — **Analytical Charts** | Seven matplotlib charts rendered to PNG and embedded via `openpyxl.drawing.image.Image`: Benford observed-vs-expected, voucher risk score distribution, payment amount distribution (log), monthly timeline (dual-axis), late-vs-on-time proportion (pie), payment-lateness distribution (days-late histogram), top-10 vendors. Each has a bold title row + italic caption. Built by `_sheet_analytical_charts`; chart builders (`_chart_benford`, `_chart_risk_distribution`, `_chart_amount_distribution`, `_chart_timeline`, `_chart_late_payment_proportion`, `_chart_days_late_distribution`, `_chart_top_vendors`, `_to_image`) live in `excel_exporter.py`. The two late-payment charts are empty-safe (no late rows → full "On-time" pie + "No late payments" placeholder). |
| 9 — **Excluded Vendors** | Read from the `Excluded vendors.xlsx` file. Amber note (only `uen` matches Vendor ID; `entity_name` reference-only; user-editable between runs). Columns: "Excluded Vendor UEN", "Entity Name". Empty set → single message row. Built by `_sheet_excluded_vendors(wb, excluded)` where `excluded` is the `ExcludedVendors(uens, names)` namedtuple. |
| 10 — **Summary** | **Dataset Overview** block (**Payment Period** label using `Payment Date` min/max formatted DD/MM/YYYY, total lines, vouchers, avg lines, unique vendors, total SGD, amount range, recurring excluded). Tier distribution + audit sample breakdown + selected score range. **Summary of Findings** block (Benford/IF/LOF/z-score/rule counts, late-payment line count + % of lines + SGD + % of total value, duplicate + split voucher counts & SGD, reversal count & abs SGD, FY split group count & SGD, count of de-prioritised/excluded vendors). Amber-background note on the file-driven Excluded vendors de-prioritisation. Two dark-navy-header blocks: "Scope and Limitations" and "Sample Selection Basis". |

**Shared review-sheet helpers:** `_amber_note(ws, text, n_cols, row)` (full-width amber banner) and `_write_summary_block(ws, start_row, items, label_span=3)` (navy-label / white-value rows) are used by the review sheets.

**`Voucher Line Description(s)` in Tab 1:** fixed column width 50. Collected in `_rollup_vouchers()` using list comprehension with `pd.notna()` guard — do NOT use `.astype(str).str.strip().pipe(...isin...)` (float NaN breaks `str.join()`).

## Word report — 4-page structure (`report_generator.py`)

The Word report is now a **methodology document**. The dataset overview, summary of findings, and all charts moved to the Excel workbook (Summary + Analytical Charts tabs).

| Page | Orientation | Content |
|---|---|---|
| 1 | Portrait | **Scope and Limitations** — 4 transparency caveats: (1) not a fraud detection tool; (2) line-item scope; (3) pre-calibrated weights; (4) declared weights are approximate. Rendered by `_page_caveats(doc)`, called first. |
| 2 | Portrait | **Executive Summary** — opening paragraph (composite risk score, stratification, diversity controls, professional-judgement caveat; does NOT name the exclusion list, Jaccard threshold, or vendor cap) + a pointer paragraph directing the reader to the Excel Summary and Analytical Charts tabs for the dataset overview, summary of findings, and charts + closing paragraph (reason codes, line-to-voucher rollup, Excel tab pointers). No dataset-overview table or findings bullets here anymore. |
| 3 | Portrait | **Methodology** — audit-grade standalone. Stage 1 feature engineering; Stage 2 five analytical methods with caveats; Stage 3 exact line-level formula + weight rationale table; Benford suppression rule; Stage 4 voucher rollup formula; ML consensus flag; "Potential Fiscal Year Split Purchases" subsection (review aid, NOT scored); "Risk Tier Assignment and Sample Selection" subsection (percentile cutoffs, stratified draw, similarity deduplication, vendor cap, **file-driven Excluded vendors de-prioritisation** with dynamic count via `excluded_count` parameter). `_page2()` accepts `excluded_count`; `export_word_report(..., excluded_count)` is passed `len(excluded.uens)` from the notebook. |
| 4 | Landscape | **Feature Reference Table** — two tables, 5 columns each: Feature, What It Measures, Threshold for Flagging, ML Models, Why It Matters. Column widths: 1.6/2.1/1.8/1.2/3.8 inches. Table 1 ("Features Used in ML Models"): 17 rows including a "Late payment" row (Payment Date > Payment Due Date); ML Models = "IF, LOF, Z-score" for z-score features, "IF, LOF" for others; intro notes Spearman pruning may reduce active count. Table 2 ("Features Outside ML Models"): 1 row for Benford's Law. Rendered via `_render_feature_table()`. References footer: Nigrini (2012) and ACFE Fraud Examiners Manual only. |

## PDF dashboard — 9-page structure (`pdf_dashboard_generator.py`)

Read-only, charts-only visual companion to the Excel workbook, written to `output/audit_dashboard.pdf`. Landscape A4 (11.69 × 8.27 in), one `PdfPages` handle, one matplotlib figure per page.

**Privacy posture — do not weaken:** PDF is chosen because it is inert (cannot run scripts or make network requests). Build with matplotlib Agg + `PdfPages` **only** — no `reportlab`, no new dependency, no network access, matplotlib's built-in fonts only. Nothing here may alter a score, flag, weight, rollup, tier, or selection result.

| Page | Content | Source |
|---|---|---|
| 1 | Cover title + 4×2 grid of bordered metric cards (dataset, Payment Period, generated, sample size, transactions, vouchers, vendors, total SGD) | `meta`, with every key falling back to a value derived from the dataframes |
| 2 | Risk tier donut, counts + % in the legend, total in the hole | `df_vouchers['voucher_risk_tier']` |
| 3 | `voucher_score` histogram with dashed 80th/95th percentile boundary markers | recomputed via `.quantile()` from the same column `_assign_risk_tier` uses, so markers cannot drift from tiers |
| 4 | Reason-code frequency bars across the selected samples | `_tally_reason_codes()` |
| 5 | Benford observed-vs-expected grouped bars + a stats/interpretation panel | `benford_stats` |
| 6 | Vendor risk concentration — `barh` shaded by a `YlOrRd` colormap + colourbar, top 20 by summed `voucher_score` | `df_vouchers` grouped by `Vendor Name` |
| 7 | Top 15 vendors by positive spend | `df_scored` |
| 8 | Top 15 `Account Description` values by positive spend | `df_scored`; missing **or entirely blank** → "Account Description not available in this dataset." |
| 9 | Potential-duplicate big-number callouts (count, total SGD, how many are in the sample) | `df_vouchers` reason codes containing "Potential duplicate payment" |

**Split purchases and reversals appear nowhere in the PDF** — deliberate; they stay in their Excel review tabs.

**Page 9 counts are dataset-wide**, not sample-only; the third card reports the sample overlap.

**Every page degrades gracefully:** empty or missing input renders a centred message via `_message_page()` instead of an empty or broken chart. The page count is always 9.

**Layout constants:** the content band is `_C_BOTTOM`/`_C_HEIGHT` (top fixed at 0.840). The bottom must clear the axis label, which sits *below* the axes — lowering `_C_BOTTOM` makes xlabels collide with the caption line. Pages passing an explicit rect must reuse these constants, not literals.

**Palette:** navy `#1F3864` titles, tiers HIGH `#C00000` / MEDIUM `#FF8C00` / LOW `#375623`, grey `#8C8C8C` footer, card fill `#F7F9FC`. Money always `SGD 1,234.56` (`_fmt_sgd`), dates always DD/MM/YYYY (`_fmt_date`). Bars carry value labels so charts stay readable in greyscale.

**Reason-code parser (`_tally_reason_codes`):** `voucher_reason_codes` is `' | '`-joined; multi-line vouchers prefix each entry with `[Account Code] ` and vendor-capped vouchers append a `NOTE FOR AUDITOR` fragment. The parser drops that fragment, strips the account-code prefix, then maps free text to a canonical label via the ordered `_REASON_TYPES` substring table — necessary because several reasons embed variable numbers. **`_REASON_TYPES` mirrors `sample_selector._build_reason()` — if a reason string is reworded there, update the substring here or its bar silently falls into "Other".**

**Benford interpretation text is shared wording** with the Excel Benford tab's Key Takeaway (`excel_exporter._sheet_benford`). `_benford_interpretation()` reproduces the same three branches (not-significant / high-MAD-significant / low-MAD-significant) — reword both together.

## Typography

- **Font:** Times New Roman throughout — set on Normal, Heading 1, Heading 2, List Bullet styles, and explicitly on every run in `_heading`, `_body`, `_bullet`, `_coloured_para`, and all table-building code.
- **Alignment:** `WD_ALIGN_PARAGRAPH.JUSTIFY` for all paragraph text and table cell paragraphs.
- **Size scale** — `_body`, `_bullet`, `_coloured_para` each add 2pt internally (caller passes `size=10` → `Pt(12)` in document):
  - Normal/portrait body/bullets: 12 pt (caller passes 10)
  - Weight-rationale table: 11 pt (caller passes 9)
  - Landscape sub-descriptions: 11 pt (caller passes 9)
  - Feature reference table header: 10 pt (caller passes 8)
  - Feature reference table rows: 9.5 pt (caller passes 7.5)
  - References footer: 9.5 pt (caller passes 7.5)
- **Do not revert** to pre-April-2026 sizes (body=10, weight table=9, feature header=8, feature rows=7.5).

**Formula line exception:** the `risk_score` formula (Stage 3) and both `voucher_score` formulas (Stage 4) render at `Pt(10)` (caller passes `size=8`) with `WD_ALIGN_PARAGRAPH.LEFT`. Apply by capturing `_body()` return value and setting `p.alignment = WD_ALIGN_PARAGRAPH.LEFT` immediately after.

## Implementation notes

- **Landscape sections (Word):** created by `_set_landscape()` via `doc.add_section()` — still used by the Feature Reference Table page.
- **Charts (Excel):** generated as in-memory PNG `BytesIO` objects using matplotlib (Agg backend) in `excel_exporter.py`, embedded into the Analytical Charts sheet via `openpyxl.drawing.image.Image` (`XLImage`). Requires Pillow (`pillow` in `requirements.txt`). Charts no longer appear in the Word report.
- **Cell shading helper (Word):** `_shade_cell()` uses lxml `find(qn(...))` directly — do NOT use `get_or_add_tcPr()` (removed in python-docx 1.x).
- **HTML dashboard (`src/dashboard.py`):** removed April 2025. `plotly` remains in `requirements.txt` but is unused.
