# src/outputs/ — Excel & Word Report Reference

Read this when editing `excel_exporter.py` or `report_generator.py`.

## Excel workbook — 6-tab structure (`excel_exporter.py`)

| Tab | Content |
|---|---|
| 1 — **Selected Vouchers** | One row per selected voucher, colour-coded by risk tier (HIGH=red, MEDIUM=orange, LOW=yellow). Columns: `Voucher ID`, `Vendor Name`, `Invoice Number(s)`, `Voucher Line Description(s)` (pipe-separated, col width 50), `Total Amount (SGD)` (`#,##0.00`), scores, tier, flag count, ML Consensus Flag, `Vendor Capped` (True/False), reason codes. No Sample Rationale column. |
| 2 — **Voucher Line Detail** | All lines for selected vouchers, alternating background shading per voucher group. Includes all 9 rule flags + `if_anomaly`, `lof_anomaly`, `zscore_anomaly` (binary 0/1) via `_LINE_FLAG_COLS`. |
| 3 — **All Vouchers Scored** | Full voucher-level rollup sorted by `voucher_score` desc, colour-scale conditional formatting. Includes `Total Amount (SGD)`. |
| 4 — **All Lines Scored** | Full row-level scored dataset, colour-scale on `risk_score`. Flag columns: all 9 rule flags + `if_anomaly`, `lof_anomaly`, `zscore_anomaly` via inline `flag_cols` list. |
| 5 — **Benford's Law** | Rows 4–8: summary stats. Row 10: digit frequency table header. Rows 11–19: digit data, deviant digits highlighted orange. Below: recurring payment exclusions note → "Understanding These Metrics" section (MAD, Chi-Square, Conformity Verdict at font 10) → "Key Takeaway" section (soft blue header, warm yellow body, dynamic text based on `stats['mad']` and `stats['p_value']`). |
| 6 — **Summary** | Dataset counts, tier distribution, audit sample breakdown. Amber-background note on T08 de-prioritisation. Two dark-navy-header blocks: "Scope and Limitations" and "Sample Selection Basis". |

**`Voucher Line Description(s)` in Tab 1:** fixed column width 50. Collected in `_rollup_vouchers()` using list comprehension with `pd.notna()` guard — do NOT use `.astype(str).str.strip().pipe(...isin...)` (float NaN breaks `str.join()`).

## Word report — 7-page structure (`report_generator.py`)

| Page | Orientation | Content |
|---|---|---|
| 1 | Portrait | **Scope and Limitations** — 4 transparency caveats: (1) not a fraud detection tool; (2) line-item scope; (3) pre-calibrated weights; (4) declared weights are approximate. Rendered by `_page_caveats(doc)`, called first. |
| 2 | Portrait | **Executive Summary** — two body paragraphs + dataset overview table + summary of findings bullets. Opening paragraph: composite risk score, stratification, diversity controls (no technical detail), professional judgement caveat. Does NOT name T08, Jaccard threshold, or vendor cap. Closing paragraph: reason codes, line-to-voucher rollup, pointers to Excel tabs. Bullets include duplicate payment count and split purchase risk count. Dataset overview "Payment Voucher Period" uses `Voucher Accounting Date` min/max. |
| 3 | Portrait | **Methodology** — audit-grade standalone. Opening: purpose + cross-reference to Executive Summary (does NOT re-summarise). Then: Stage 1 feature engineering; Stage 2 five analytical methods with individual caveats; Stage 3 exact line-level formula + weight rationale table; Benford suppression rule; Stage 4 voucher rollup formula; ML consensus flag explanation; "Risk Tier Assignment and Sample Selection" subsection (percentile cutoffs, stratified draw, similarity deduplication, vendor cap, T08 de-prioritisation with dynamic T08 count via `t08_count` parameter). `_page2()` accepts `t08_count`; `export_word_report()` computes from `df_vouchers['is_t08_vendor'].sum()`. |
| 4 | Landscape | **Analytical Charts** — Benford's Law distribution + voucher risk score histogram, side by side in borderless 2-column table. |
| 5 | Landscape | **Payment Distribution & Timeline** — amount distribution (log scale) + monthly timeline (dual-axis bar/line), stacked full-width. |
| 6 | Landscape | **Vendor Analysis** — top 10 vendors by transaction count and by total amount. Total-amount chart x-axis uses `MaxNLocator(nbins=4)` — do NOT remove (prevents cluttered large SGD amounts). |
| 7 | Landscape | **Feature Reference Table** — two tables, 5 columns each: Feature, What It Measures, Threshold for Flagging, ML Models, Why It Matters. Column widths: 1.6/2.1/1.8/1.2/3.8 inches. Table 1 ("Features Used in ML Models"): 16 rows; ML Models = "IF, LOF, Z-score" for z-score features, "IF, LOF" for others; intro notes Spearman pruning may reduce active count. Table 2 ("Features Outside ML Models"): 1 row for Benford's Law. Rendered via `_render_feature_table()`. References footer: Nigrini (2012) and ACFE Fraud Examiners Manual only. |

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

- **Landscape sections:** created by `_set_landscape()` via `doc.add_section()`.
- **Charts:** generated as in-memory PNG `BytesIO` objects using matplotlib (Agg backend), embedded with `run.add_picture()`.
- **Borderless table layout:** `_remove_table_borders()` used for side-by-side chart layout.
- **Cell shading / border helpers:** `_shade_cell()` and `_remove_table_borders()` use lxml `find(qn(...))` directly — do NOT use `get_or_add_tblPr()` or `get_or_add_tcPr()` (removed in python-docx 1.x).
- **HTML dashboard (`src/dashboard.py`):** removed April 2025. Charts now embedded in Word report via matplotlib. `plotly` remains in `requirements.txt` but is unused.
