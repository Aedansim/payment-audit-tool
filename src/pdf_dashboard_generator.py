"""Static PDF dashboard — charts-only visual overview of the audit run.

A read-only companion to the Excel workbook (which remains the place for working
with the detailed, filterable data). Everything rendered here is derived from data
already computed by the pipeline: this module never changes a score, flag, weight,
rollup, tier, or selection result.

Privacy posture: PDF is chosen because it is inert — it cannot run scripts, make
network requests, or transmit data. Built with matplotlib's non-interactive Agg
backend and PdfPages only. No new dependencies, no network access at any point,
matplotlib's built-in fonts only.
"""

import re
import textwrap
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

AMOUNT_COL = 'Payment Voucher Amount (SGD, Excluding GST)'
BENFORD_EXPECTED = {d: np.log10(1 + 1 / d) for d in range(1, 10)}

# Landscape A4
_PAGE_W, _PAGE_H = 11.69, 8.27

# Palette — clean corporate / audit-professional
_NAVY       = '#1F3864'
_BLUE       = '#2E75B6'
_ORANGE     = '#ED7D31'
_GREY_TEXT  = '#595959'
_GREY_LIGHT = '#8C8C8C'
_RULE       = '#BFBFBF'
_CARD_BG    = '#F7F9FC'

_TIER_COLOURS = {'HIGH': '#C00000', 'MEDIUM': '#FF8C00', 'LOW': '#375623'}
_TIER_ORDER   = ['HIGH', 'MEDIUM', 'LOW']

_FOOTER_TEXT = (
    "Risk-based suggestions only - not a determination of fraud. Vendors in the Excluded "
    "vendors file are de-prioritised. Apply professional judgement. Generated locally; "
    "no data leaves this device."
)

# Content band: below the title rule, above the caption/footer band. The bottom
# must clear the axis label, which sits below the axes.
_C_TOP    = 0.840
_C_BOTTOM = 0.185
_C_HEIGHT = _C_TOP - _C_BOTTOM
_CONTENT_RECT = [0.075, _C_BOTTOM, 0.855, _C_HEIGHT]

_ACCT_PREFIX_RE = re.compile(r'^\[[^\]]*\]\s*')

# Canonical reason-code types, mirroring sample_selector._build_reason(). Several
# reason strings embed variable numbers ("Amount 3.2 std devs from ..."), so each
# type is matched on a stable substring. Order matters — most specific first.
_REASON_TYPES = [
    ("Payment to individual (NRIC/FIN)",      "Payment to individual"),
    ("Amount outlier vs vendor average",      "std devs from vendor average"),
    ("Amount outlier vs cost centre average", "std devs from cost centre average"),
    ("Round number amount",                   "Round number amount"),
    ("Weekend payment voucher",               "processed on weekend"),
    ("Amount just below approval threshold",  "just below approval threshold"),
    ("Repeated amount (irregular schedule)",  "Repeated amount for same vendor"),
    ("Potential duplicate payment",           "Potential duplicate payment"),
    ("Reversal / credit note",                "Reversal or credit note"),
    ("Split purchase risk",                   "Split purchase risk"),
    ("Possible transposed amount",            "Possible transposed amount"),
    ("Potential FY split purchase",           "Potential FY split purchase"),
    ("Unusual processing time",               "Unusual processing time"),
    ("Late payment",                          "Late payment"),
    ("Unusual description length",            "Unusual description length"),
    ("Benford's Law deviation",               "Benford's Law deviation"),
    ("Isolation Forest anomaly",              "Isolation Forest"),
    ("Local Outlier Factor anomaly",          "Local Outlier Factor"),
    ("Elevated composite risk score",         "Elevated composite risk score"),
]

_DUPLICATE_REASON = "Potential duplicate payment"


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_sgd(value):
    """Monetary values are always rendered 'SGD 1,234.56'."""
    try:
        return f"SGD {float(value):,.2f}"
    except (TypeError, ValueError):
        return "SGD 0.00"


def _fmt_date(value):
    """Dates are always rendered DD/MM/YYYY."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    try:
        if pd.isna(value):
            return "N/A"
    except (TypeError, ValueError):
        pass
    try:
        return pd.Timestamp(value).strftime('%d/%m/%Y')
    except (TypeError, ValueError):
        return "N/A"


def _fmt_datetime(value):
    try:
        return pd.Timestamp(value).strftime('%d/%m/%Y %H:%M')
    except (TypeError, ValueError):
        return "N/A"


def _truncate(text, n=34):
    s = str(text).strip()
    if not s or s.lower() in ('nan', 'none'):
        s = '(blank)'
    return s if len(s) <= n else s[:n - 1].rstrip() + '…'


def _is_blank(series):
    """True where a descriptive text column holds no usable value."""
    s = series.astype(str).str.strip()
    return s.eq('') | s.str.lower().isin(['nan', 'none', 'nat'])


# ---------------------------------------------------------------------------
# Page scaffolding
# ---------------------------------------------------------------------------

def _new_page(state, title=None, rule=True):
    """Create a landscape-A4 figure carrying the page title, caveat footer and
    page number. Returns the figure; callers add their own content axes."""
    fig = plt.figure(figsize=(_PAGE_W, _PAGE_H), facecolor='white')
    if title:
        fig.text(0.075, 0.935, title, ha='left', va='center',
                 fontsize=17, fontweight='bold', color=_NAVY)
    if rule:
        fig.add_artist(Line2D([0.075, 0.93], [0.898, 0.898],
                              color=_RULE, linewidth=1.0))

    state['page'] += 1
    fig.text(0.075, 0.038, _FOOTER_TEXT, ha='left', va='center',
             fontsize=6.2, color=_GREY_LIGHT)
    fig.text(0.93, 0.038, f"Page {state['page']}", ha='right', va='center',
             fontsize=7.0, color=_GREY_LIGHT)
    return fig


def _caption(fig, text):
    fig.text(0.075, 0.088, text, ha='left', va='center',
             fontsize=8.5, style='italic', color=_GREY_TEXT)


def _overlay_axes(fig):
    """Full-page axes in 0–1 coordinates for hand-drawn layouts (cards, messages)."""
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    ax.patch.set_visible(False)
    return ax


def _message_page(fig, text):
    """Graceful degradation: a friendly message instead of an empty/broken chart."""
    ax = _overlay_axes(fig)
    ax.text(0.5, 0.5, text, ha='center', va='center',
            fontsize=14, color=_GREY_TEXT, wrap=True)
    return ax


def _content_axes(fig, rect=None):
    ax = fig.add_axes(rect or _CONTENT_RECT)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(_RULE)
    ax.tick_params(colors=_GREY_TEXT, labelsize=9)
    return ax


def _wrap_value(text, wrap_at):
    """Wrap on whitespace where possible; fall back to breaking on separators so a
    long unbroken token (a filename, say) never splits mid-word."""
    s = str(text)
    soft = textwrap.fill(s, wrap_at, break_long_words=False)
    if max(len(line) for line in soft.split('\n')) <= wrap_at:
        return soft

    lines, cur = [], ''
    for token in re.split(r'(?<=[ _\-./,])', s):
        if cur and len(cur) + len(token) > wrap_at:
            lines.append(cur)
            cur = token
        else:
            cur += token
    if cur:
        lines.append(cur)

    out = []
    for line in lines:                       # hard-break anything still oversized
        while len(line) > wrap_at:
            out.append(line[:wrap_at])
            line = line[wrap_at:]
        out.append(line)
    return '\n'.join(out)


def _card(ax, x, y, w, h, label, value, value_size=14.5, wrap_at=19):
    """Bordered metric card: small grey label above a large navy value."""
    ax.add_patch(Rectangle((x, y), w, h, linewidth=1.1,
                           edgecolor=_RULE, facecolor=_CARD_BG, zorder=1))
    ax.text(x + w / 2, y + h * 0.74, str(label).upper(), ha='center', va='center',
            fontsize=7.5, fontweight='bold', color=_GREY_TEXT, zorder=2)
    wrapped = _wrap_value(value, wrap_at)
    n_lines = wrapped.count('\n') + 1
    size = value_size if n_lines == 1 else max(9.0, value_size - 3.0 * (n_lines - 1))
    ax.text(x + w / 2, y + h * 0.34, wrapped, ha='center', va='center',
            fontsize=size, fontweight='bold', color=_NAVY,
            linespacing=1.3, zorder=2)


def _money_axis(ax, axis='x'):
    fmt = mticker.FuncFormatter(lambda v, _: f'{v:,.0f}')
    (ax.xaxis if axis == 'x' else ax.yaxis).set_major_formatter(fmt)


# ---------------------------------------------------------------------------
# Reason-code parsing
# ---------------------------------------------------------------------------

def _classify_reason(text):
    for label, needle in _REASON_TYPES:
        if needle in text:
            return label
    return "Other"


def _tally_reason_codes(vouchers):
    """Count how often each canonical reason type appears across the given vouchers.

    Reason codes are ' | '-joined; multi-line vouchers prefix each entry with
    '[Account Code] ' and vendor-capped vouchers carry a trailing NOTE FOR AUDITOR
    fragment. Each type is counted once per voucher, so a count reads 'N of the M
    vouchers show this reason'.
    """
    counts = {}
    if 'voucher_reason_codes' not in vouchers.columns:
        return counts
    for raw in vouchers['voucher_reason_codes'].astype(str):
        seen = set()
        for part in raw.split(' | '):
            part = part.strip()
            if not part or part.upper().startswith('NOTE FOR AUDITOR'):
                continue
            seen.add(_classify_reason(_ACCT_PREFIX_RE.sub('', part)))
        for label in seen:
            counts[label] = counts.get(label, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Page 1 — Cover / summary metrics
# ---------------------------------------------------------------------------

def _cover_values(meta, df_scored, df_vouchers, selected_vouchers):
    meta = meta or {}

    if 'Payment Date' in df_scored.columns:
        pay_dates = pd.to_datetime(df_scored['Payment Date'], errors='coerce').dropna()
    else:
        pay_dates = pd.Series(dtype='datetime64[ns]')

    total_value = float(df_scored[AMOUNT_COL].sum()) if AMOUNT_COL in df_scored.columns else 0.0
    unique_vendors = int(df_scored['Vendor ID'].nunique()) if 'Vendor ID' in df_scored.columns else 0

    return {
        'dataset_name':       meta.get('dataset_name', 'N/A'),
        'period_start':       meta.get('period_start', pay_dates.min() if len(pay_dates) else None),
        'period_end':         meta.get('period_end',   pay_dates.max() if len(pay_dates) else None),
        'generated_at':       meta.get('generated_at', datetime.now()),
        'total_transactions': meta.get('total_transactions', len(df_scored)),
        'total_vouchers':     meta.get('total_vouchers',     len(df_vouchers)),
        'unique_vendors':     meta.get('unique_vendors',     unique_vendors),
        'total_value':        meta.get('total_value',        total_value),
        'sample_size':        meta.get('sample_size',        len(selected_vouchers)),
    }


def _page_cover(pdf, state, meta, df_scored, df_vouchers, selected_vouchers):
    fig = _new_page(state, rule=False)
    ax = _overlay_axes(fig)
    v = _cover_values(meta, df_scored, df_vouchers, selected_vouchers)

    ax.text(0.5, 0.855, "Payment Transaction Audit — Dashboard",
            ha='center', va='center', fontsize=26, fontweight='bold', color=_NAVY)
    ax.text(0.5, 0.795,
            "Risk-based sample selection — visual overview. "
            "Detailed, filterable data remains in the Excel workbook.",
            ha='center', va='center', fontsize=10.5, color=_GREY_TEXT)
    ax.add_line(Line2D([0.30, 0.70], [0.755, 0.755], color=_RULE, linewidth=1.0))

    period = (f"{_fmt_date(v['period_start'])} – {_fmt_date(v['period_end'])}"
              if v['period_start'] is not None and v['period_end'] is not None else "N/A")

    cards = [
        ("Dataset",            str(v['dataset_name'])),
        ("Payment Period",     period),
        ("Generated",          _fmt_datetime(v['generated_at'])),
        ("Sample Size Selected", f"{int(v['sample_size']):,}"),
        ("Total Transactions", f"{int(v['total_transactions']):,}"),
        ("Total Vouchers",     f"{int(v['total_vouchers']):,}"),
        ("Unique Vendors",     f"{int(v['unique_vendors']):,}"),
        ("Total Value",        _fmt_sgd(v['total_value'])),
    ]

    x0, x1 = 0.075, 0.93
    gap_x, gap_y = 0.028, 0.055
    n_cols = 4
    w = (x1 - x0 - gap_x * (n_cols - 1)) / n_cols
    h = 0.185
    top_y = 0.475

    for i, (label, value) in enumerate(cards):
        row, col = divmod(i, n_cols)
        x = x0 + col * (w + gap_x)
        y = top_y - row * (h + gap_y)
        _card(ax, x, y, w, h, label, value)

    pdf.savefig(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Page 2 — Risk tier distribution
# ---------------------------------------------------------------------------

def _page_risk_tiers(pdf, state, df_vouchers):
    fig = _new_page(state, "Risk Tier Distribution (all vouchers)")

    if 'voucher_risk_tier' not in df_vouchers.columns or df_vouchers.empty:
        _message_page(fig, "No voucher data available for tier analysis.")
        pdf.savefig(fig)
        plt.close(fig)
        return

    counts = (df_vouchers['voucher_risk_tier']
              .value_counts()
              .reindex(_TIER_ORDER, fill_value=0))
    total = int(counts.sum())
    present = [t for t in _TIER_ORDER if counts[t] > 0]

    if total == 0 or not present:
        _message_page(fig, "No vouchers were assigned a risk tier.")
        pdf.savefig(fig)
        plt.close(fig)
        return

    ax = fig.add_axes([0.16, 0.16, 0.42, 0.66])
    values = [int(counts[t]) for t in present]
    wedges, _ = ax.pie(
        values,
        colors=[_TIER_COLOURS[t] for t in present],
        startangle=90, counterclock=False,
        wedgeprops={'width': 0.38, 'edgecolor': 'white', 'linewidth': 2},
    )
    ax.axis('equal')
    ax.text(0, 0.08, f"{total:,}", ha='center', va='center',
            fontsize=22, fontweight='bold', color=_NAVY)
    ax.text(0, -0.14, "vouchers", ha='center', va='center',
            fontsize=10, color=_GREY_TEXT)

    legend_ax = fig.add_axes([0.60, 0.16, 0.33, 0.66])
    legend_ax.set_axis_off()
    legend_ax.legend(
        wedges,
        [f"{t}  —  {int(counts[t]):,} vouchers ({counts[t] / total * 100:.1f}%)"
         for t in present],
        loc='center left', frameon=False, fontsize=12,
        handlelength=1.4, handleheight=1.4, labelspacing=1.5,
    )

    _caption(fig,
             "HIGH = top 5% of voucher risk scores, MEDIUM = next 15%, LOW = the remainder. "
             "All HIGH-tier vouchers are mandatory candidates for the audit sample.")
    pdf.savefig(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Page 3 — Composite score distribution
# ---------------------------------------------------------------------------

def _page_score_distribution(pdf, state, df_vouchers):
    fig = _new_page(state, "Voucher Risk Score Distribution")

    scores = (df_vouchers['voucher_score'].dropna()
              if 'voucher_score' in df_vouchers.columns else pd.Series(dtype=float))
    if scores.empty:
        _message_page(fig, "No voucher risk scores available to plot.")
        pdf.savefig(fig)
        plt.close(fig)
        return

    # Same cut-offs sample_selector._assign_risk_tier() uses, recomputed from the
    # same column so the markers can never drift from the assigned tiers.
    med_cut = float(scores.quantile(0.80))
    high_cut = float(scores.quantile(0.95))

    ax = _content_axes(fig)
    ax.hist(scores, bins=50, color=_BLUE, alpha=0.80, edgecolor='white', linewidth=0.6)
    ax.axvline(med_cut, color=_TIER_COLOURS['MEDIUM'], linestyle='--', linewidth=1.8,
               label=f"MEDIUM boundary — 80th percentile ({med_cut:.3f})")
    ax.axvline(high_cut, color=_TIER_COLOURS['HIGH'], linestyle='--', linewidth=1.8,
               label=f"HIGH boundary — 95th percentile ({high_cut:.3f})")
    ax.set_xlabel('Composite Voucher Risk Score', fontsize=10, color=_GREY_TEXT)
    ax.set_ylabel('Number of Vouchers', fontsize=10, color=_GREY_TEXT)
    ax.grid(axis='y', alpha=0.25, color=_RULE)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9.5, frameon=False, loc='upper right')

    _caption(fig,
             "Selected samples are drawn from the top of this distribution. "
             "The long left tail is the routine population; the thin right tail carries the anomalies.")
    pdf.savefig(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Page 4 — Reason code frequency
# ---------------------------------------------------------------------------

def _page_reason_codes(pdf, state, selected_vouchers):
    fig = _new_page(state, "What Is Driving the Selected Samples")

    counts = _tally_reason_codes(selected_vouchers)
    counts = {k: v for k, v in counts.items() if v > 0}
    if not counts:
        _message_page(fig, "No reason codes recorded for the selected samples.")
        pdf.savefig(fig)
        plt.close(fig)
        return

    items = sorted(counts.items(), key=lambda kv: kv[1])   # ascending → barh reads desc
    labels = [k for k, _ in items]
    values = [v for _, v in items]
    n_sel = len(selected_vouchers)

    ax = _content_axes(fig, [0.30, _C_BOTTOM, 0.63, _C_HEIGHT])
    y = np.arange(len(labels))
    ax.barh(y, values, color=_NAVY, alpha=0.88, height=0.68)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.set_xlabel('Number of selected samples in which the reason appears',
                  fontsize=10, color=_GREY_TEXT)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.set_xlim(0, max(values) * 1.14 + 0.5)
    ax.grid(axis='x', alpha=0.25, color=_RULE)
    ax.set_axisbelow(True)

    for yi, val in zip(y, values):
        ax.text(val + max(values) * 0.015, yi, f"{val:,}",
                va='center', ha='left', fontsize=9, fontweight='bold', color=_NAVY)

    _caption(fig,
             f"Counted across the {n_sel:,} selected samples; each reason type is counted once "
             f"per voucher. A voucher usually triggers several reasons.")
    pdf.savefig(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Page 5 — Benford's Law
# ---------------------------------------------------------------------------

def _benford_interpretation(mad, p_value):
    """Same three-branch guidance as the Excel Benford tab's Key Takeaway — keep
    the two files in step if either is reworded."""
    if p_value >= 0.05:
        return (
            f"MAD of {mad:.4f} with a non-significant p-value (p = {p_value:.4f}) indicates no "
            "statistically significant deviation from Benford's expected distribution at this "
            "dataset size. Benford's Law signals are advisory only — the composite risk score "
            "is driven primarily by the other components. The absence of a significant result "
            "does not confirm data integrity."
        )
    if mad > 0.015:
        return (
            f"MAD of {mad:.4f} with a statistically significant p-value (p = {p_value:.4f}) "
            "indicates the distortion is large enough to be visible at the aggregate level. "
            "Anomalies are either widespread, or concentrated transactions are extreme enough to "
            "drag the overall distribution. A broader review of the dataset is warranted — not "
            "just a focus on a few deviant digit groups."
        )
    return (
        f"MAD of {mad:.4f} with a statistically significant p-value (p = {p_value:.4f}) indicates "
        "the overall distribution still looks broadly healthy, but the anomaly is real and subtle. "
        "Fewer transactions are likely involved, or any manipulation is more targeted. The audit "
        "response should be surgical: focus on patterns within the flagged digit groups rather "
        "than the dataset as a whole."
    )


def _page_benford(pdf, state, benford_stats):
    fig = _new_page(state, "Benford's Law - First Digit Analysis")

    stats = benford_stats or {}
    observed = stats.get('observed_pct')
    if observed is None or len(observed) == 0:
        _message_page(fig, "Benford's Law analysis not available for this dataset.")
        pdf.savefig(fig)
        plt.close(fig)
        return

    digits = list(range(1, 10))
    obs_pct = [float(observed.get(d, 0)) * 100 for d in digits]
    exp_pct = [BENFORD_EXPECTED[d] * 100 for d in digits]
    deviant = list(stats.get('deviant_digits', []))

    ax = _content_axes(fig, [0.075, _C_BOTTOM, 0.53, _C_HEIGHT])
    x = np.arange(len(digits))
    width = 0.40
    ax.bar(x - width / 2, obs_pct, width, label='Observed',
           color=[_TIER_COLOURS['HIGH'] if d in deviant else _BLUE for d in digits],
           alpha=0.90)
    ax.bar(x + width / 2, exp_pct, width, label="Benford's expected",
           color=_ORANGE, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(digits)
    ax.set_xlabel('First Digit', fontsize=10, color=_GREY_TEXT)
    ax.set_ylabel('Frequency (%)', fontsize=10, color=_GREY_TEXT)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:.0f}%'))
    ax.grid(axis='y', alpha=0.25, color=_RULE)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9.5, frameon=False)

    panel = fig.add_axes([0.635, _C_BOTTOM, 0.295, _C_HEIGHT])
    panel.set_axis_off()
    panel.set_xlim(0, 1)
    panel.set_ylim(0, 1)
    panel.add_patch(Rectangle((0, 0), 1, 1, linewidth=1.1,
                              edgecolor=_RULE, facecolor=_CARD_BG))

    mad = float(stats.get('mad', 0.0))
    p_value = float(stats.get('p_value', 1.0))
    conformity = stats.get('conformity', 'N/A')
    n_analyzed = int(stats.get('n_analyzed', 0))

    panel.text(0.5, 0.945, "TEST RESULTS", ha='center', va='center',
               fontsize=8.5, fontweight='bold', color=_GREY_TEXT)
    rows = [
        ("MAD", f"{mad:.4f}"),
        ("Conformity verdict", str(conformity)),
        ("Chi-square p-value", f"{p_value:.4f}"),
        ("Transactions analysed", f"{n_analyzed:,}"),
        ("Most deviant digits", ", ".join(str(d) for d in deviant) if deviant else "—"),
    ]
    y = 0.885
    for label, value in rows:
        panel.text(0.06, y, label, ha='left', va='center', fontsize=9, color=_GREY_TEXT)
        panel.text(0.94, y, value, ha='right', va='center',
                   fontsize=9.5, fontweight='bold', color=_NAVY)
        y -= 0.062

    panel.add_line(Line2D([0.06, 0.94], [y + 0.018, y + 0.018],
                          color=_RULE, linewidth=0.9))
    panel.text(0.06, y - 0.045, "INTERPRETATION", ha='left', va='center',
               fontsize=8.5, fontweight='bold', color=_GREY_TEXT)
    panel.text(0.06, y - 0.09, textwrap.fill(_benford_interpretation(mad, p_value), 46),
               ha='left', va='top', fontsize=8.2, color=_NAVY, linespacing=1.5)

    _caption(fig,
             "Recurring payments are excluded from the Benford population. Benford's Law carries "
             "only a 5% weight in the composite score and cannot select a voucher on its own.")
    pdf.savefig(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Page 6 — Vendor risk concentration (whole dataset)
# ---------------------------------------------------------------------------

def _page_vendor_risk(pdf, state, df_vouchers, top_n=20):
    fig = _new_page(state, "Vendor Risk Concentration (entire dataset)")

    needed = {'Vendor Name', 'voucher_score'}
    if not needed.issubset(df_vouchers.columns) or df_vouchers.empty:
        _message_page(fig, "No voucher data available for vendor risk aggregation.")
        pdf.savefig(fig)
        plt.close(fig)
        return

    work = df_vouchers.copy()
    work['_vendor'] = work['Vendor Name'].astype(str).str.strip().replace('', '(unnamed vendor)')
    if 'voucher_total_amount' not in work.columns:
        work['voucher_total_amount'] = 0.0

    agg = (work.groupby('_vendor')
           .agg(agg_score=('voucher_score', 'sum'),
                n_vouchers=('voucher_score', 'size'),
                total_amount=('voucher_total_amount', 'sum'))
           .sort_values('agg_score', ascending=False)
           .head(top_n))

    if agg.empty or float(agg['agg_score'].max()) <= 0:
        _message_page(fig, "No aggregate vendor risk could be computed for this dataset.")
        pdf.savefig(fig)
        plt.close(fig)
        return

    agg = agg.iloc[::-1]           # ascending → barh reads highest at the top
    scores = agg['agg_score'].to_numpy(dtype=float)

    cmap = matplotlib.colormaps['YlOrRd']
    norm = Normalize(vmin=float(scores.min()), vmax=float(scores.max()))
    # Restrict to the upper half of the colormap so even the lowest bar stays visible.
    colours = cmap(0.30 + 0.65 * norm(scores))

    ax = _content_axes(fig, [0.255, _C_BOTTOM, 0.60, _C_HEIGHT])
    y = np.arange(len(agg))
    ax.barh(y, scores, color=colours, height=0.72,
            edgecolor=_RULE, linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels([_truncate(v, 30) for v in agg.index], fontsize=8.5)
    ax.set_xlabel('Aggregate risk score (sum of voucher scores)',
                  fontsize=10, color=_GREY_TEXT)
    ax.set_xlim(0, float(scores.max()) * 1.42)
    ax.grid(axis='x', alpha=0.25, color=_RULE)
    ax.set_axisbelow(True)

    for yi, (_, row) in zip(y, agg.iterrows()):
        ax.text(row['agg_score'] + scores.max() * 0.015, yi,
                f"{row['agg_score']:.2f}   "
                f"{int(row['n_vouchers']):,} vch · {_fmt_sgd(row['total_amount'])}",
                va='center', ha='left', fontsize=7.8, color=_NAVY)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.028, pad=0.02)
    cbar.set_label('Aggregate risk score', fontsize=8.5, color=_GREY_TEXT)
    cbar.ax.tick_params(labelsize=7.5, colors=_GREY_TEXT)
    cbar.outline.set_edgecolor(_RULE)

    _caption(fig,
             "Vendors with high aggregate risk here would generally be represented in the selected "
             "sample, subject to the two-vouchers-per-vendor diversity cap.")
    pdf.savefig(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Pages 7 & 8 — spend concentration
# ---------------------------------------------------------------------------

def _spend_bar_page(pdf, state, title, series_labels, series_values,
                    xlabel, caption, empty_message, top_n=15):
    fig = _new_page(state, title)

    if series_labels is None or len(series_labels) == 0:
        _message_page(fig, empty_message)
        pdf.savefig(fig)
        plt.close(fig)
        return

    labels = list(series_labels)[::-1]
    values = np.asarray(list(series_values), dtype=float)[::-1]

    ax = _content_axes(fig, [0.30, _C_BOTTOM, 0.60, _C_HEIGHT])
    y = np.arange(len(labels))
    ax.barh(y, values, color=_NAVY, alpha=0.88, height=0.68)
    ax.set_yticks(y)
    ax.set_yticklabels([_truncate(v, 34) for v in labels], fontsize=8.5)
    ax.set_xlabel(xlabel, fontsize=10, color=_GREY_TEXT)
    _money_axis(ax, 'x')
    ax.set_xlim(0, float(values.max()) * 1.30)
    ax.grid(axis='x', alpha=0.25, color=_RULE)
    ax.set_axisbelow(True)

    for yi, val in zip(y, values):
        ax.text(val + values.max() * 0.012, yi, _fmt_sgd(val),
                va='center', ha='left', fontsize=8, color=_NAVY)

    _caption(fig, caption)
    pdf.savefig(fig)
    plt.close(fig)


def _positive_spend(df_scored, group_col, top_n=15):
    """Top-N groups by total positive payment amount. Returns (labels, values)."""
    if group_col not in df_scored.columns or AMOUNT_COL not in df_scored.columns:
        return [], []
    work = df_scored[df_scored[AMOUNT_COL] > 0].copy()
    if work.empty:
        return [], []
    work = work[~_is_blank(work[group_col])]
    if work.empty:
        return [], []
    work['_key'] = work[group_col].astype(str).str.strip()
    totals = (work.groupby('_key')[AMOUNT_COL].sum()
              .sort_values(ascending=False).head(top_n))
    return totals.index.tolist(), totals.to_numpy(dtype=float).tolist()


def _page_top_vendor_spend(pdf, state, df_scored):
    labels, values = _positive_spend(df_scored, 'Vendor Name', top_n=15)
    _spend_bar_page(
        pdf, state, "Top Vendors by Total Spend", labels, values,
        xlabel='Total spend, positive amounts (SGD)',
        caption="Total positive payment value per vendor across the whole dataset. "
                "Spend size is not itself a risk signal — read alongside the vendor risk "
                "concentration page.",
        empty_message="No positive payment amounts available to summarise vendor spend.",
    )


def _page_spend_categories(pdf, state, df_scored):
    labels, values = _positive_spend(df_scored, 'Account Description', top_n=15)
    _spend_bar_page(
        pdf, state, "Top Spend Categories (by Account Description)", labels, values,
        xlabel='Total spend, positive amounts (SGD)',
        caption="Total positive payment value per account description across the whole dataset. "
                "Account Description is reference data only and is never used in scoring.",
        empty_message="Account Description not available in this dataset.",
    )


# ---------------------------------------------------------------------------
# Page 9 — Potential duplicate payments
# ---------------------------------------------------------------------------

def _page_duplicates(pdf, state, df_vouchers, selected_vouchers):
    fig = _new_page(state, "Potential Duplicate Payments")

    if 'voucher_reason_codes' not in df_vouchers.columns or df_vouchers.empty:
        _message_page(fig, "No potential duplicate payments identified in this dataset.")
        pdf.savefig(fig)
        plt.close(fig)
        return

    mask = df_vouchers['voucher_reason_codes'].astype(str).str.contains(
        _DUPLICATE_REASON, regex=False, na=False)
    flagged = df_vouchers[mask]
    n_flagged = int(len(flagged))

    if n_flagged == 0:
        _message_page(fig, "No potential duplicate payments identified in this dataset.")
        pdf.savefig(fig)
        plt.close(fig)
        return

    total_value = (float(flagged['voucher_total_amount'].sum())
                   if 'voucher_total_amount' in flagged.columns else 0.0)
    if 'Voucher ID' in selected_vouchers.columns and 'Voucher ID' in flagged.columns:
        n_in_sample = int(flagged['Voucher ID'].isin(
            set(selected_vouchers['Voucher ID'])).sum())
    else:
        n_in_sample = 0

    ax = _overlay_axes(fig)
    ax.text(0.075, 0.825, textwrap.fill(
                "Vouchers whose reason codes flag a potential duplicate payment — same vendor, "
                "invoice number, invoice date and amount appearing in more than one voucher.", 96),
            ha='left', va='top', fontsize=10.5, color=_GREY_TEXT, linespacing=1.5)

    cards = [
        ("Vouchers Flagged",       f"{n_flagged:,}"),
        ("Total Value",            _fmt_sgd(total_value)),
        ("Included in Audit Sample", f"{n_in_sample:,} of {n_flagged:,}"),
    ]
    x0, x1 = 0.075, 0.93
    gap = 0.035
    w = (x1 - x0 - gap * (len(cards) - 1)) / len(cards)
    h = 0.26
    y = 0.44
    for i, (label, value) in enumerate(cards):
        _card(ax, x0 + i * (w + gap), y, w, h, label, value, value_size=22, wrap_at=16)

    ax.text(0.075, 0.335,
            "Figures cover all vouchers in the dataset, not only the selected samples. "
            "Full detail in the Excel workbook.",
            ha='left', va='center', fontsize=10, color=_GREY_TEXT)

    _caption(fig,
             "A duplicate flag is a prompt to check, not a finding — legitimate reasons include "
             "re-issued vouchers and instalment billing against a single invoice reference.")
    pdf.savefig(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def export_pdf_dashboard(df_scored, df_vouchers, selected_vouchers, benford_stats,
                         excluded, meta, output_path):
    """Write the nine-page static PDF dashboard to output_path.

    Parameters
    ----------
    df_scored, df_vouchers, selected_vouchers : DataFrame
        Outputs of sample_selector.select_samples(), used read-only.
    benford_stats : dict
        stats dict from benfords_law.analyze().
    excluded : ExcludedVendors or None
        Present for signature completeness; the exclusion posture is stated in the
        page footer. No filtering is applied here.
    meta : dict or None
        dataset_name, period_start, period_end, generated_at, total_transactions,
        total_vouchers, unique_vendors, total_value, sample_size. Any missing key is
        derived from the dataframes.
    output_path : str
    """
    state = {'page': 0}

    with PdfPages(output_path) as pdf:
        _page_cover(pdf, state, meta, df_scored, df_vouchers, selected_vouchers)
        _page_risk_tiers(pdf, state, df_vouchers)
        _page_score_distribution(pdf, state, df_vouchers)
        _page_reason_codes(pdf, state, selected_vouchers)
        _page_benford(pdf, state, benford_stats)
        _page_vendor_risk(pdf, state, df_vouchers)
        _page_top_vendor_spend(pdf, state, df_scored)
        _page_spend_categories(pdf, state, df_scored)
        _page_duplicates(pdf, state, df_vouchers, selected_vouchers)

        pdf.infodict().update({
            'Title': 'Payment Transaction Audit — Dashboard',
            'Subject': 'Risk-based audit sample selection — visual overview',
        })

    print(f"  PDF dashboard written: {state['page']} pages.")
    return output_path
