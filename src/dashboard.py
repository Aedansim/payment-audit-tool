from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

AMOUNT_COL = 'Payment Voucher Amount (SGD, Excluding GST)'
BENFORD_EXPECTED = {d: np.log10(1 + 1 / d) for d in range(1, 10)}

_COLOURS = {
    'navy':   '#1F3864',
    'blue':   '#2E75B6',
    'orange': '#ED7D31',
    'red':    '#C00000',
    'green':  '#70AD47',
    'yellow': '#FFC000',
    'grey':   '#808080',
    'light_grey': '#F2F2F2',
}


# ---------------------------------------------------------------------------
# Individual chart builders
# ---------------------------------------------------------------------------

def _kpi_cards(df, selected):
    total_txns = len(df)
    total_amt  = df[AMOUNT_COL].sum()
    date_min   = df['Invoice Date'].min()
    date_max   = df['Invoice Date'].max()
    n_vendors  = df['Vendor ID'].nunique()
    n_indiv    = int(df.get('is_individual_payee', pd.Series(0)).sum())
    n_recurring = int(df.get('is_recurring_payment', pd.Series(0)).sum())
    high_risk  = int((df['risk_score'] >= selected['risk_score'].min()).sum()) if 'risk_score' in df.columns else 0

    kpis = [
        ("Total Transactions",   f"{total_txns:,}",        _COLOURS['navy']),
        ("Total Amount (SGD)",   f"{total_amt:,.0f}",       _COLOURS['blue']),
        ("Unique Vendors",       f"{n_vendors:,}",          _COLOURS['navy']),
        ("Individual Payees",    f"{n_indiv:,}",            _COLOURS['orange']),
        ("Recurring Payments",   f"{n_recurring:,}",        _COLOURS['grey']),
        ("Transactions Flagged", f"{high_risk:,}",          _COLOURS['red']),
        ("Samples Selected",     f"{len(selected)}",        _COLOURS['green']),
        ("Analysis Period",
         f"{date_min.strftime('%d %b %Y')} – {date_max.strftime('%d %b %Y')}" if pd.notna(date_min) else "N/A",
         _COLOURS['navy']),
    ]
    return kpis


def _fig_benford(benford_stats):
    digits = list(range(1, 10))
    obs_pct = [benford_stats['observed_pct'].get(d, 0) * 100 for d in digits]
    exp_pct = [BENFORD_EXPECTED[d] * 100 for d in digits]
    deviant = benford_stats['deviant_digits']

    bar_colors = [
        _COLOURS['red'] if d in deviant else _COLOURS['blue']
        for d in digits
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Observed', x=digits, y=obs_pct,
        marker_color=bar_colors,
        text=[f"{v:.1f}%" for v in obs_pct],
        textposition='outside',
    ))
    fig.add_trace(go.Scatter(
        name="Benford's Expected", x=digits, y=exp_pct,
        mode='lines+markers',
        line=dict(color=_COLOURS['orange'], width=2, dash='dash'),
        marker=dict(size=7),
    ))
    fig.update_layout(
        title=dict(text="Benford's Law — First Digit Distribution", font=dict(size=14)),
        xaxis=dict(title="First Digit", tickvals=digits, dtick=1),
        yaxis=dict(title="Frequency (%)", ticksuffix="%"),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(t=80, b=40, l=60, r=20),
        annotations=[dict(
            text=f"MAD: {benford_stats['mad']:.4f} | Verdict: {benford_stats['conformity']}",
            xref='paper', yref='paper', x=0.5, y=-0.18,
            showarrow=False, font=dict(size=11, color=_COLOURS['grey']),
        )],
    )
    return fig


def _fig_risk_distribution(df, cutoff_score):
    scores = df['risk_score'].dropna()
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=scores,
        nbinsx=50,
        marker_color=_COLOURS['blue'],
        name='All transactions',
        opacity=0.75,
    ))
    fig.add_vline(
        x=cutoff_score, line_dash='dash', line_color=_COLOURS['red'], line_width=2,
        annotation_text=f"Selection threshold ({cutoff_score:.3f})",
        annotation_position='top right',
        annotation_font=dict(color=_COLOURS['red']),
    )
    fig.update_layout(
        title=dict(text="Risk Score Distribution", font=dict(size=14)),
        xaxis=dict(title="Composite Risk Score"),
        yaxis=dict(title="Number of Transactions"),
        margin=dict(t=60, b=40, l=60, r=20),
        showlegend=False,
    )
    return fig


def _fig_amount_distribution(df):
    amounts = df[AMOUNT_COL].dropna()
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=np.log10(amounts.clip(lower=0.01)),
        nbinsx=60,
        marker_color=_COLOURS['navy'],
        opacity=0.8,
    ))
    tick_vals = [0, 1, 2, 3, 4, 5, 6]
    tick_text = ['1', '10', '100', '1K', '10K', '100K', '1M']
    fig.update_layout(
        title=dict(text="Payment Amount Distribution (log scale)", font=dict(size=14)),
        xaxis=dict(title="Amount (SGD)", tickvals=tick_vals, ticktext=tick_text),
        yaxis=dict(title="Number of Transactions"),
        margin=dict(t=60, b=40, l=60, r=20),
        showlegend=False,
    )
    return fig


def _fig_top_vendors(df):
    top_count = (
        df.groupby('Vendor Name')[AMOUNT_COL]
        .agg(Count='count', Total='sum')
        .nlargest(10, 'Count')
        .reset_index()
    )
    top_amt = (
        df.groupby('Vendor Name')[AMOUNT_COL]
        .agg(Count='count', Total='sum')
        .nlargest(10, 'Total')
        .reset_index()
    )

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Top 10 Vendors by Transaction Count",
                        "Top 10 Vendors by Total Amount (SGD)"),
        horizontal_spacing=0.12,
    )
    fig.add_trace(go.Bar(
        y=top_count['Vendor Name'], x=top_count['Count'],
        orientation='h', marker_color=_COLOURS['blue'],
        text=top_count['Count'], textposition='outside',
        name='Count',
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        y=top_amt['Vendor Name'], x=top_amt['Total'],
        orientation='h', marker_color=_COLOURS['navy'],
        text=[f"${v:,.0f}" for v in top_amt['Total']], textposition='outside',
        name='Total SGD',
    ), row=1, col=2)
    fig.update_layout(
        title=dict(text="Top Vendors Overview", font=dict(size=14)),
        showlegend=False,
        margin=dict(t=80, b=40, l=20, r=20),
        height=400,
    )
    fig.update_yaxes(autorange='reversed')
    return fig


def _fig_timeline(df):
    df2 = df.copy()
    df2['Month'] = df2['Invoice Date'].dt.to_period('M')
    monthly = (
        df2.groupby('Month')[AMOUNT_COL]
        .agg(Total='sum', Count='count')
        .reset_index()
    )
    monthly['Month_str'] = monthly['Month'].astype(str)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=monthly['Month_str'], y=monthly['Total'],
        name='Total Amount (SGD)', marker_color=_COLOURS['blue'], opacity=0.8,
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=monthly['Month_str'], y=monthly['Count'],
        name='Transaction Count', line=dict(color=_COLOURS['orange'], width=2),
        mode='lines+markers', marker=dict(size=5),
    ), secondary_y=True)
    fig.update_layout(
        title=dict(text="Monthly Payment Timeline", font=dict(size=14)),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(t=80, b=60, l=60, r=60),
        xaxis=dict(tickangle=-45),
    )
    fig.update_yaxes(title_text="Total Amount (SGD)", secondary_y=False)
    fig.update_yaxes(title_text="Transaction Count", secondary_y=True)
    return fig


def _fig_samples_table(selected):
    orig_cols = [
        'Sample #', 'Vendor Name', 'Cost Centre', AMOUNT_COL,
        'Invoice Date', 'risk_score', 'Selection Reasons',
    ]
    disp_cols = [c for c in orig_cols if c in selected.columns]
    sub = selected[disp_cols].copy()

    header_labels = {
        AMOUNT_COL: 'Amount (SGD)',
        'risk_score': 'Risk Score',
        'Invoice Date': 'Invoice Date',
    }
    display_headers = [header_labels.get(c, c) for c in disp_cols]

    # Format cells
    cell_values = []
    for col in disp_cols:
        vals = sub[col].tolist()
        if col == AMOUNT_COL:
            vals = [f"{v:,.2f}" if isinstance(v, (int, float)) and not pd.isna(v) else str(v)
                    for v in vals]
        elif col == 'risk_score':
            vals = [f"{v:.3f}" if isinstance(v, (int, float)) and not pd.isna(v) else str(v)
                    for v in vals]
        elif col == 'Invoice Date':
            vals = [v.strftime('%d/%m/%Y') if hasattr(v, 'strftime') else str(v)
                    for v in vals]
        else:
            vals = [str(v) if v is not None else '' for v in vals]
        cell_values.append(vals)

    n = len(sub)
    fill_colors = []
    top_third = n // 3
    for i, row in enumerate(sub.itertuples()):
        rank = i + 1
        if rank <= top_third:
            fill_colors.append('#FFB3B3')
        elif rank <= top_third * 2:
            fill_colors.append('#FFDDB3')
        else:
            fill_colors.append('#FFFAB3')

    row_fill = [fill_colors] * len(disp_cols)

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=display_headers,
            fill_color=_COLOURS['navy'],
            font=dict(color='white', size=11),
            align='left',
            height=30,
        ),
        cells=dict(
            values=cell_values,
            fill_color=row_fill,
            align='left',
            font=dict(size=10),
            height=28,
        ),
    )])
    fig.update_layout(
        title=dict(text=f"Selected Samples ({n} transactions)", font=dict(size=14)),
        margin=dict(t=60, b=20, l=10, r=10),
        height=max(400, n * 30 + 100),
    )
    return fig


# ---------------------------------------------------------------------------
# HTML assembly
# ---------------------------------------------------------------------------

def _kpi_html(kpis):
    cards = ""
    for label, value, color in kpis:
        cards += f"""
        <div class="kpi-card" style="border-top: 4px solid {color};">
            <div class="kpi-value" style="color:{color};">{value}</div>
            <div class="kpi-label">{label}</div>
        </div>"""
    return f'<div class="kpi-row">{cards}</div>'


def _fig_to_div(fig, div_id, first=False, height=None):
    if height:
        fig.update_layout(height=height)
    return fig.to_html(
        full_html=False,
        include_plotlyjs='cdn' if first else False,
        div_id=div_id,
        config={'responsive': True, 'displaylogo': False},
    )


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Payment Audit Dashboard</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{
    font-family: 'Segoe UI', Arial, sans-serif;
    background: #f4f6fb;
    margin: 0;
    padding: 0;
    color: #222;
  }}
  header {{
    background: #1F3864;
    color: white;
    padding: 24px 40px 18px;
  }}
  header h1 {{ margin: 0 0 6px; font-size: 24px; }}
  header p  {{ margin: 0; font-size: 13px; opacity: 0.8; }}
  .content  {{ padding: 28px 40px; max-width: 1400px; margin: 0 auto; }}
  .section-title {{
    font-size: 18px; font-weight: 700;
    color: #1F3864; margin: 32px 0 14px;
    border-bottom: 2px solid #1F3864; padding-bottom: 6px;
  }}
  .kpi-row {{
    display: flex; flex-wrap: wrap; gap: 16px; margin-bottom: 8px;
  }}
  .kpi-card {{
    background: white; border-radius: 8px; padding: 16px 22px;
    flex: 1 1 150px; min-width: 150px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.07);
  }}
  .kpi-value {{ font-size: 22px; font-weight: 700; margin-bottom: 4px; }}
  .kpi-label {{ font-size: 12px; color: #666; text-transform: uppercase; letter-spacing: 0.5px; }}
  .chart-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
    margin-bottom: 20px;
  }}
  .chart-card {{
    background: white; border-radius: 8px; padding: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.07);
  }}
  .chart-card-full {{
    background: white; border-radius: 8px; padding: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    margin-bottom: 20px;
  }}
  .legend-row {{
    display: flex; gap: 20px; margin: 8px 0 16px; font-size: 12px;
  }}
  .legend-item {{ display: flex; align-items: center; gap: 6px; }}
  .legend-dot {{
    width: 16px; height: 16px; border-radius: 3px; display: inline-block;
  }}
  footer {{
    text-align: center; padding: 20px; font-size: 12px;
    color: #999; border-top: 1px solid #ddd; margin-top: 40px;
  }}
</style>
{plotlyjs_cdn}
</head>
<body>

<header>
  <h1>Payment Transaction Audit Dashboard</h1>
  <p>{subtitle}</p>
</header>

<div class="content">

  <div class="section-title">Dataset Overview</div>
  {kpi_html}

  <div class="section-title">Risk Analysis</div>
  <div class="chart-grid">
    <div class="chart-card">{benford_div}</div>
    <div class="chart-card">{risk_dist_div}</div>
    <div class="chart-card">{amount_dist_div}</div>
    <div class="chart-card"></div>
  </div>

  <div class="section-title">Payment Trends</div>
  <div class="chart-card-full">{timeline_div}</div>

  <div class="section-title">Vendor Analysis</div>
  <div class="chart-card-full">{vendor_div}</div>

  <div class="section-title">Selected Samples</div>
  <div class="legend-row">
    <div class="legend-item">
      <span class="legend-dot" style="background:#FFB3B3;"></span> High risk (top third)
    </div>
    <div class="legend-item">
      <span class="legend-dot" style="background:#FFDDB3;"></span> Medium risk
    </div>
    <div class="legend-item">
      <span class="legend-dot" style="background:#FFFAB3;"></span> Lower risk
    </div>
  </div>
  <div class="chart-card-full">{samples_div}</div>

</div>

<footer>Generated by the Payment Transaction Audit Tool &nbsp;|&nbsp; {generated_at}</footer>
</body>
</html>
"""


def export_dashboard(df, selected, benford_stats, output_path):
    from datetime import datetime

    print("  Building dashboard charts...")
    kpis = _kpi_cards(df, selected)
    cutoff = float(selected['risk_score'].min()) if 'risk_score' in selected.columns else 0.0

    fig_benford    = _fig_benford(benford_stats)
    fig_risk       = _fig_risk_distribution(df, cutoff)
    fig_amount     = _fig_amount_distribution(df)
    fig_vendors    = _fig_top_vendors(df)
    fig_timeline   = _fig_timeline(df)
    fig_samples    = _fig_samples_table(selected)

    # Export figures to HTML divs (first one loads Plotly CDN)
    benford_div  = _fig_to_div(fig_benford,  'benford',  first=True,  height=380)
    risk_div     = _fig_to_div(fig_risk,     'risk',                  height=380)
    amount_div   = _fig_to_div(fig_amount,   'amount',                height=380)
    vendor_div   = _fig_to_div(fig_vendors,  'vendors',               height=450)
    timeline_div = _fig_to_div(fig_timeline, 'timeline',              height=360)
    samples_div  = _fig_to_div(fig_samples,  'samples')

    # Extract the <script> tag for Plotly CDN from first div
    import re
    cdn_match = re.search(r'<script[^>]*plotly[^>]*>.*?</script>', benford_div, re.DOTALL | re.IGNORECASE)
    plotlyjs_cdn = cdn_match.group(0) if cdn_match else ''
    # Remove cdn tag from benford_div to avoid duplication
    benford_div_clean = re.sub(r'<script[^>]*plotly[^>]*>.*?</script>', '', benford_div, flags=re.DOTALL | re.IGNORECASE)

    date_min = df['Invoice Date'].min()
    date_max = df['Invoice Date'].max()
    period_str = ""
    if pd.notna(date_min) and pd.notna(date_max):
        period_str = f" | Period: {date_min.strftime('%d %b %Y')} – {date_max.strftime('%d %b %Y')}"

    subtitle = (
        f"{len(df):,} transactions analysed | "
        f"{df['Vendor ID'].nunique():,} vendors{period_str} | "
        f"{len(selected)} samples selected"
    )

    html = HTML_TEMPLATE.format(
        plotlyjs_cdn=plotlyjs_cdn,
        subtitle=subtitle,
        kpi_html=_kpi_html(kpis),
        benford_div=benford_div_clean,
        risk_dist_div=risk_div,
        amount_dist_div=amount_div,
        vendor_div=vendor_div,
        timeline_div=timeline_div,
        samples_div=samples_div,
        generated_at=datetime.now().strftime('%d %b %Y %H:%M'),
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"  Dashboard saved: {output_path}")
