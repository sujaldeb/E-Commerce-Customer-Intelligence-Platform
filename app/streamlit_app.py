# ------------------------------------------------------------------ #
#  Streamlit Dashboard · E-Commerce Customer Intelligence Platform   #
#  Run with: streamlit run app/streamlit_app.py                      #
# ------------------------------------------------------------------ #

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import warnings

warnings.filterwarnings('ignore')

# ── Page config ────────────────────────────────────────────────────
st.set_page_config(
    page_title='Customer Intelligence Platform',
    page_icon=None,
    layout='wide',
    initial_sidebar_state='expanded'
)

# ── Theme — matches portfolio: dark bg, blue/purple accents ────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    .stApp {
        background-color: #0a0f1e;
        color: #e2e8f0;
    }
    [data-testid="stSidebar"] {
        background-color: #0d1326;
        border-right: 1px solid #1e2d4a;
    }
    [data-testid="stSidebar"] * {
        color: #94a3b8 !important;
    }
    .sidebar-title {
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #475569 !important;
        margin-bottom: 12px;
    }
    .main .block-container {
        padding: 2rem 2.5rem;
        max-width: 1400px;
    }
    h1 {
        font-size: 2.4rem !important;
        font-weight: 700 !important;
        color: #f1f5f9 !important;
        letter-spacing: -0.02em;
        margin-bottom: 0 !important;
    }
    .page-caption {
        font-size: 0.9rem;
        color: #64748b;
        margin-top: 4px;
        margin-bottom: 24px;
    }
    hr {
        border-color: #1e2d4a !important;
        margin: 1.5rem 0 !important;
    }
    .kpi-card {
        background: #111827;
        border: 1px solid #1e2d4a;
        border-radius: 12px;
        padding: 20px 24px;
        transition: border-color 0.2s;
    }
    .kpi-card:hover {
        border-color: #3b82f6;
    }
    .kpi-label {
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #64748b;
        margin-bottom: 8px;
    }
    .kpi-value {
        font-size: 1.9rem;
        font-weight: 700;
        color: #f1f5f9;
        font-family: 'DM Mono', monospace;
        letter-spacing: -0.02em;
    }
    h2, h3 {
        color: #f1f5f9 !important;
        font-weight: 600 !important;
        letter-spacing: -0.01em;
    }
    [data-testid="stDataFrame"] {
        border: 1px solid #1e2d4a;
        border-radius: 10px;
        overflow: hidden;
    }
    .stTextInput input {
        background-color: #111827 !important;
        border: 1px solid #1e2d4a !important;
        color: #e2e8f0 !important;
        border-radius: 8px !important;
        font-family: 'DM Mono', monospace !important;
    }
    .stTextInput input:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 2px rgba(59,130,246,0.15) !important;
    }
    .stRadio > div {
        gap: 4px;
    }
</style>
""", unsafe_allow_html=True)

# ── Shared Plotly theme ─────────────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor='#111827',
    plot_bgcolor='#111827',
    font=dict(family='DM Sans, sans-serif', color='#94a3b8', size=12),
    title_font=dict(color='#f1f5f9', size=15, family='DM Sans, sans-serif'),
    xaxis=dict(gridcolor='#1e2d4a', linecolor='#1e2d4a', tickcolor='#475569', zerolinecolor='#1e2d4a'),
    yaxis=dict(gridcolor='#1e2d4a', linecolor='#1e2d4a', tickcolor='#475569', zerolinecolor='#1e2d4a'),
    legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor='#1e2d4a', borderwidth=1, font=dict(color='#94a3b8')),
    margin=dict(l=16, r=16, t=48, b=16),
    hoverlabel=dict(bgcolor='#1e2d4a', bordercolor='#3b82f6', font=dict(color='#f1f5f9', family='DM Sans'))
)

MIXED_PALETTE = ['#3b82f6', '#8b5cf6', '#06b6d4', '#10b981', '#f59e0b', '#ef4444', '#ec4899', '#6366f1']

# ── Paths ──────────────────────────────────────────────────────────
BASE = r'E:\Projects\python\E-Commerce Customer Intelligence Platform\data\processed'

# ── Data loading ───────────────────────────────────────────────────
@st.cache_data
def load_data():
    clean    = pd.read_parquet(f'{BASE}\\retail_clean.parquet')
    rfm      = pd.read_parquet(f'{BASE}\\rfm_segments.parquet')
    cohort   = pd.read_parquet(f'{BASE}\\cohort_matrix.parquet')
    forecast = pd.read_parquet(f'{BASE}\\forecast.parquet')
    rules    = pd.read_parquet(f'{BASE}\\association_rules.parquet')
    geo      = pd.read_parquet(f'{BASE}\\geo_summary.parquet')
    return clean, rfm, cohort, forecast, rules, geo

clean, rfm, cohort, forecast, rules, geo = load_data()

# ── Sidebar ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">Navigation</div>', unsafe_allow_html=True)
    page = st.radio(
        '',
        ['Overview', 'RFM Segmentation', 'Cohort Retention', 'Forecasting', 'Market Basket', 'Geography'],
        label_visibility='collapsed'
    )
    st.divider()
    st.markdown('<div class="sidebar-title">Dataset</div>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:0.8rem;color:#475569;">UCI Online Retail II<br>Dec 2009 – Dec 2011</p>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# PAGE: Overview
# ═══════════════════════════════════════════════════════════════════
if page == 'Overview':
    st.title('Customer Intelligence Platform')
    st.markdown('<p class="page-caption">UCI Online Retail II &nbsp;·&nbsp; 776K transactions &nbsp;·&nbsp; Dec 2009 – Dec 2011</p>', unsafe_allow_html=True)

    clean['InvoiceDate'] = pd.to_datetime(clean['InvoiceDate'])
    order_totals = clean.groupby('Invoice')['Revenue'].sum()

    # KPI row
    c1, c2, c3, c4, c5 = st.columns(5)
    kpis = [
        ('Total Revenue',   f"£{clean['Revenue'].sum():,.0f}"),
        ('Transactions',    f"{clean['Invoice'].nunique():,}"),
        ('Customers',       f"{clean['Customer ID'].nunique():,}"),
        ('Products',        f"{clean['StockCode'].nunique():,}"),
        ('Avg Order Value', f"£{order_totals.mean():,.0f}"),
    ]
    for col, (label, value) in zip([c1, c2, c3, c4, c5], kpis):
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{value}</div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # Monthly revenue trend
    monthly = (
        clean.groupby('YearMonth')['Revenue']
        .sum().reset_index().sort_values('YearMonth')
    )
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly['YearMonth'], y=monthly['Revenue'],
        mode='lines', name='Revenue',
        line=dict(color='#3b82f6', width=2.5),
        fill='tozeroy',
        fillcolor='rgba(59,130,246,0.08)'
    ))
    fig.update_layout(
        **PLOT_LAYOUT,
        title='Monthly Revenue Trend',
        yaxis_tickprefix='£', yaxis_tickformat=',.0f',
        height=320
    )
    st.plotly_chart(fig, use_container_width=True)

    # Top products and countries
    col_a, col_b = st.columns(2)

    with col_a:
        top_p = (
            clean.groupby('Description')['Revenue']
            .sum().nlargest(10).reset_index().sort_values('Revenue')
        )
        fig2 = go.Figure(go.Bar(
            x=top_p['Revenue'], y=top_p['Description'],
            orientation='h',
            marker=dict(
                color=top_p['Revenue'],
                colorscale=[[0, '#1d4ed8'], [1, '#60a5fa']],
                showscale=False
            )
        ))
        fig2.update_layout(
            **PLOT_LAYOUT, title='Top 10 Products by Revenue',
            xaxis_tickprefix='£', xaxis_tickformat=',.0f', height=380
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col_b:
        top_c = (
            geo[geo['Country'] != 'United Kingdom']
            .nlargest(10, 'Revenue')[['Country', 'Revenue']].sort_values('Revenue')
        )
        fig3 = go.Figure(go.Bar(
            x=top_c['Revenue'], y=top_c['Country'],
            orientation='h',
            marker=dict(
                color=top_c['Revenue'],
                colorscale=[[0, '#5b21b6'], [1, '#a78bfa']],
                showscale=False
            )
        ))
        fig3.update_layout(
            **PLOT_LAYOUT, title='Top 10 Countries by Revenue (excl. UK)',
            xaxis_tickprefix='£', xaxis_tickformat=',.0f', height=380
        )
        st.plotly_chart(fig3, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════
# PAGE: RFM Segmentation
# ═══════════════════════════════════════════════════════════════════
elif page == 'RFM Segmentation':
    st.title('RFM Segmentation')
    st.markdown('<p class="page-caption">Rule-based segments and K-Means clustering validation</p>', unsafe_allow_html=True)
    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        seg_counts = rfm['Segment'].value_counts().reset_index()
        seg_counts.columns = ['Segment', 'Customers']
        seg_counts = seg_counts.sort_values('Customers')
        fig1 = go.Figure(go.Bar(
            x=seg_counts['Customers'], y=seg_counts['Segment'],
            orientation='h',
            marker=dict(
                color=seg_counts['Customers'],
                colorscale=[[0, '#1d4ed8'], [1, '#60a5fa']],
                showscale=False
            )
        ))
        fig1.update_layout(**PLOT_LAYOUT, title='Customer Count by Segment', height=380)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        cluster_counts = rfm['Cluster_Label'].value_counts().reset_index()
        cluster_counts.columns = ['Cluster', 'Customers']
        fig2 = go.Figure(go.Pie(
            labels=cluster_counts['Cluster'],
            values=cluster_counts['Customers'],
            hole=0.5,
            marker=dict(colors=MIXED_PALETTE),
            textfont=dict(color='#f1f5f9', size=11)
        ))
        fig2.update_layout(**PLOT_LAYOUT, title='K-Means Cluster Distribution (k=5)', height=380)
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    st.subheader('Segment Profiles')

    profile = (
        rfm.groupby('Segment')
        .agg(
            Customers     = ('Customer ID', 'count'),
            Avg_Recency   = ('Recency',     'mean'),
            Avg_Frequency = ('Frequency',   'mean'),
            Avg_Monetary  = ('Monetary',    'mean')
        )
        .round(1)
        .sort_values('Avg_Monetary', ascending=False)
        .reset_index()
    )
    st.dataframe(
        profile,
        use_container_width=True,
        hide_index=True,
        column_config={
            'Avg_Monetary': st.column_config.NumberColumn('Avg Monetary (£)', format='£%.0f'),
            'Avg_Recency':  st.column_config.NumberColumn('Avg Recency (days)'),
        }
    )

    st.divider()
    st.subheader('Customer Lookup')
    customer_id = st.text_input('Enter Customer ID', placeholder='e.g. 12347.0')
    if customer_id:
        result = rfm[rfm['Customer ID'] == customer_id]
        if len(result) > 0:
            st.dataframe(result, use_container_width=True, hide_index=True)
        else:
            st.warning('Customer ID not found.')

# ═══════════════════════════════════════════════════════════════════
# PAGE: Cohort Retention
# ═══════════════════════════════════════════════════════════════════
elif page == 'Cohort Retention':
    st.title('Cohort Retention Analysis')
    st.markdown('<p class="page-caption">Monthly retention rates by customer acquisition cohort</p>', unsafe_allow_html=True)
    st.divider()

    cohort_display = cohort.copy()
    cohort_display.index = cohort_display.index.astype(str)
    cohort_display.columns = cohort_display.columns.astype(str)

    fig = px.imshow(
        cohort_display,
        text_auto='.0%',
        color_continuous_scale=[[0, '#0d1326'], [0.25, '#1d4ed8'], [1, '#60a5fa']],
        zmin=0, zmax=0.5,
        title='Monthly Cohort Retention Rate',
        labels={'x': 'Months Since First Purchase', 'y': 'Cohort (Acquisition Month)'}
    )
    fig.update_layout(
        **PLOT_LAYOUT,
        height=680,
        coloraxis_colorbar=dict(
            title=dict(text='Retention', font=dict(color='#94a3b8')),
            tickformat='.0%',
            tickfont=dict(color='#94a3b8')
        )
    )
    fig.update_traces(textfont_size=8)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader('Average Retention Curve')

    avg_ret = cohort.mean(axis=0).dropna().reset_index()
    avg_ret.columns = ['Month', 'Retention']
    avg_ret = avg_ret[avg_ret['Month'] > 0]

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=avg_ret['Month'], y=avg_ret['Retention'],
        mode='lines+markers',
        line=dict(color='#8b5cf6', width=2.5),
        marker=dict(size=6, color='#8b5cf6'),
        fill='tozeroy',
        fillcolor='rgba(139,92,246,0.08)',
        name='Avg Retention'
    ))
    fig2.update_layout(
        **PLOT_LAYOUT,
        title='Average Retention Across All Cohorts (Month 1 onwards)',
        xaxis_title='Months Since First Purchase',
        yaxis_title='Retention Rate',
        yaxis_tickformat='.0%',
        height=320
    )
    st.plotly_chart(fig2, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════
# PAGE: Forecasting
# ═══════════════════════════════════════════════════════════════════
elif page == 'Forecasting':
    st.title('Revenue Forecasting')
    st.markdown('<p class="page-caption">12-week ahead forecast using Prophet with yearly seasonality</p>', unsafe_allow_html=True)
    st.divider()

    history = forecast[~forecast['Is_Future']]
    future  = forecast[forecast['Is_Future']].copy()
    future['Lower'] = future['Lower'].clip(lower=0)

    # Closed polygon for the confidence band
    band_x = list(future['Date']) + list(future['Date'][::-1])
    band_y = list(future['Upper']) + list(future['Lower'][::-1])

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=history['Date'], y=history['Forecast'],
        name='Actual (fitted)',
        mode='lines',
        line=dict(color='#475569', width=1.5, dash='dot')
    ))
    fig.add_trace(go.Scatter(
        x=band_x, y=band_y,
        fill='toself',
        fillcolor='rgba(59,130,246,0.12)',
        line=dict(color='rgba(59,130,246,0.2)', width=1),
        name='95% Interval'
    ))
    fig.add_trace(go.Scatter(
        x=future['Date'], y=future['Forecast'],
        name='Forecast',
        mode='lines+markers',
        line=dict(color='#3b82f6', width=2.5),
        marker=dict(size=6, color='#3b82f6', line=dict(color='#0d1326', width=1.5))
    ))
    fig.add_vline(
        x=history['Date'].max(),
        line=dict(color='#1e2d4a', width=1.5, dash='dash')
    )
    fig.update_layout(
        **PLOT_LAYOUT,
        title='12-Week Revenue Forecast',
        xaxis_title='Date',
        yaxis_title='Weekly Revenue (£)',
        yaxis_tickprefix='£',
        yaxis_tickformat=',.0f',
        height=480
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader('Forecast Values')

    forecast_table = future[['Date', 'Forecast', 'Lower', 'Upper']].copy()
    forecast_table.columns = ['Week', 'Forecast (£)', 'Lower Bound (£)', 'Upper Bound (£)']
    forecast_table['Week'] = forecast_table['Week'].dt.strftime('%Y-%m-%d')
    for col in ['Forecast (£)', 'Lower Bound (£)', 'Upper Bound (£)']:
        forecast_table[col] = forecast_table[col].round(0).astype(int)

    st.dataframe(
        forecast_table,
        use_container_width=True,
        hide_index=True,
        column_config={
            'Forecast (£)':    st.column_config.NumberColumn(format='£%d'),
            'Lower Bound (£)': st.column_config.NumberColumn(format='£%d'),
            'Upper Bound (£)': st.column_config.NumberColumn(format='£%d'),
        }
    )

# ═══════════════════════════════════════════════════════════════════
# PAGE: Market Basket
# ═══════════════════════════════════════════════════════════════════
elif page == 'Market Basket':
    st.title('Market Basket Analysis')
    st.markdown('<p class="page-caption">Product association rules derived using the Apriori algorithm</p>', unsafe_allow_html=True)
    st.divider()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Rules Generated</div>
            <div class="kpi-value">{len(rules):,}</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Avg Lift</div>
            <div class="kpi-value">{rules['lift'].mean():.1f}x</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Max Lift</div>
            <div class="kpi-value">{rules['lift'].max():.1f}x</div>
        </div>""", unsafe_allow_html=True)

    st.divider()

    col_a, col_b = st.columns(2)

    with col_a:
        top_rules = rules.nlargest(15, 'lift').copy()
        top_rules['label'] = (
            top_rules['antecedents'].str[:22] + ' → ' + top_rules['consequents'].str[:22]
        )
        top_rules = top_rules.sort_values('lift')
        fig1 = go.Figure(go.Bar(
            x=top_rules['lift'],
            y=top_rules['label'],
            orientation='h',
            marker=dict(
                color=top_rules['lift'],
                colorscale=[[0, '#1d4ed8'], [1, '#60a5fa']],
                showscale=False
            ),
            text=top_rules['lift'].round(1).astype(str) + 'x',
            textposition='outside',
            textfont=dict(color='#94a3b8', size=10)
        ))
        fig1.update_layout(
            **PLOT_LAYOUT, title='Top 15 Rules by Lift',
            xaxis_title='Lift', height=480
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col_b:
        fig2 = go.Figure(go.Scatter(
            x=rules['support'],
            y=rules['confidence'],
            mode='markers',
            marker=dict(
                size=8,
                color=rules['lift'],
                colorscale=[[0, '#1d4ed8'], [0.5, '#8b5cf6'], [1, '#f59e0b']],
                showscale=True,
                colorbar=dict(
                    title=dict(text='Lift', font=dict(color='#94a3b8')),
                    tickfont=dict(color='#94a3b8')
                ),
                opacity=0.8,
                line=dict(color='#0d1326', width=0.5)
            ),
            text=rules['antecedents'] + ' → ' + rules['consequents'],
            hovertemplate='<b>%{text}</b><br>Support: %{x:.3f}<br>Confidence: %{y:.3f}<extra></extra>'
        ))
        fig2.update_layout(
            **PLOT_LAYOUT,
            title='Support vs Confidence (coloured by Lift)',
            xaxis_title='Support',
            yaxis_title='Confidence',
            height=480
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    st.subheader('All Rules')

    min_lift = st.slider(
        'Minimum Lift', min_value=1.0,
        max_value=float(rules['lift'].max()),
        value=5.0, step=0.5
    )
    filtered = (
        rules[rules['lift'] >= min_lift]
        [['antecedents', 'consequents', 'support', 'confidence', 'lift']]
        .copy().reset_index(drop=True)
    )
    filtered.columns = ['Antecedent', 'Consequent', 'Support', 'Confidence', 'Lift']
    st.dataframe(
        filtered,
        use_container_width=True,
        hide_index=True,
        column_config={
            'Support':    st.column_config.NumberColumn(format='%.4f'),
            'Confidence': st.column_config.NumberColumn(format='%.4f'),
            'Lift':       st.column_config.NumberColumn(format='%.2f'),
        }
    )

# ═══════════════════════════════════════════════════════════════════
# PAGE: Geography
# ═══════════════════════════════════════════════════════════════════
elif page == 'Geography':
    st.title('Geographic Analysis')
    st.markdown('<p class="page-caption">Revenue and order behaviour by country</p>', unsafe_allow_html=True)
    st.divider()

    geo_map = geo.copy()
    geo_map['Country'] = geo_map['Country'].replace({'EIRE': 'Ireland'})

    fig_map = px.choropleth(
        geo_map,
        locations='Country',
        locationmode='country names',
        color='Revenue',
        hover_name='Country',
        hover_data={
            'Revenue':       ':,.0f',
            'Orders':        ':,',
            'Customers':     ':,',
            'AOV':           ':,.0f',
            'Revenue_Share': ':.2f'
        },
        color_continuous_scale=[[0, '#0d1326'], [0.15, '#1d4ed8'], [1, '#60a5fa']],
        title='Global Revenue Distribution'
    )
    fig_map.update_layout(
        **PLOT_LAYOUT,
        height=460,
        geo=dict(
            bgcolor='#0a0f1e',
            showframe=False,
            showcoastlines=True,
            coastlinecolor='#1e2d4a',
            showland=True,
            landcolor='#111827',
            showocean=True,
            oceancolor='#0a0f1e',
            showlakes=False,
            showcountries=True,
            countrycolor='#1e2d4a'
        ),
        coloraxis_colorbar=dict(
            title=dict(text='Revenue (£)', font=dict(color='#94a3b8')),
            tickformat=',.0f',
            tickprefix='£',
            tickfont=dict(color='#94a3b8')
        )
    )
    st.plotly_chart(fig_map, use_container_width=True)

    st.divider()

    col_a, col_b = st.columns(2)

    with col_a:
        intl = geo[geo['Country'] != 'United Kingdom'].nlargest(15, 'Revenue').sort_values('Revenue')
        fig2 = go.Figure(go.Bar(
            x=intl['Revenue'], y=intl['Country'],
            orientation='h',
            marker=dict(
                color=intl['Revenue'],
                colorscale=[[0, '#1d4ed8'], [1, '#60a5fa']],
                showscale=False
            )
        ))
        fig2.update_layout(
            **PLOT_LAYOUT,
            title='Top 15 International Markets by Revenue',
            xaxis_tickprefix='£', xaxis_tickformat=',.0f',
            height=440
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col_b:
        intl_aov = geo[geo['Country'] != 'United Kingdom'].nlargest(15, 'AOV').sort_values('AOV')
        fig3 = go.Figure(go.Bar(
            x=intl_aov['AOV'], y=intl_aov['Country'],
            orientation='h',
            marker=dict(
                color=intl_aov['AOV'],
                colorscale=[[0, '#5b21b6'], [1, '#a78bfa']],
                showscale=False
            )
        ))
        fig3.update_layout(
            **PLOT_LAYOUT,
            title='Average Order Value by Country (excl. UK)',
            xaxis_tickprefix='£', xaxis_tickformat=',.0f',
            height=440
        )
        st.plotly_chart(fig3, use_container_width=True)

    st.divider()
    st.subheader('Country Summary Table')

    display_geo = geo.copy()
    display_geo['Revenue'] = display_geo['Revenue'].round(0).astype(int)
    display_geo['AOV']     = display_geo['AOV'].round(0).astype(int)
    st.dataframe(
        display_geo,
        use_container_width=True,
        hide_index=True,
        column_config={
            'Revenue':       st.column_config.NumberColumn('Revenue (£)',  format='£%d'),
            'AOV':           st.column_config.NumberColumn('AOV (£)',      format='£%d'),
            'Revenue_Share': st.column_config.NumberColumn('Share (%)',    format='%.2f%%'),
        }
    )