%%writefile app.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from io import BytesIO

st.set_page_config(page_title='Weather and Wait Times for Uber and Lyft', layout='wide')

@st.cache_data(show_spinner=False)
def load_results():
    return pd.read_csv('weather_wait_result.csv')

@st.cache_data(show_spinner=False)
def load_period_results():
    return pd.read_csv('weather_wait_period_result.csv')

@st.cache_data(show_spinner=False)
def load_trips():
    return pd.read_csv(
        'weather_wait_analysis.csv',
        usecols=['request_datetime','month','platform',
                 'pickup_zone','condition','wait_minutes'])

def multiselect(label, options, key_prefix):
    st.markdown(f"**{label}**")
    col1, col2 = st.columns(2)
    state_keys = [f'{key_prefix}-{label}-{opt}' for opt in options]

    for opt, k in zip(options, state_keys):
        if k not in st.session_state:
            st.session_state[k]=True
            
    if col1.button('All', key=f'all-{key_prefix}-{label}'):
        for k in state_keys:
            st.session_state[k]=True
    if col2.button('None', key=f'none-{key_prefix}-{label}'):
        for k in state_keys:
            st.session_state[k]=False
    return[
        opt
        for opt, k in zip(options, state_keys)
        if st.checkbox(str(opt), key=k)
    ]

def select_results(data, month, platform, pickup_zone):
    return data.loc[
        data['month'].isin(month) &
        data['platform'].isin(platform) &
        data['pickup_zone'].isin(pickup_zone) 
    ].copy()

def summarize_all_months(data):
    return(
        data.groupby(['platform','pickup_zone','period'],
        as_index=False,
        observed=True)
    .agg(
        median_diff=('median_diff','median'),
        p90_diff=('p90_diff','median'),
        months_used=('month','nunique')
    ))

def show_chart(fig):
    buffer = BytesIO()
    fig.patch.set_alpha(1)
    fig.savefig(
        buffer, format='png', dpi=300, bbox_inches='tight',transparent=False, facecolor='white'
    )
    fig.patch.set_alpha(0)
    st.pyplot(fig, bbox_inches='tight')
    st.download_button(
        label='Download chart',
        data=buffer.getvalue(),
        file_name='weather_wait_chart.png',
        mime='image/png')
    plt.close(fig)
    

def style(fig, axes):
    fig.patch.set_alpha(0)
    ax_list = axes.ravel() if isinstance(axes, np.ndarray) else [axes]
    for ax in ax_list:
        ax.patch.set_alpha(0)
        legend = ax.get_legend()
        if legend is not None:
            legend.get_frame().set_alpha(0)

def box_chart(trips, month, platform, pickup_zone):
    case = trips.loc[
        (trips['month']==month) &
        (trips['platform']==platform) &
        (trips['pickup_zone']==pickup_zone)  
    ]
    fig, ax = plt.subplots(figsize=(8,5))
    sns.boxplot(
        data=case, x='condition', y='wait_minutes',
        order=['clear','rain'], whis=(10,90),
        showfliers=False, ax=ax
        )
    ax.set_xlabel('Weather Condition')
    ax.set_ylabel('Wait Time (minutes)')
    ax.set_title(f'{pickup_zone} | {platform} | Month {month}')
    style(fig, ax)
    show_chart(fig)
    st.caption('Whiskers show the 10th and 90th percentiles.')
    

def comparison_chart(selected, view, metric, all_results):
    grid = selected.pivot(
        index=['pickup_zone','platform'],
        columns='month',
        values=metric
    )
    if view == 'Heatmap':
        fig, ax = plt.subplots(figsize=(9, 6))
        limit = all_results[metric].abs().max()
        sns.heatmap(
            grid, annot=True, fmt='.2f',
            cmap='RdBu_r', center=0,
            vmin=-limit, vmax=limit,
            ax=ax,
            cbar_kws={'label':'Rain-clear (minutes)'}
        )
        ax.set_xlabel('Month in 2024')
        ax.set_ylabel('Pickup zone / platform')
    else:
        fig, ax = plt.subplots(figsize=(10, 5.5))
        bars = grid.T.copy()
        bars.columns = [
            f"{zone} / {platform}" for zone, platform in bars.columns
        ]
        bars.plot(kind='bar', ax=ax, colormap='Set2')
        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_xlabel('Month in 2024')
        ax.set_ylabel('Rain-clear (minutes)')
        ax.tick_params(axis='x', rotation=0)
        ax.legend(
            title="Pickup zone / platform",
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
            fontsize=8,
            frameon=False,
        )
    style(fig, ax)
    show_chart(fig)

def cross_zone_period_bars(case, month, platform):
    zones = list(dict.fromkeys(case['pickup_zone']))
    n_zones = len(zones)
    cols = min(2, n_zones)
    rows = (n_zones + cols - 1) // cols

    fig, axes = plt.subplots(
        rows, cols, figsize=(8.5*cols, 5.2*rows), squeeze=False
    )
    ax_list = axes.ravel()
    y_min = min(case['median_diff'].min(), case['p90_diff'].min(), 0.0)
    y_max = max(case['median_diff'].max(), case['p90_diff'].max(), 0.0)
    pad = max(0.3, 0.1* (y_max - y_min))

    for idx, zone in enumerate(zones):
        ax = ax_list[idx]
        sub = case.loc[case['pickup_zone'] == zone].sort_values('period')
        plot_df = sub.set_index('period')[['median_diff','p90_diff']].rename(
            columns={'median_diff':'Median', 'p90_diff':'90th percentile'}
        )
        plot_df.plot(kind='bar', ax=ax, colormap='Set2', legend=(idx==0))
        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_ylim(y_min-pad, y_max+pad)
        ax.set_title(zone)
        ax.set_xlabel('Request period')
        ax.set_ylabel('Rain-clear (minutes)')
        ax.tick_params(axis='x', rotation=0)
    for idx in range(n_zones, len(ax_list)):
        ax_list[idx].set_visible(False)

    fig.suptitle(
        f'Daytime Period Differences Across Zones ({platform} | Month {month})',
        y=1.02
    )
    style(fig, axes)
    show_chart(fig)

def cross_zone_period_heatmap(case, metric, metric_label, month, platform, all_period):
    pivot_df = case.pivot(
        index='pickup_zone', columns='period', values=metric
    )
    fig, ax = plt.subplots(figsize=(6,4))
    limit = all_period[metric].abs().max()

    sns.heatmap(
        pivot_df, annot=True, fmt='.2f',
        cmap='RdBu_r', center=0,
        vmin=-limit, vmax=limit,
        ax=ax,
        cbar_kws={'label':f"{metric_label}: Rain-clear (min)"})
    ax.set_title(f"{metric_label} Heatmap ({platform} | Month {month})")
    ax.set_xlabel('Request period')
    ax.set_ylabel('Pickup zone')
    style(fig, ax)
    show_chart(fig)    


results = load_results()
period_results = load_period_results()

months_all = sorted(results['month'].unique())
platforms_all = sorted(results['platform'].unique())
zones_all = sorted(results['pickup_zone'].unique())

st.title('Weather and Wait Times for Uber and Lyft')

with st.sidebar:
    analysis_view = st.radio('Analysis',['Overall comparison','Daytime periods'])
    if analysis_view == 'Overall comparison':
        view = st.radio('Visualization', ['Bar chart', 'Heatmap', 'Box plot'])
    else:
        view = None

if analysis_view == 'Daytime periods':
    st.subheader('Compare daytime period across pickup zone')
    c1, c2 = st.columns(2)
    with c1:
        sel_month = st.selectbox('Month', ['All months'] + months_all, key='period-month')
    with c2:
        sel_platform = st.selectbox(
            'Platform', platforms_all, key='period-platform'
        )
    with st.sidebar:
        st.divider()
        sel_zones = multiselect('Pickup zone', zones_all, 'period')
    if not sel_zones:
        st.info('Select at least one pickup zone form the sidebar.')
        st.stop()
    case = period_results.loc[
        (period_results['platform']==sel_platform) &
        (period_results['pickup_zone'].isin(sel_zones))  
    ].copy()

    if sel_month == 'All months':
        case = summarize_all_months(case)
        st.caption(
            'Each value is the median of the available monthly '
            'rain-clear differences.'
        )
        if (case['months_used'] < len(months_all)).any():
            st.caption(
                'Some periods have fewer months of data. '
                'See months_used in the table below.'
            )
    else:
        case = case.loc[case['month']==sel_month].copy()

    if case.empty:
        st.info('No daytime period comparison is available for this selection.')
        st.stop()
    period_view = st.radio(
        'Period chart type',
        ['Faceted bar chart (Median & P90)','Heatmap'],
        horizontal=True
    )

    if period_view == 'Heatmap':
        metric_map = {
            'Median':'median_diff',
            '90th percentile':'p90_diff'
        }
        chosen_label = st.radio(
            'Heatmap metric', list(metric_map.keys()), horizontal=True
        )
        cross_zone_period_heatmap(
            case, metric_map[chosen_label], chosen_label,
            sel_month, sel_platform, period_results
        )
    else:
        cross_zone_period_bars(case, sel_month, sel_platform)

    with st.expander('Period metrics & trip counts'):
        st.dataframe(case.sort_values(['pickup_zone','period']).round(2))
    st.stop()

if view == 'Box plot':
    st.subheader('Select a single combination')
    col_m, col_p, col_z = st.columns(3)
    with col_m:
        sel_month = st.selectbox('Month', months_all, key='single-month')
    with col_p:
        sel_platform = st.selectbox(
            'Platform', platforms_all, key='single-platform'
        )
    with col_z:
        sel_zone = st.selectbox('Pickup zone', zones_all, key='single-zone')
    matched = select_results(
        results, [sel_month], [sel_platform], [sel_zone]
    )
    if matched.empty:
        st.warning('No summary data found for this combination.')
        st.stop()

    row = matched.iloc[0]
    m1, m2 = st.columns(2)
    m1.metric('Median difference', f"{row['median_diff']:+.2f} min")
    m2.metric('90th-percentile difference', f"{row['p90_diff']:+.2f} min")
    st.caption(
        f"Rain - clear. Clear: {int(row['trip_count_clear']):,} trips; "
        f"rain: {int(row['trip_count_rain']):,} trips.")
    box_chart(load_trips(), sel_month, sel_platform, sel_zone)
    with st.expander('All metrics'):
        st.dataframe(matched.round(2).T)
else:
    with st.sidebar:
        st.divider()
        selected_months = multiselect('Month', months_all, 'overall')
        selected_platforms = multiselect('Platform', platforms_all, 'overall')
        selected_zones = multiselect('Pickup zone', zones_all, 'overall')
    selected = select_results(
        results, selected_months, selected_platforms, selected_zones)
    if selected.empty:
        st.info('Select at least one month, platform, and pickup zone')
        st.stop()
    metric_columns = {
        'Median':'median_diff',
        '90th percentile':'p90_diff',
        'Mean':'mean_diff',
        'Standard deviation':'std_diff'}
    choice = st.radio('Chart metric', list(metric_columns))
    metric = metric_columns[choice]

    if len(selected) == 1:
        row = selected.iloc[0]
        m1, m2 = st.columns(2)
        m1.metric('Median difference', f"{row['median_diff']:+.2f} min")
        m2.metric('90th-percentile difference', f"{row['p90_diff']:+.2f} min")
        st.caption(
            f"Rain-clear. Clear: {row['trip_count_clear']:,} trips; "
            f"rain: {row['trip_count_rain']:,} trips.")

    comparison_chart(selected, view, metric, results)
    with st.expander('All metrics'):
        details = selected.round(2).T if len(selected) == 1 else selected.round(2)
        st.dataframe(details)
