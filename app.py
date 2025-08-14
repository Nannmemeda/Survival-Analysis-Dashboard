# Import required packages
import numpy as np
import pandas as pd
import streamlit as st
from lifelines import KaplanMeierFitter
import plotly.graph_objects as go
from scipy.stats import chi2_contingency
from lifelines.statistics import multivariate_logrank_test
from streamlit_option_menu import option_menu

#######################
# Page configuration
st.set_page_config(
    page_title="United States 2012–2016 Breast Cancer Survival Analysis Dashboard",
    page_icon="💉",
    layout="wide",
    initial_sidebar_state="expanded")

# Load data
df = pd.read_csv('cleaned_SEER.csv')

#######################
# CSS styling
st.markdown("""
<style>
            
.filter-title{
    font-weight:700;
    font-size:1.05rem;
    margin-bottom:14px;
    margin-top:-4px;
}

div[data-testid="stSelectbox"] label p {
    font-weight: 700;
    font-size: 1.05rem;
    margin-bottom: 3px;
}
            
div[data-testid="stCheckbox"] {
    margin-top: -13px;
    margin-bottom: -5px
}

hr.my-divider {
    border: none;
    border-top: 1px dashed #bbb;
    margin-top: -7px;
    margin-bottom: 2px;
}
            
[data-testid="stMetric"] {
    background-color: #F5F7FA ;
    text-align: center;
    padding: 15px 0;
    border-radius: 7px;
}
            
[data-testid="stMetricLabel"] {
    display: flex;
    justify-content: center;
}
            
.topbar {{
    position: sticky; 
    top: 0; 
    z-index: 999; 
    background: white; 
    padding: 6px 0 8px; }}


</style>
""", unsafe_allow_html=True)

#######################
# Sidebar
with st.sidebar:
    st.title('Compare By & Filters')
    # Line color options
    color_map = {
        "All": None,
        "Race": 'Race_Group',
        "Age Group": "Age_Group",
        "Grade": "Grade_Type",
        "Stage": "Stage",
    }

    color_by = st.selectbox(
        "Compare by:",
        list(color_map.keys()),
        index = 0
    )
    color_col = color_map[color_by]

    # Race filters 
    ALL_RACE_KEY = "race_all"
    NHW_KEY = "race_nhw"
    HW_KEY = "race_hw"
    BLK_KEY = "race_blk"
    API_KEY = "race_api"
    INDIVIDUAL_RACE_KEYS = [NHW_KEY, HW_KEY, BLK_KEY, API_KEY]

    for k in [ALL_RACE_KEY, *INDIVIDUAL_RACE_KEYS]:
        st.session_state.setdefault(k, True)
    
    def _on_change_all_race():
        set_to = bool(st.session_state[ALL_RACE_KEY])
        for k in INDIVIDUAL_RACE_KEYS:
            st.session_state[k] = set_to
        st.session_state[ALL_RACE_KEY] = all(st.session_state[k] for k in INDIVIDUAL_RACE_KEYS)

    def _on_change_individual_race():
        st.session_state[ALL_RACE_KEY] = all(st.session_state[k] for k in INDIVIDUAL_RACE_KEYS)

    filter_box = st.container(key="race_box", border=True)
    with filter_box:
        st.markdown('<div class="filter-title">Race / Ethnicity</div>', unsafe_allow_html=True)
        st.checkbox("All Races / Ethnicities", key=ALL_RACE_KEY, on_change=_on_change_all_race)
        st.markdown('<hr class="my-divider">', unsafe_allow_html=True)
        st.checkbox("Non-Hispanic White", key=NHW_KEY, on_change=_on_change_individual_race)
        st.checkbox("Hispanic White", key=HW_KEY, on_change=_on_change_individual_race)
        st.checkbox("Black", key=BLK_KEY, on_change=_on_change_individual_race)
        st.checkbox("Asian or Pacific Islander", key=API_KEY, on_change=_on_change_individual_race)

    st.markdown('</div>', unsafe_allow_html=True)
    
    sel_races = []
    if st.session_state[NHW_KEY]: sel_races.append("Non-Hispanic White")
    if st.session_state[HW_KEY]:  sel_races.append("Hispanic White")
    if st.session_state[BLK_KEY]: sel_races.append("Black")
    if st.session_state[API_KEY]: sel_races.append("Asian or Pacific Islander")

    # Age Group filters 
    ALL_AGE_KEY = "age_all"
    L40_KEY = "age_l40"
    A40_KEY = "age_a40"
    A50_KEY = "age_a50"
    A60_KEY = "age_a60"
    A70_KEY = "age_a70"
    A80_KEY = "age_a80"
    INDIVIDUAL_AGE_KEYS = [L40_KEY, A40_KEY, A50_KEY, A60_KEY, A70_KEY, A80_KEY]

    for k in [ALL_AGE_KEY, *INDIVIDUAL_AGE_KEYS]:
        st.session_state.setdefault(k, True)

    def _on_change_all_age():
        set_to = bool(st.session_state[ALL_AGE_KEY])
        for k in INDIVIDUAL_AGE_KEYS:
            st.session_state[k] = set_to
        st.session_state[ALL_AGE_KEY] = all(st.session_state[k] for k in INDIVIDUAL_AGE_KEYS)

    def _on_change_individual_age():
        st.session_state[ALL_AGE_KEY] = all(st.session_state[k] for k in INDIVIDUAL_AGE_KEYS)

    with filter_box:
        st.markdown('<div class="filter-title">Age</div>', unsafe_allow_html=True)
        st.checkbox("All Ages", key=ALL_AGE_KEY, on_change=_on_change_all_age)
        st.markdown('<hr class="my-divider">', unsafe_allow_html=True)
        st.checkbox("Age < 40", key=L40_KEY, on_change=_on_change_individual_age)
        st.checkbox("Age 40-49", key=A40_KEY, on_change=_on_change_individual_age)
        st.checkbox("Age 50-59", key=A50_KEY, on_change=_on_change_individual_age)
        st.checkbox("Age 60-69", key=A60_KEY, on_change=_on_change_individual_age)
        st.checkbox("Age 70-79", key=A70_KEY, on_change=_on_change_individual_age)
        st.checkbox("Age 80+", key=A80_KEY, on_change=_on_change_individual_age)

    sel_ages = []
    if st.session_state[L40_KEY]: sel_ages.append("40-")
    if st.session_state[A40_KEY]: sel_ages.append("40-49")
    if st.session_state[A50_KEY]: sel_ages.append("50-59")
    if st.session_state[A60_KEY]: sel_ages.append("60-69")
    if st.session_state[A70_KEY]: sel_ages.append("70-79")
    if st.session_state[A80_KEY]: sel_ages.append("80+")

    # Stage filters
    ALL_STAGE_KEY = "stage_all"
    LOC_KEY = "stage_local"
    RLN_KEY = "stage_rln"
    DIS_KEY = "stage_dis"
    DELN_KEY = "stage_deln"
    DE_KEY = "stage_de"
    NOS_KEY = "stage_nos"
    UNS_KEY = "stage_uns"
    INDIVIDUAL_STAGE_KEYS = [LOC_KEY, RLN_KEY, DIS_KEY, DELN_KEY, DE_KEY, NOS_KEY, UNS_KEY]

    for k in [ALL_STAGE_KEY, *INDIVIDUAL_STAGE_KEYS]:
        st.session_state.setdefault(k, True)
    
    def _on_change_all_stage():
        set_to = bool(st.session_state[ALL_STAGE_KEY])
        for k in INDIVIDUAL_STAGE_KEYS:
            st.session_state[k] = set_to
        st.session_state[ALL_STAGE_KEY] = all(st.session_state[k] for k in INDIVIDUAL_STAGE_KEYS)

    def _on_change_individual_stage():
        st.session_state[ALL_STAGE_KEY] = all(st.session_state[k] for k in INDIVIDUAL_STAGE_KEYS)

    with filter_box:
        st.markdown('<div class="filter-title">Stage</div>', unsafe_allow_html=True)
        st.checkbox("All Stages", key=ALL_STAGE_KEY, on_change=_on_change_all_stage)
        st.markdown('<hr class="my-divider">', unsafe_allow_html=True)
        st.checkbox("Localized only", key=LOC_KEY, on_change=_on_change_individual_stage)
        st.checkbox("Regional lymph nodes involved only", key=RLN_KEY, on_change=_on_change_individual_stage)
        st.checkbox("Distant site(s)/node(s) involved", key=DIS_KEY, on_change=_on_change_individual_stage)
        st.checkbox("Regional by both direct extension and lymph node involvement", key=DELN_KEY, on_change=_on_change_individual_stage)
        st.checkbox("Regional by direct extension only", key=DE_KEY, on_change=_on_change_individual_stage)
        st.checkbox("Regional, NOS", key=NOS_KEY, on_change=_on_change_individual_stage)
        st.checkbox("Unknown/unstaged/unspecified/DCO", key=UNS_KEY, on_change=_on_change_individual_stage)
    
    sel_stages = []
    if st.session_state[LOC_KEY]: sel_stages.append("Localized only")
    if st.session_state[RLN_KEY]: sel_stages.append("Regional lymph nodes involved only")
    if st.session_state[DIS_KEY]: sel_stages.append("Distant site(s)/node(s) involved")
    if st.session_state[DELN_KEY]: sel_stages.append("Regional by both direct extension and lymph node involvement")
    if st.session_state[DE_KEY]: sel_stages.append("Regional by direct extension only")
    if st.session_state[NOS_KEY]: sel_stages.append("Regional, NOS")
    if st.session_state[UNS_KEY]: sel_stages.append("Unknown/unstaged/unspecified/DCO")
    
    # Grade filters
    ALL_GRADE_KEY = "grade_all"
    I_KEY = "grade_i"
    II_KEY = "grade_ii"
    III_KEY = "grade_iii"
    IV_KEY = "grade_iv"
    UNK_KEY = "grade_unk"
    INDIVIDUAL_GRADE_KEYS = [I_KEY, II_KEY, III_KEY, IV_KEY, UNK_KEY]

    for k in [ALL_GRADE_KEY, *INDIVIDUAL_GRADE_KEYS]:
        st.session_state.setdefault(k, True)
    
    def _on_change_all_grade():
        set_to = bool(st.session_state[ALL_GRADE_KEY])
        for k in INDIVIDUAL_GRADE_KEYS:
            st.session_state[k] = set_to
        st.session_state[ALL_GRADE_KEY] = all(st.session_state[k] for k in INDIVIDUAL_GRADE_KEYS)

    def _on_change_individual_grade():
        st.session_state[ALL_GRADE_KEY] = all(st.session_state[k] for k in INDIVIDUAL_GRADE_KEYS)

    with filter_box:
        st.markdown('<div class="filter-title">Grade</div>', unsafe_allow_html=True)
        st.checkbox("All Grades", key=ALL_GRADE_KEY, on_change=_on_change_all_grade)
        st.markdown('<hr class="my-divider">', unsafe_allow_html=True)
        st.checkbox("Grade I: Well differentiated", key=I_KEY, on_change=_on_change_individual_grade)
        st.checkbox("Grade II: Moderately differentiated", key=II_KEY, on_change=_on_change_individual_grade)
        st.checkbox("Grade III: Poorly differentiated", key=III_KEY, on_change=_on_change_individual_grade)
        st.checkbox("Grade IV: Undifferentiated; anaplastic", key=IV_KEY, on_change=_on_change_individual_grade)
        st.checkbox("Unknown", key=UNK_KEY, on_change=_on_change_individual_grade)
    
    sel_grades = []
    if st.session_state[I_KEY]: sel_grades.append("Grade I: Well differentiated")
    if st.session_state[II_KEY]: sel_grades.append("Grade II: Moderately differentiated")
    if st.session_state[III_KEY]: sel_grades.append("Grade III: Poorly differentiated")
    if st.session_state[IV_KEY]: sel_grades.append("Grade IV: Undifferentiated; anaplastic")
    if st.session_state[UNK_KEY]: sel_grades.append("Unknown")

    # Histology select box
    with filter_box:
        count_hist = df['Histology_Name'].dropna().value_counts()
        hist_by_freq = count_hist.index.tolist()
        hist_cat_opts = ['All histology'] + hist_by_freq
        sel_hist = st.selectbox('Histology',hist_cat_opts,index=0,key="hist_name_select")

# Apply filters 
df_filtered = df[
    df['Race_Group'].isin(sel_races)
    & df['Age_Group'].isin(sel_ages)
    & df['Grade_Type'].isin(sel_grades)
    & df['Stage'].isin(sel_stages)
]

if sel_hist != 'All histology':
    df_filtered = df_filtered[df_filtered["Histology_Name"] == sel_hist]

#######################
# Kaplan-Meier Computation
def compute_km(df,group_col):
    out = {}
    if group_col is None:
        groups = [(None,df)]
    else:
        groups = list(df.groupby(group_col,dropna=False))
    
    for gname,gdf in groups:
        gdf = gdf.dropna(subset=['Survival_months','Status'])
        
        # Create the Kaplan Meier Survival Model
        kmf = KaplanMeierFitter()
        kmf.fit(durations = gdf['Survival_months'], event_observed = gdf['Status'])

        # Create the timeline and fit the model based on the timeline
        t_min = 0
        t_max = 66
        timeline = np.arange(t_min,t_max,1)
        surv = kmf.survival_function_at_times(timeline).rename("survival").reset_index()
        surv.rename(columns={"index": "Survival_months"}, inplace=True)

        # Calculate the confidence interval
        ci_full = kmf.confidence_interval_.copy()
        ci_full = ci_full.rename_axis("Survival_months").reset_index()
        lower_col = [c for c in ci_full.columns if "lower" in c][0]
        upper_col = [c for c in ci_full.columns if "upper" in c][0]
        lower_interp = np.interp(timeline, ci_full["Survival_months"], ci_full[lower_col])
        upper_interp = np.interp(timeline, ci_full["Survival_months"], ci_full[upper_col])
        surv["lower_ci"] = lower_interp
        surv["upper_ci"] = upper_interp

        # Convert to percentage
        surv["Relative_Survival_%"] = 100.0 * surv["survival"].astype(float)
        surv["lower_ci_%"] = 100.0 * surv["lower_ci"].astype(float)
        surv["upper_ci_%"] = 100.0 * surv["upper_ci"].astype(float)
        out[gname if gname is not None else "All"] = surv[[
            "Survival_months", "Relative_Survival_%", "lower_ci_%", "upper_ci_%"
        ]]

    return out

# Calculate 1 year and 3 year relative survival rate
def km_survival_pct(df_in, months):
    dd = df_in.dropna(subset=["Survival_months", "Status"])
    if dd.empty:
        return float("nan")
    kmf = KaplanMeierFitter()
    kmf.fit(durations=dd["Survival_months"], event_observed=dd["Status"])
    # lifelines carries the last step if 'months' exceeds last event time
    return float(kmf.survival_function_at_times(months).values[0] * 100.0)

# Chisq test on survival months
def log_rank_test_surv_month(df, group_col):
    if group_col is None:
        return None
    
    dd= df[[group_col,'Survival_months','Status']]
    group = dd[group_col].dropna().unique().tolist()
    if len(group) < 2:
        return None
    
    res = multivariate_logrank_test(
        dd['Survival_months'].astype(float),
        dd[group_col].astype(str),
        dd['Status'].astype(int)
    )

    return {
        "chi2": float(res.test_statistic),
        "p_value": float(res.p_value),
        "degrees_freedom": int(res.degrees_of_freedom),
    }

# Survival trend computation
def compute_surv_trend(df, group_col):
    df2 = df.dropna(subset=['YearDx', 'Survival_months', 'Status'])
    df2 = df2[~((df2['Status'] == 0) & (df2['Survival_months'] < 12))]
    df2 = df2[~(df2['YearDx'] == 2016)]

    if group_col is None:
        groups = [(None, df2)]
    else:
        groups = list(df2.groupby(group_col, dropna=False))

    out = {}
    H = 12
    timeline = np.arange(0, H + 1, 1)

    for gname, gdf in groups:
        rows = []
        for year, cohort in gdf.groupby('YearDx'):
            cohort = cohort.dropna(subset=['Survival_months', 'Status'])
            n = len(cohort)

            kmf = KaplanMeierFitter()
            kmf.fit(
                durations=cohort['Survival_months'].astype(float),
                event_observed=cohort['Status'].astype(int),
                timeline=timeline
            )
            s12 = float(kmf.survival_function_.loc[H].values[0]) * 100.0
            ci = kmf.confidence_interval_.loc[H]
            lower = float(ci.iloc[0]) * 100.0
            upper = float(ci.iloc[1]) * 100.0

            rows.append({
                'YearDx': int(year),
                'Survival_12m_%': s12,
                'lower_ci_%': lower,
                'upper_ci_%': upper,
                'n': int(n)
            })

        df_out = pd.DataFrame(rows).sort_values('YearDx').reset_index(drop=True)
        out[gname if gname is not None else 'All'] = df_out

    return out

# Trend Metrics Calculations
def trend_metrics(df):
    trend_all = compute_surv_trend(df_filtered, None).get("All", pd.DataFrame())
    t = trend_all.sort_values("YearDx").dropna(subset=["YearDx", "Survival_12m_%"]).reset_index(drop=True)

    # Min/Max Year
    min_year = int(t["YearDx"].min())
    max_year = int(t["YearDx"].max())
    s_min = t.loc[t["YearDx"] == min_year, "Survival_12m_%"]
    s_max = t.loc[t["YearDx"] == max_year, "Survival_12m_%"]
    s_min_val = float(s_min.iloc[0])
    s_max_val = float(s_max.iloc[0])

    # Latest survival
    latest_surv = s_max_val

    # Average absolute gap
    if t["Survival_12m_%"].size >= 2:
        diffs = np.abs(np.diff(t["Survival_12m_%"]))
        avg_abs_diff = float(np.nanmean(diffs))
    else:
        avg_abs_diff = 0

    # Gap
    gap = np.nanmax(t["Survival_12m_%"]) - np.nanmin(t["Survival_12m_%"])

    # Avg survival rate
    avg_surv = float(t["Survival_12m_%"].mean())

    return min_year, max_year, latest_surv, avg_abs_diff, gap, avg_surv

# Chisq test on 1-year survival
def chisq_1yr_surv(df,group_col):
    if group_col is None:
        return None
    
    dd = df[[group_col,'Survival_months','Status']]
    dd = dd[~((dd['Status'] == 0) & (dd['Survival_months'] < 12))]
    if dd[group_col].nunique() < 2:
        return None
    
    dd['alive_12m'] = (dd['Survival_months'] >= 12).astype(int)

    contingency = (
        pd.crosstab(dd[group_col], dd['alive_12m'])
          .reindex(columns=[1, 0], fill_value=0)
    )

    if contingency.shape[0] < 2:
        return None
    
    chi2, p, dof, expected = chi2_contingency(contingency, correction=False)
    
    return {
        "chi2": float(chi2),
        "p_value": float(p),
        "degrees_freedom": int(dof),
        "expected": pd.DataFrame(expected, index=contingency.index, columns=contingency.columns)
    }

#######################
# Dashboard

st.markdown(
    "<h1 style='text-align: center;'>United States 2012–2016 Breast Cancer Survival Analysis Dashboard</h1>",
    unsafe_allow_html=True
)

MODE_TREND = "Recent Trend Analysis"
MODE_KM    = "Time Since Diagnosis Analysis"

ACCENT = "#C1121F"
with st.container():
    st.markdown('<div class="topbar">', unsafe_allow_html=True)
    mode = option_menu(
        None,
        [MODE_KM, MODE_TREND],              # KM first = default
        icons=["activity", "graph-up"],     # use Bootstrap icons (tried & visible)
        orientation="horizontal",
        default_index=0,
        styles={                             # <-- single braces here
            "container": {"padding": "0", "background-color": "transparent"},
            "icon": {"color": ACCENT, "font-size": "18px"},
            "nav-link": {
                "font-weight": "700",
                "color": ACCENT,            # unselected text = red
                "background-color": "transparent",
                "border": f"1px solid {ACCENT}",
                "border-radius": "9999px",
                "padding": "10px 16px",
                "margin-right": "8px",
            },
            "nav-link-selected": {
                "background-color": ACCENT, # selected pill = red
                "color": "white",           # selected text = white
            },
        }
    )
    st.markdown('</div>', unsafe_allow_html=True)


# if "analysis_mode" not in st.session_state:
#     st.session_state.analysis_mode = MODE_KM
#     st.rerun()

# with st.container():
#     c1, c2 = st.columns([1,1])
#     with c1:
#         if st.button(
#         MODE_KM,
#         key="btn_mode_km",
#         type="primary" if st.session_state.analysis_mode == MODE_KM else "secondary",
#         use_container_width=True
#         ):
#             st.session_state.analysis_mode = MODE_KM
#     with c2:
#         if st.button(
#             MODE_TREND,
#             key="btn_mode_trend",
#             type="primary" if st.session_state.analysis_mode == MODE_TREND else "secondary",
#             use_container_width=True
#         ):
#             st.session_state.analysis_mode = MODE_TREND

# mode = st.session_state.analysis_mode

group_col = color_col

legend_title_map = {
            'Race_Group': 'Race',
            'Age_Group': 'Age Group',
            'Grade_Type': 'Grade',
            'Stage': 'Stage'
        }
legend_title_text = legend_title_map.get(group_col, "") if group_col else ""

if mode == MODE_KM:
    if df_filtered.empty:
        st.warning("No data after filters. Adjust selections to see results.")
    else:
        # Metrics Bar
        col = st.columns((1.5, 1.5, 1.5, 1.5), gap='medium')

        with col[0]:
            st.metric(label='Total Cases', value=f"{len(df_filtered):,}")

        with col[1]:
            death_event = len(df_filtered[df_filtered['Status'] == 1])
            st.metric(label='Total Death Events', value=f"{death_event:,}")
        
        s12 = km_survival_pct(df_filtered, 12)
        s36 = km_survival_pct(df_filtered, 36)

        with col[2]:
            st.metric(label='1-year Relative Survival Rate', value=f"{s12:.1f} %")

        with col[3]:
            st.metric(label='3-year Relative Survival Rate', value=f"{s36:.1f} %")

        # Line plot
        km_curves = compute_km(df_filtered, group_col)

        fig = go.Figure()
        for gname, gdf in sorted(km_curves.items(), key=lambda x: str(x[0])):
            # Line
            fig.add_trace(
                go.Scatter(
                    x=gdf["Survival_months"],
                    y=gdf["Relative_Survival_%"],
                    mode="lines",
                    name=str(gname),
                    hovertemplate=
                        "<b><span style='font-size:13px'>%{x} months since diagnosis</span></b><br>" +
                        "-------------------------------------------------------<br>" +
                        "Relative Survival (%):                             " + "%{y:.2f}<br>" +
                        "95% Confidence Interval:      " + "%{customdata[0]:.2f} – %{customdata[1]:.2f}"
                        f"<extra>{gname}</extra>",
                    customdata=np.stack([gdf["lower_ci_%"], gdf["upper_ci_%"]], axis=-1)
                    )
                )
            # CI band
            fig.add_trace(
                go.Scatter(
                    x=pd.concat([gdf["Survival_months"], gdf["Survival_months"][::-1]]),
                    y=pd.concat([gdf["upper_ci_%"], gdf["lower_ci_%"][::-1]]),
                    fill="toself",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                    opacity=0.5,
                    name=f"{gname} CI"
                        )
                    )
        fig.update_layout(
            height=550,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(
                dtick=6,
                range=[0, 65],
                zeroline=True,
                zerolinecolor='black',
                title=dict(
                    text="Month(s) since Diagnosis",
                    font=dict(
                        size=18,
                        color='black'
                    )
                ),
                tickfont=dict(
                    size=15,
                    color='black'
                )
                ),
            yaxis=dict(
                range=[0, 100],
                tickmode="array",
                tickvals=[0] + list(range(10, 101, 10)),
                ticktext=[""] + [str(v) for v in range(10, 101, 10)],
                zeroline=True,
                zerolinecolor='black',
                title=dict(
                    text="Relative Survival Rate (%)",
                    font=dict(
                        size=18,
                        color='black'
                    )
                ),
                tickfont=dict(
                    size=15,
                    color='black'
                )
            ),
            legend=dict(
            orientation="h",
            yanchor="top", y=-0.18,
            xanchor="center", x=0.5,
            title=dict(text=legend_title_text,
                    font=dict(color='black')),
            bgcolor="rgba(0,0,0,0)",
            traceorder="normal"
            )
        )
        st.plotly_chart(fig, use_container_width=True)

        # Chisq test
        chisq_res = log_rank_test_surv_month(df_filtered, group_col)
        
        if chisq_res:
            p_val = chisq_res["p_value"]
        
            if p_val < 1e-16:
                p_str = "< 1e-16"
            else:
                p_str = f"{p_val:.2e}" if p_val < 0.001 else f"{p_val:.4f}"

            if chisq_res["p_value"] < 0.05:
                st.success(f"✅ Statistically significant difference in survival months between {color_by.lower()}s (p = {p_str} on degrees of freedom {chisq_res['degrees_freedom']})")
            else:
                st.info(f"ℹ️ No statistically significant difference in survival months between {color_by.lower()}s (p = {p_str} on degrees of freedom {chisq_res['degrees_freedom']})")
else:
    if df_filtered.empty:
        st.warning("No data after filters. Adjust selections to see results.")
    else:
        # Metrics Bar
        col = st.columns((1.5, 1.5, 1.5, 1.5), gap='medium')

        min_year, max_year, latest_surv, avg_abs_diff, gap, avg_surv = trend_metrics(df_filtered)

        with col[0]:
            st.metric(label=f'Latest 1-year Survival Rate (Year {max_year})', value=f"{latest_surv:.2f} %")

        with col[1]:
            st.metric(label='Average 1-year Survival Rate', value=f"{avg_surv:.2f} %")

        with col[2]:
            st.metric(label=f'Average Difference — {min_year} -> {max_year}', value=f"{avg_abs_diff:.2f}")

        with col[3]:
            st.metric(label=f'Best-Worst Gap — {min_year} -> {max_year}', value=f"{gap:.2f}")
        
        # Line plot
        trend_dict = compute_surv_trend(df_filtered, color_col)
        all_years = []
        for _g, _df in trend_dict.items():
            if not _df.empty:
                all_years.extend(_df['YearDx'].tolist())

        fig_trend = go.Figure()

        for gname, gdf in sorted(trend_dict.items(), key=lambda x: str(x[0])):
            if gdf.empty:
                continue
            gdf = gdf.sort_values("YearDx").reset_index(drop=True)

            y = gdf["Survival_12m_%"].astype(float).to_numpy()
            prev = np.r_[np.nan, y[:-1]]
            delta = y - prev

            symbols = np.where(delta > 0, "▲", np.where(delta < 0, "▼", "▬"))
            delta_label = np.where(np.isfinite(delta),
                           [f"{d:+.2f} {s}" for d, s in zip(delta, symbols)],
                           "—")
            
            customdata = np.column_stack((
                gdf["lower_ci_%"].to_numpy(float),
                gdf["upper_ci_%"].to_numpy(float),
                delta_label  # strings → dtype object
            ))

            # Main line
            fig_trend.add_trace(
                go.Scatter(
                    x=gdf["YearDx"],
                    y=gdf["Survival_12m_%"],
                    mode="lines+markers",
                    name=str(gname),
                    customdata=customdata,
                    hovertemplate=(
                        "<b><span style='font-size:13px'> Diagnosis Year %{x}</span></b><br>" +
                        "-----------------------------------------------------------<br>" +
                        "1-year Relative Survival (%):                    " + "%{y:.2f}<br>" +
                        "95% Confidence Interval:           " + "%{customdata[0]:.2f} – %{customdata[1]:.2f}<br>" +
                        "Absolute Change vs Previous Year:     %{customdata[2]}"
                    )
                )
            )

            # CI band
            fig_trend.add_trace(
                go.Scatter(
                    x=pd.concat([gdf["YearDx"], gdf["YearDx"][::-1]]),
                    y=pd.concat([gdf["upper_ci_%"], gdf["lower_ci_%"][::-1]]),
                    fill="toself",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                    opacity=0.35,
                    name=f"{gname} CI"
                )
            )

        # Axes ranges & ticks
        if all_years:
            xmin = min(all_years) - 0.1
            xmax = max(all_years) + 0.1
        else:
            xmin, xmax = 2012, 2016  # sensible fallback

        fig_trend.update_layout(
            height=550,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(
                title=dict(text="Year of Diagnosis", font=dict(size=18, color='black')),
                tickmode="linear",
                dtick=1,
                range=[2011.94, xmax],
                zeroline=False,
                tickfont=dict(size=15, color='black')
            ),
            yaxis=dict(
                title=dict(text="1-year Relative Survival Rate (%)", font=dict(size=18, color='black')),
                range=[60, 100],
                tickmode="array",
                tickvals=[0] + list(range(60, 101, 10)),
                ticktext=[""] + [str(v) for v in range(60, 101, 10)],
                zeroline=False,
                tickfont=dict(size=15, color='black')
            ),
            shapes=[
                dict(
                    type="line",
                    xref="paper", x0=0, x1=1,
                    yref="y", y0=60, y1=60,
                    line=dict(color="black")
                ),
                dict(
                    type="line",
                    xref="x", x0=2011.94, x1=2011.94,
                    yref="paper", y0=0, y1=1,
                    line=dict(color="black")
                )
            ],
            legend=dict(
                orientation="h",
                yanchor="top", y=-0.18,
                xanchor="center", x=0.5,
                title=dict(text=legend_title_text, font=dict(color='black')),
                bgcolor="rgba(0,0,0,0)",
                traceorder="normal"
            )
        )

        st.plotly_chart(fig_trend, use_container_width=True)

        # Chisq test
        chisq_res2 = chisq_1yr_surv(df_filtered, group_col)
        
        if chisq_res2:
            p_val2 = chisq_res2["p_value"]
        
            if p_val2 < 1e-16:
                p_str2 = "< 1e-16"
            else:
                p_str2 = f"{p_val2:.2e}" if p_val2 < 0.001 else f"{p_val2:.4f}"

            if chisq_res2["p_value"] < 0.05:
                st.success(f"✅ Statistically significant difference in 1-year survival status between {color_by.lower()}s (p = {p_str2} on degrees of freedom {chisq_res2['degrees_freedom']})")
            else:
                st.info(f"ℹ️ No statistically significant difference in 1-year survival status between {color_by.lower()}s (p = {p_str2} on degrees of freedom {chisq_res2['degrees_freedom']})")




