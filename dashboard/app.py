"""Interactive portfolio dashboard for fraud detection and alert review."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


CORAL = "#F08FA0"
CORAL_DARK = "#D86F82"
TEAL = "#0E6268"
DARK = "#15262B"
MUTED = "#60767B"
SLATE = "#728489"
LIGHT = "#F7FAF9"
DATA = Path(__file__).resolve().parent / "data"
MODEL_NAMES = {
    "logistic_regression": "Logistic regression",
    "hist_gradient_boosting": "Histogram gradient boosting",
}
POLICY_NAMES = {
    "random_capacity": "Random capacity",
    "global_skill": "Global skill",
    "risk_band_specialist": "Risk-band specialist",
}


st.set_page_config(
    page_title="Financial Fraud Detection & Alert Review",
    page_icon="◉",
    layout="wide",
)
st.markdown(
    f"""
    <style>
    .stApp {{background: {LIGHT}; color: {DARK};}}
    .block-container {{max-width: 1480px; padding-top: 1.45rem; padding-bottom: 4rem;}}
    .portfolio-eyebrow {{color: {TEAL}; font-size: .78rem; font-weight: 750; letter-spacing: .16em; text-transform: uppercase; margin-bottom: .45rem;}}
    h1 {{font-size: clamp(2.45rem, 4vw, 3.65rem) !important; line-height: 1.06 !important; letter-spacing: -.035em; margin: 0 0 .35rem !important;}}
    div[data-testid="stCaptionContainer"] {{margin: 0 0 .65rem;}}
    div[data-testid="stCaptionContainer"] p {{color: {MUTED} !important; font-size: 1.02rem;}}
    .scope-row {{display: flex; flex-wrap: wrap; gap: .5rem; margin: .25rem 0 1rem;}}
    .scope-chip {{background: #EAF4F2; border: 1px solid #CFE2DE; border-radius: 999px; color: {TEAL}; font-size: .76rem; font-weight: 650; padding: .35rem .7rem;}}
    div[data-testid="stTabs"] {{margin-top: .05rem;}}
    div[data-baseweb="tab-list"] {{gap: 1.25rem; border-bottom: 1px solid #D8E4E1;}}
    button[data-baseweb="tab"] {{padding: .7rem 0 .62rem;}}
    div[data-testid="stMetric"] {{background: #FFFFFF; border: 1px solid #D9E5E3; border-radius: 15px; padding: .95rem 1.05rem; min-height: 112px; box-shadow: 0 8px 24px rgba(14,98,104,.055);}}
    div[data-testid="stMetricLabel"] p {{color: {MUTED} !important; font-size: .9rem;}}
    div[data-testid="stMetricValue"] {{color: {DARK} !important;}}
    div[data-testid="stMetricDelta"] {{font-weight: 650;}}
    div[data-testid="stPlotlyChart"] {{background: #FFFFFF; border: 1px solid #E0EAE8; border-radius: 16px; padding: .25rem; box-shadow: 0 8px 24px rgba(14,98,104,.035);}}
    .section-intro {{color: {MUTED}; font-size: .96rem; margin: -.2rem 0 1rem; max-width: 900px;}}
    .insight-card {{background: linear-gradient(135deg,#EAF4F2 0%,#FFF3F5 100%); border: 1px solid #CFE2DE; border-left: 5px solid {TEAL}; border-radius: 14px; color: {DARK}; margin: .75rem 0 1.1rem; padding: .9rem 1.05rem;}}
    .insight-card strong {{color: {TEAL};}}
    .decision-card {{background: #FFFFFF; border: 1px solid #D9E5E3; border-radius: 16px; min-height: 100%; padding: 1rem 1.15rem;}}
    .decision-card h4 {{color: {TEAL}; margin: 0 0 .6rem;}}
    .decision-card p {{color: {MUTED}; font-size: .9rem; line-height: 1.55;}}
    .decision-card ul {{color: {DARK}; font-size: .9rem; line-height: 1.6; padding-left: 1.2rem;}}
    .small-note {{color: {MUTED}; font-size: .82rem; line-height: 1.5;}}
    div[data-testid="stDataFrame"] {{border: 1px solid #D9E5E3; border-radius: 14px; overflow: hidden;}}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data() -> tuple[pd.DataFrame, ...]:
    return (
        pd.read_csv(DATA / "monthly_summary.csv"),
        pd.read_csv(DATA / "capacity_summary.csv"),
        pd.read_csv(DATA / "model_summary.csv"),
        pd.read_csv(DATA / "analyst_summary.csv"),
        pd.read_csv(DATA / "review_strategy_summary.csv"),
    )


monthly, capacity, model_summary, analysts, review_strategies = load_data()


def style_figure(figure: go.Figure, height: int = 500) -> go.Figure:
    """Apply the dashboard's visual system to a Plotly figure."""
    figure.update_layout(
        template="plotly_white",
        height=height,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font={"color": DARK, "size": 13},
        title={"font": {"color": DARK, "size": 20}, "x": 0.025, "xanchor": "left"},
        margin={"l": 55, "r": 30, "t": 72, "b": 52},
        legend={"title": None, "orientation": "h", "y": 1.08, "x": 1, "xanchor": "right"},
        hoverlabel={"bgcolor": "#FFFFFF", "font": {"color": DARK}},
        separators=".,",
    )
    figure.update_xaxes(gridcolor="#E8EFED", linecolor="#C9D7D4", zeroline=False)
    figure.update_yaxes(gridcolor="#E8EFED", linecolor="#C9D7D4", zeroline=False)
    return figure


def executive_impact_figure(capacity_data: pd.DataFrame, total_fraud: int) -> go.Figure:
    frame = capacity_data.loc[
        capacity_data["model"].eq("hist_gradient_boosting")
    ].sort_values("review_share").copy()
    frame["Fraud captured"] = frame["fraud_captured"].astype(int)
    frame["Fraud outside queue"] = total_fraud - frame["Fraud captured"]
    frame["Capacity"] = frame.apply(
        lambda row: f"{row.review_share:.0%}<br>{int(row.review_capacity):,} reviewed",
        axis=1,
    )
    frame["Recovery"] = frame["recall_at_capacity"].map(lambda value: f"{value:.0%} captured")

    figure = go.Figure()
    figure.add_bar(
        x=frame["Capacity"],
        y=frame["Fraud captured"],
        name="Fraud captured",
        marker_color=CORAL,
        text=frame["Recovery"],
        textposition="inside",
        textfont={"color": DARK, "size": 12},
        customdata=frame[["review_capacity", "precision_at_capacity"]],
        hovertemplate=(
            "<b>%{x}</b><br>%{y:,.0f} fraud cases captured"
            "<br>Queue precision: %{customdata[1]:.1%}<extra></extra>"
        ),
    )
    figure.add_bar(
        x=frame["Capacity"],
        y=frame["Fraud outside queue"],
        name="Fraud outside queue",
        marker_color="#DDE7E5",
        hovertemplate="<b>%{x}</b><br>%{y:,.0f} fraud cases outside queue<extra></extra>",
    )
    figure.update_layout(
        barmode="stack",
        title="Fraud coverage as review capacity expands",
        xaxis_title="Share of final-month applications sent to review",
        yaxis_title="Observed fraud cases",
        bargap=0.28,
    )
    style_figure(figure, height=465)
    figure.update_yaxes(range=[0, total_fraud * 1.08])
    return figure


def capacity_curve(split: str = "test") -> go.Figure:
    frame = capacity.loc[capacity["split"].eq(split)].copy()
    frame["Review capacity (%)"] = frame["review_share"] * 100
    frame["Fraud recovered (%)"] = frame["recall_at_capacity"] * 100
    frame["Model"] = frame["model"].map(MODEL_NAMES)
    figure = px.line(
        frame,
        x="Review capacity (%)",
        y="Fraud recovered (%)",
        color="Model",
        markers=True,
        title="Fraud recovered as review capacity increases",
        color_discrete_map={
            "Logistic regression": SLATE,
            "Histogram gradient boosting": CORAL,
        },
    )
    figure.update_traces(line_width=3, marker_size=9)
    style_figure(figure)
    figure.update_xaxes(tickmode="array", tickvals=[1, 3, 5, 10], ticksuffix="%")
    figure.update_yaxes(ticksuffix="%", range=[0, 70])
    return figure


def strategy_figure() -> go.Figure:
    frame = review_strategies.copy()
    frame["Policy"] = frame["strategy"].map(POLICY_NAMES)
    long = frame.melt(
        id_vars="Policy",
        value_vars=["mean_accuracy", "mean_precision", "mean_recall"],
        var_name="Metric",
        value_name="Result",
    )
    long["Metric"] = long["Metric"].str.replace("mean_", "", regex=False).str.title()
    long["Result (%)"] = long["Result"] * 100
    figure = px.bar(
        long,
        x="Policy",
        y="Result (%)",
        color="Metric",
        barmode="group",
        title="Assignment results across 25 team scenarios",
        color_discrete_map={"Accuracy": TEAL, "Precision": CORAL, "Recall": SLATE},
    )
    style_figure(figure)
    figure.update_yaxes(ticksuffix="%", range=[0, 100])
    return figure


st.markdown('<div class="portfolio-eyebrow">Portfolio case study · Fraud operations</div>', unsafe_allow_html=True)
st.title("Financial Fraud Detection & Alert Review")
st.caption(
    "Bank-account application risk ranking and capacity-aware alert review using synthetic research data."
)
st.markdown(
    """
    <div class="scope-row">
      <span class="scope-chip">Temporal validation</span>
      <span class="scope-chip">Rare-event modelling</span>
      <span class="scope-chip">Capacity planning</span>
      <span class="scope-chip">50 synthetic reviewers</span>
      <span class="scope-chip">25 team scenarios</span>
    </div>
    """,
    unsafe_allow_html=True,
)


overview_tab, model_tab, capacity_tab, operations_tab, quality_tab = st.tabs(
    ["Executive overview", "Model performance", "Review capacity", "Alert operations", "Data quality"]
)


test_models = model_summary.loc[model_summary["split"].eq("test")].set_index("model")
selected_test = test_models.loc["hist_gradient_boosting"]
test_capacity = capacity.loc[capacity["split"].eq("test")]
boosting_three = test_capacity.loc[
    test_capacity["model"].eq("hist_gradient_boosting")
    & test_capacity["review_share"].eq(0.03)
].iloc[0]
logistic_three = test_capacity.loc[
    test_capacity["model"].eq("logistic_regression")
    & test_capacity["review_share"].eq(0.03)
].iloc[0]
final_month = monthly.loc[monthly["month"].eq(7)].iloc[0]
fraud_total = int(final_month["fraud_cases"])


with overview_tab:
    st.markdown(
        '<p class="section-intro">An executive view of model lift, review workload and the final-month operating result.</p>',
        unsafe_allow_html=True,
    )
    first, second, third, fourth = st.columns(4)
    first.metric("Applications supplied", "917,174")
    second.metric("Final-month fraud prevalence", f"{final_month.fraud_rate:.2%}")
    third.metric(
        "Fraud captured at 3% capacity",
        f"{int(boosting_three.fraud_captured):,}",
        delta=f"+{int(boosting_three.fraud_captured - logistic_three.fraud_captured)} vs logistic",
    )
    fourth.metric("Queue precision at 3%", f"{boosting_three.precision_at_capacity:.1%}")

    st.markdown(
        f"""
        <div class="insight-card">
          <strong>Decision point.</strong> Reviewing the highest-risk 3% of final-month applications recovers
          <strong>{boosting_three.recall_at_capacity:.1%}</strong> of observed fraud. The selected model captures
          <strong>{int(boosting_three.fraud_captured - logistic_three.fraud_captured)} more fraud cases</strong>
          than logistic regression at the same workload.
        </div>
        """,
        unsafe_allow_html=True,
    )

    chart_column, decision_column = st.columns([2.15, 1])
    with chart_column:
        st.plotly_chart(executive_impact_figure(test_capacity, fraud_total), width="stretch")
    with decision_column:
        st.markdown(
            f"""
            <div class="decision-card">
              <h4>Final-month operating view</h4>
              <ul>
                <li><strong>{int(boosting_three.review_capacity):,}</strong> applications enter review.</li>
                <li><strong>{int(boosting_three.fraud_captured):,}</strong> of {fraud_total:,} fraud cases are captured.</li>
                <li><strong>{fraud_total - int(boosting_three.fraud_captured):,}</strong> fraud cases remain outside the queue.</li>
                <li><strong>{int(boosting_three.review_capacity - boosting_three.fraud_captured):,}</strong> reviewed applications are legitimate.</li>
              </ul>
              <p>The 3% capacity is a transparent comparison point, not a production recommendation. A final policy requires investigation and fraud-loss costs.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

with model_tab:
    st.subheader("Model ranking performance")
    st.markdown(
        '<p class="section-intro">Models are selected on later validation months and reported once on the untouched final month. Average precision is the primary ranking metric.</p>',
        unsafe_allow_html=True,
    )

    display_models = test_models.reset_index().copy()
    display_models["Model"] = display_models["model"].map(MODEL_NAMES)
    display_models["Average precision"] = display_models["average_precision"]
    display_models["ROC AUC"] = display_models["roc_auc"]
    display_models["Threshold precision"] = display_models["precision"]
    display_models["Threshold recall"] = display_models["recall"]
    display_models = display_models[
        ["Model", "Average precision", "ROC AUC", "Threshold precision", "Threshold recall"]
    ]
    st.dataframe(
        display_models.style.format(
            {
                "Average precision": "{:.3f}",
                "ROC AUC": "{:.3f}",
                "Threshold precision": "{:.1%}",
                "Threshold recall": "{:.1%}",
            }
        ),
        width="stretch",
        hide_index=True,
    )

    left_chart, right_chart = st.columns([1, 1.65])
    with left_chart:
        comparison = display_models.melt(
            id_vars="Model",
            value_vars=["Average precision", "ROC AUC"],
            var_name="Metric",
            value_name="Score",
        )
        figure = px.bar(
            comparison,
            x="Score",
            y="Model",
            color="Metric",
            barmode="group",
            orientation="h",
            title="Final-month ranking metrics",
            color_discrete_map={"Average precision": CORAL, "ROC AUC": TEAL},
        )
        style_figure(figure)
        figure.update_xaxes(range=[0, 1], tickformat=".1f")
        st.plotly_chart(figure, width="stretch")
    with right_chart:
        st.plotly_chart(capacity_curve("test"), width="stretch")

    st.markdown(
        f"""
        <div class="insight-card">
          Histogram gradient boosting increases final-month average precision from
          <strong>{test_models.loc['logistic_regression', 'average_precision']:.3f}</strong> to
          <strong>{selected_test.average_precision:.3f}</strong> and recovers more fraud at every tested review capacity.
        </div>
        """,
        unsafe_allow_html=True,
    )


with capacity_tab:
    st.subheader("Review-capacity planning")
    st.markdown(
        '<p class="section-intro">Select a workload to see how many fraud cases enter the queue, how many remain outside it and how much legitimate-review effort is created.</p>',
        unsafe_allow_html=True,
    )
    capacity_choice = st.select_slider("Applications sent to review", options=[1, 3, 5, 10], value=3)
    row = test_capacity.loc[
        test_capacity["model"].eq("hist_gradient_boosting")
        & (test_capacity["review_share"] * 100).round().eq(capacity_choice)
    ].iloc[0]
    captured = int(row.fraud_captured)
    missed = fraud_total - captured
    legitimate_reviews = int(row.review_capacity) - captured

    first, second, third, fourth = st.columns(4)
    first.metric("Applications reviewed", f"{int(row.review_capacity):,}")
    second.metric("Fraud cases captured", f"{captured:,}")
    third.metric("Fraud cases outside queue", f"{missed:,}", delta_color="inverse")
    fourth.metric("Queue precision", f"{row.precision_at_capacity:.1%}")

    recovery_chart, workload_chart = st.columns(2)
    with recovery_chart:
        figure = go.Figure(
            go.Pie(
                labels=["Fraud captured", "Fraud outside queue"],
                values=[captured, missed],
                hole=0.62,
                marker={"colors": [CORAL, "#DDE7E5"]},
                textinfo="label+percent",
                sort=False,
            )
        )
        figure.update_layout(title="Final-month fraud recovery")
        style_figure(figure, height=470)
        st.plotly_chart(figure, width="stretch")
    with workload_chart:
        workload = pd.DataFrame(
            {
                "Queue outcome": ["Fraud captured", "Legitimate review"],
                "Applications": [captured, legitimate_reviews],
            }
        )
        figure = px.bar(
            workload,
            x="Queue outcome",
            y="Applications",
            color="Queue outcome",
            title="Composition of the review queue",
            color_discrete_map={"Fraud captured": CORAL, "Legitimate review": TEAL},
        )
        style_figure(figure, height=470)
        figure.update_layout(showlegend=False)
        st.plotly_chart(figure, width="stretch")

    st.markdown(
        '<p class="small-note">Increasing capacity recovers more fraud but reduces its concentration inside the queue. No monetary-loss or review-cost field is supplied, so the dashboard does not invent an optimal production threshold.</p>',
        unsafe_allow_html=True,
    )


with operations_tab:
    st.subheader("Alert assignment and reviewer capacity")
    st.markdown(
        '<p class="section-intro">Historical reviewer outcomes are used to compare three capacity-constrained assignment policies on a later alert month. All reviewer profiles are synthetic.</p>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(strategy_figure(), width="stretch")

    comparison = review_strategies.set_index("strategy")
    random_row = comparison.loc["random_capacity"]
    specialist = comparison.loc["risk_band_specialist"]
    first, second, third, fourth = st.columns(4)
    first.metric("Team scenarios", "25")
    second.metric(
        "Mean false positives avoided",
        f"{random_row.mean_false_positive - specialist.mean_false_positive:.0f}",
        delta="vs random allocation",
    )
    third.metric(
        "Precision change",
        f"{(specialist.mean_precision - random_row.mean_precision) * 100:+.2f} pp",
    )
    fourth.metric(
        "Recall change",
        f"{(specialist.mean_recall - random_row.mean_recall) * 100:+.2f} pp",
        delta_color="inverse",
    )

    st.markdown(
        '<div class="insight-card"><strong>Operating trade-off.</strong> Risk-band assignment reduces unnecessary positive decisions, but random allocation retains slightly more fraud. The appropriate policy depends on the relative cost of false positives and missed fraud.</div>',
        unsafe_allow_html=True,
    )

    analyst_figure = px.scatter(
        analysts,
        x="recall",
        y="precision",
        size="positive_rate",
        hover_name="analyst",
        title="Synthetic reviewer precision–recall profiles",
        labels={"recall": "Recall", "precision": "Precision", "positive_rate": "Positive-decision rate"},
        color_discrete_sequence=[TEAL],
    )
    style_figure(analyst_figure, height=520)
    analyst_figure.update_xaxes(tickformat=".0%")
    analyst_figure.update_yaxes(tickformat=".0%")
    st.plotly_chart(analyst_figure, width="stretch")
    st.markdown(
        '<p class="small-note">The reviewer chart represents simulated behaviour supplied with the research dataset. It must not be interpreted as employee evaluation.</p>',
        unsafe_allow_html=True,
    )


with quality_tab:
    st.subheader("Data quality and interpretation boundaries")
    st.markdown(
        '<p class="section-intro">Source limitations remain visible so that model results are not separated from the conditions under which they were produced.</p>',
        unsafe_allow_html=True,
    )

    figure = go.Figure()
    figure.add_bar(
        x=monthly["month"],
        y=monthly["applications"],
        name="Applications",
        marker_color=TEAL,
        opacity=0.82,
    )
    figure.add_scatter(
        x=monthly["month"],
        y=monthly["fraud_rate"] * 100,
        name="Fraud rate",
        mode="lines+markers",
        line={"color": CORAL, "width": 3},
        marker={"size": 8},
        yaxis="y2",
    )
    figure.add_vrect(
        x0=3.65,
        x1=4.35,
        fillcolor=CORAL,
        opacity=0.12,
        line_width=0,
        annotation_text="Incomplete source month",
        annotation_position="top left",
    )
    figure.update_layout(
        title="Application volume and observed fraud rate",
        xaxis_title="Source month",
        yaxis={"title": "Applications"},
        yaxis2={"title": "Fraud rate (%)", "overlaying": "y", "side": "right", "showgrid": False},
    )
    style_figure(figure, height=520)
    st.plotly_chart(figure, width="stretch")

    source_column, boundary_column = st.columns(2)
    with source_column:
        st.markdown(
            """
            <div class="decision-card">
              <h4>Verified source scope</h4>
              <ul>
                <li>917,174 supplied application rows</li>
                <li>one structurally incomplete final row</li>
                <li>30,622 selected alerts</li>
                <li>50 synthetic reviewer profiles</li>
                <li>25 final-month team scenarios</li>
              </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with boundary_column:
        st.markdown(
            """
            <div class="decision-card">
              <h4>Interpretation boundaries</h4>
              <ul>
                <li>month 4 is excluded from primary temporal comparisons</li>
                <li>applications and reviewer decisions are synthetic</li>
                <li>protected-field exclusion does not prove fairness</li>
                <li>no investigation cost or monetary loss is supplied</li>
                <li>results do not demonstrate production readiness</li>
              </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
