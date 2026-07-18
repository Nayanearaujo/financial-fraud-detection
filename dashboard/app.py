"""Interactive portfolio dashboard for fraud detection and alert review."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


CORAL = "#F08FA0"
TEAL = "#0E6268"
DARK = "#15262B"
DATA = Path(__file__).resolve().parent / "data"

st.set_page_config(page_title="Financial Fraud Detection", page_icon="◉", layout="wide")
st.markdown(
    f"""
    <style>
    .stApp {{background: #F7FAF9; color: {DARK};}}
    .block-container {{max-width: 1500px; padding-top: 1.8rem; padding-bottom: 3rem;}}
    h1 {{font-size: clamp(2.5rem, 4vw, 3.75rem) !important; line-height: 1.05 !important; margin: 0 0 .45rem !important;}}
    div[data-testid="stCaptionContainer"] {{margin: 0 0 .8rem;}}
    div[data-testid="stCaptionContainer"] p {{color: #60767B !important; font-size: 1rem;}}
    div[data-testid="stTabs"] {{margin-top: .15rem;}}
    div[data-baseweb="tab-list"] {{gap: 1.35rem;}}
    button[data-baseweb="tab"] {{padding: .75rem 0 .65rem;}}
    div[data-testid="stMetric"] {{background: white; border: 1px solid #D9E5E3; border-radius: 14px; padding: 1rem 1.15rem; min-height: 112px; box-shadow: 0 8px 24px rgba(14, 98, 104, .05);}}
    div[data-testid="stMetricLabel"] p {{color: #60767B !important; font-size: .95rem;}}
    div[data-testid="stMetricValue"] {{color: {DARK} !important;}}
    div[data-testid="stPlotlyChart"] {{background: white; border: 1px solid #E1EAE8; border-radius: 16px; padding: .4rem;}}
    </style>
    """,
    unsafe_allow_html=True,
)

monthly = pd.read_csv(DATA / "monthly_summary.csv")
capacity = pd.read_csv(DATA / "capacity_summary.csv")
analysts = pd.read_csv(DATA / "analyst_summary.csv")
review_strategies = pd.read_csv(DATA / "review_strategy_summary.csv")


def style_figure(figure, height: int = 520):
    """Apply the dashboard's light visual system to every Plotly figure."""
    figure.update_layout(
        template="plotly_white",
        height=height,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font={"color": DARK, "size": 14},
        title={"font": {"color": DARK, "size": 21}, "x": 0.02, "xanchor": "left"},
        margin={"l": 55, "r": 30, "t": 75, "b": 55},
        legend={"title": None},
    )
    figure.update_xaxes(gridcolor="#E8EFED", linecolor="#C9D7D4")
    figure.update_yaxes(gridcolor="#E8EFED", linecolor="#C9D7D4")
    return figure

st.title("Financial Fraud Detection & Alert Review")
st.caption(
    "Bank-account application risk ranking and capacity-aware alert review using synthetic research data."
)

overview, capacity_tab, assignment_tab, analysts_tab, quality_tab = st.tabs(
    ["Overview", "Review capacity", "Alert assignment", "Analyst variation", "Data quality"]
)

with overview:
    left, middle, right, fourth = st.columns(4)
    left.metric("Applications supplied", "917,174")
    middle.metric("Fraud prevalence", "1.20%")
    right.metric("Alerts selected", "30,622")
    fourth.metric("Fraud prevalence among alerts", "12.13%")

    monthly_chart = monthly.copy()
    monthly_chart["fraud_rate_percent"] = monthly_chart["fraud_rate"] * 100
    monthly_chart.loc[monthly_chart["month"].eq(4), "fraud_rate_percent"] = float("nan")
    figure = px.line(
        monthly_chart,
        x="month",
        y="fraud_rate_percent",
        markers=True,
        title="Observed fraud rate by complete source month",
        labels={"month": "Source month", "fraud_rate_percent": "Fraud rate (%)"},
        color_discrete_sequence=[CORAL],
    )
    figure.update_traces(line_width=3, marker_size=9, connectgaps=False)
    figure.add_vrect(
        x0=3.65,
        x1=4.35,
        fillcolor=CORAL,
        opacity=0.12,
        line_width=0,
        annotation_text="Incomplete month excluded",
        annotation_position="top left",
        annotation_font={"color": DARK, "size": 12},
    )
    style_figure(figure)
    y_ticks = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    figure.update_xaxes(tickmode="array", tickvals=list(range(8)), ticktext=[str(value) for value in range(8)])
    figure.update_yaxes(
        range=[0.72, 1.58],
        tickmode="array",
        tickvals=y_ticks,
        ticktext=[f"{value:.1f}" for value in y_ticks],
    )
    st.plotly_chart(figure, width="stretch")
    st.info("Month 4 is excluded from primary comparisons because the supplied CSV terminates during that month.")

with capacity_tab:
    split = st.radio("Evaluation split", ["test", "validation"], horizontal=True)
    selected = capacity.loc[capacity["split"].eq(split)].copy()
    selected["review_percent"] = selected["review_share"] * 100
    selected["recovered_percent"] = selected["recall_at_capacity"] * 100
    selected["precision_percent"] = selected["precision_at_capacity"] * 100

    chart = px.line(
        selected,
        x="review_percent",
        y="recovered_percent",
        color="model",
        markers=True,
        title="Fraud recovered as review capacity increases",
        labels={
            "review_percent": "Applications reviewed (%)",
            "recovered_percent": "Fraud recovered (%)",
            "model": "Model",
        },
        color_discrete_map={"logistic_regression": "#728489", "hist_gradient_boosting": CORAL},
    )
    chart.update_traces(line_width=3, marker_size=9)
    style_figure(chart)
    st.plotly_chart(chart, width="stretch")

    capacity_choice = st.select_slider("Review capacity", options=[1, 3, 5, 10], value=3)
    row = selected.loc[
        selected["model"].eq("hist_gradient_boosting")
        & selected["review_percent"].round().eq(capacity_choice)
    ].iloc[0]
    a, b, c = st.columns(3)
    a.metric("Applications reviewed", f"{int(row.review_capacity):,}")
    b.metric("Fraud cases captured", f"{int(row.fraud_captured):,}")
    c.metric("Queue precision", f"{row.precision_at_capacity:.1%}")

with analysts_tab:
    figure = px.scatter(
        analysts,
        x="recall",
        y="precision",
        size="positive_rate",
        hover_name="analyst",
        title="Synthetic analysts make different review decisions",
        labels={"recall": "Recall", "precision": "Precision", "positive_rate": "Positive decision rate"},
        color_discrete_sequence=[TEAL],
    )
    style_figure(figure)
    st.plotly_chart(figure, width="stretch")
    st.caption("All analyst decisions are synthetic; the chart does not represent employee performance.")

with assignment_tab:
    friendly_names = {
        "random_capacity": "Random capacity",
        "global_skill": "Global skill",
        "risk_band_specialist": "Risk-band specialist",
    }
    comparison = review_strategies.copy()
    comparison["policy"] = comparison["strategy"].map(friendly_names)
    long = comparison.melt(
        id_vars="policy",
        value_vars=["mean_accuracy", "mean_precision", "mean_recall"],
        var_name="metric",
        value_name="value",
    )
    long["metric"] = long["metric"].str.replace("mean_", "", regex=False).str.title()
    long["value_percent"] = long["value"] * 100
    chart = px.bar(
        long,
        x="policy",
        y="value_percent",
        color="metric",
        barmode="group",
        title="Capacity-aware assignment across 25 team scenarios",
        labels={"policy": "Assignment policy", "value_percent": "Mean result (%)", "metric": "Metric"},
        color_discrete_map={"Accuracy": TEAL, "Precision": CORAL, "Recall": "#728489"},
    )
    style_figure(chart)
    st.plotly_chart(chart, width="stretch")

    random_row = comparison.set_index("strategy").loc["random_capacity"]
    specialist = comparison.set_index("strategy").loc["risk_band_specialist"]
    first, second, third = st.columns(3)
    first.metric("Mean false positives avoided", f"{random_row.mean_false_positive - specialist.mean_false_positive:.0f}")
    second.metric("Precision change", f"{(specialist.mean_precision - random_row.mean_precision) * 100:+.1f} pp")
    third.metric("Recall change", f"{(specialist.mean_recall - random_row.mean_recall) * 100:+.1f} pp")
    st.caption(
        "Risk-band assignment reduces unnecessary positive decisions but retains slightly fewer fraud cases. "
        "The appropriate policy depends on the relative cost of each error."
    )

with quality_tab:
    chart = px.bar(
        monthly,
        x="month",
        y="applications",
        color=monthly["month"].eq(4).map({True: "Incomplete", False: "Complete"}),
        title="Applications supplied by source month",
        labels={"month": "Source month", "applications": "Applications", "color": "Source status"},
        color_discrete_map={"Complete": TEAL, "Incomplete": CORAL},
    )
    style_figure(chart)
    st.plotly_chart(chart, width="stretch")
    st.markdown(
        "The archive contains one truncated final row and a materially incomplete month 4. "
        "No records or missing values were fabricated to repair the source."
    )
