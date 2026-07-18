"""Generate repository figures from verified audit and model outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


CORAL = "#F08FA0"
TEAL = "#0E6268"
DARK = "#15262B"
MUTED = "#6E7F83"
BACKGROUND = "#F7FAF9"


def style() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "figure.facecolor": BACKGROUND,
            "axes.facecolor": BACKGROUND,
            "axes.edgecolor": "#CBD7D5",
            "axes.labelcolor": DARK,
            "text.color": DARK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "font.size": 11,
        }
    )


def monthly_figure(audit: dict, output: Path) -> None:
    monthly = pd.DataFrame(audit["base_by_month"])
    fig, left = plt.subplots(figsize=(11, 6))
    right = left.twinx()
    bars = left.bar(monthly["month"], monthly["applications"], color=TEAL, alpha=0.82, label="Applications")
    right.plot(
        monthly["month"],
        monthly["fraud_rate"] * 100,
        color=CORAL,
        marker="o",
        linewidth=2.5,
        label="Fraud rate",
    )
    left.axvspan(3.65, 4.35, color=CORAL, alpha=0.12)
    left.text(
        4.28,
        monthly["applications"].max() * 0.72,
        "P4 — incomplete",
        ha="center",
        va="center",
        rotation=90,
        color=DARK,
        fontsize=9,
    )
    left.bar_label(bars, labels=[f"{value / 1000:.0f}k" for value in monthly["applications"]], padding=3, fontsize=9)
    for month, rate in zip(monthly["month"], monthly["fraud_rate"] * 100):
        right.annotate(
            f"{rate:.2f}%",
            (month, rate),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            color=CORAL,
            fontsize=9,
            fontweight="bold",
        )
    left.set(
        title="Application volume and observed fraud rate by source period",
        xlabel="Source period (dataset index)",
        ylabel="Applications",
    )
    left.set_xticks(monthly["month"], [f"P{month}" for month in monthly["month"]])
    right.set_ylabel("Fraud rate (%)")
    left.grid(axis="x", visible=False)
    right.grid(visible=False)
    left.margins(y=0.14)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def capacity_figure(metrics: dict, output: Path) -> None:
    fig, axis = plt.subplots(figsize=(10, 6))
    for name, colour in [("logistic_regression", MUTED), ("hist_gradient_boosting", CORAL)]:
        rows = pd.DataFrame(metrics["models"][name]["test_capacity"])
        axis.plot(
            rows["review_share"] * 100,
            rows["recall_at_capacity"] * 100,
            marker="o",
            linewidth=2.5,
            color=colour,
            label=name.replace("_", " ").title(),
        )
    axis.set(
        title="Fraud recovered as investigation capacity increases",
        xlabel="Applications reviewed (%)",
        ylabel="Fraud cases recovered (%)",
    )
    axis.set_xticks([1, 3, 5, 10])
    axis.set_ylim(0, 70)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def assignment_figure(review: dict, output: Path) -> None:
    summary = pd.DataFrame(review["strategy_summary"])
    order = ["random_capacity", "global_skill", "risk_band_specialist"]
    summary = summary.set_index("strategy").loc[order].reset_index()
    labels = ["Random capacity", "Global skill", "Risk-band specialist"]
    metrics = ["mean_accuracy", "mean_precision", "mean_recall"]
    colours = [TEAL, CORAL, MUTED]
    x = range(len(summary))
    width = 0.24
    fig, axis = plt.subplots(figsize=(11, 6))
    for offset, metric, colour in zip((-width, 0, width), metrics, colours):
        axis.bar(
            [value + offset for value in x],
            summary[metric] * 100,
            width=width,
            label=metric.removeprefix("mean_").title(),
            color=colour,
        )
    axis.set(
        title="Alert-assignment results across 25 team scenarios",
        xlabel="Assignment policy",
        ylabel="Mean result (%)",
        xticks=list(x),
        xticklabels=labels,
    )
    axis.set_ylim(0, 100)
    axis.legend(frameon=False, ncols=3)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, default=Path("reports/data_audit.json"))
    parser.add_argument("--metrics", type=Path, default=Path("reports/model_metrics.json"))
    parser.add_argument("--review", type=Path, default=Path("reports/review_strategy_metrics.json"))
    parser.add_argument("--output", type=Path, default=Path("images"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    review = json.loads(args.review.read_text(encoding="utf-8"))
    style()
    monthly_figure(audit, args.output / "monthly_data_quality.png")
    capacity_figure(metrics, args.output / "review_capacity_tradeoff.png")
    assignment_figure(review, args.output / "assignment_strategy_comparison.png")


if __name__ == "__main__":
    main()
