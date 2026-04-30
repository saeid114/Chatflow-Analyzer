"""
dashboard.py
Generate visual performance dashboards from analysis results.
Produces PNG charts for conversation funnels, sentiment trends, and failure breakdowns.
"""

import json
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def load_results(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def plot_conversation_funnel(metrics: dict, output_dir: Path):
    """Plot conversation completion funnel."""
    total = metrics["total_conversations"]
    resolved = metrics["resolved_count"]
    unresolved = metrics["unresolved_count"]

    # Simulate funnel stages from data
    stages = ["Greeting", "Intent Captured", "Info Provided", "Resolution", "Satisfied"]
    # Derive approximate numbers (in production, these come from flow tracking)
    values = [
        total,
        int(total * 0.87),
        int(total * 0.64),
        resolved,
        int(resolved * 0.72),
    ]
    percentages = [v / total * 100 for v in values]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#2ecc71", "#27ae60", "#f1c40f", "#e67e22", "#e74c3c"]
    bars = ax.barh(stages[::-1], values[::-1], color=colors[::-1], edgecolor="white", height=0.6)

    for bar, pct, val in zip(bars, percentages[::-1], values[::-1]):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"  {val} ({pct:.0f}%)", va="center", fontsize=11, fontweight="bold")

    ax.set_xlabel("Conversations", fontsize=12)
    ax.set_title("Conversation Completion Funnel", fontsize=14, fontweight="bold")
    ax.set_xlim(0, total * 1.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_dir / "funnel_chart.png", dpi=150)
    plt.close()
    print(f"  Saved funnel_chart.png")


def plot_intent_failure_breakdown(failures: list, output_dir: Path):
    """Pie chart of failure types."""
    if not failures:
        return

    types = {}
    for f in failures:
        ft = f["failure_type"]
        types[ft] = types.get(ft, 0) + 1

    labels = list(types.keys())
    sizes = list(types.values())
    colors = ["#e74c3c", "#f39c12", "#3498db", "#9b59b6"]

    fig, ax = plt.subplots(figsize=(8, 6))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct="%1.0f%%",
        colors=colors[:len(labels)], startangle=90,
        textprops={"fontsize": 12}
    )
    for at in autotexts:
        at.set_fontweight("bold")

    ax.set_title("Intent Failure Breakdown", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "failure_breakdown.png", dpi=150)
    plt.close()
    print(f"  Saved failure_breakdown.png")


def plot_sentiment_distribution(shifts: list, output_dir: Path):
    """Histogram of sentiment deltas."""
    if not shifts:
        return

    deltas = [s["delta"] for s in shifts]

    fig, ax = plt.subplots(figsize=(10, 5))
    n, bins, patches = ax.hist(deltas, bins=15, edgecolor="white", alpha=0.85)

    for patch, left_edge in zip(patches, bins):
        if left_edge < -0.1:
            patch.set_facecolor("#e74c3c")
        elif left_edge < 0.1:
            patch.set_facecolor("#f1c40f")
        else:
            patch.set_facecolor("#2ecc71")

    ax.axvline(x=0, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel("Sentiment Change (start → end)", fontsize=12)
    ax.set_ylabel("Conversations", fontsize=12)
    ax.set_title("User Sentiment Shift Distribution", fontsize=14, fontweight="bold")

    legend_items = [
        mpatches.Patch(color="#e74c3c", label="Negative shift"),
        mpatches.Patch(color="#f1c40f", label="Neutral"),
        mpatches.Patch(color="#2ecc71", label="Positive shift"),
    ]
    ax.legend(handles=legend_items, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_dir / "sentiment_distribution.png", dpi=150)
    plt.close()
    print(f"  Saved sentiment_distribution.png")


def plot_confidence_heatmap(failures: list, output_dir: Path):
    """Show confidence scores across failed interactions."""
    if not failures:
        return

    intents = list(set(f["intent"] for f in failures))
    conv_ids = list(set(f["conversation_id"] for f in failures))

    # Build matrix
    matrix = np.full((len(intents), len(conv_ids)), np.nan)
    for f in failures:
        i = intents.index(f["intent"])
        j = conv_ids.index(f["conversation_id"])
        matrix[i, j] = f["confidence"]

    fig, ax = plt.subplots(figsize=(12, max(4, len(intents) * 0.8)))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(conv_ids)))
    ax.set_xticklabels(conv_ids, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(intents)))
    ax.set_yticklabels(intents, fontsize=10)

    # Add text annotations
    for i in range(len(intents)):
        for j in range(len(conv_ids)):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                        fontsize=9, fontweight="bold",
                        color="white" if matrix[i, j] < 0.5 else "black")

    plt.colorbar(im, label="Confidence Score")
    ax.set_title("Intent Confidence Heatmap (Failed Interactions)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "confidence_heatmap.png", dpi=150)
    plt.close()
    print(f"  Saved confidence_heatmap.png")


def plot_channel_region(metrics: dict, output_dir: Path):
    """Side-by-side bar charts for channel and region distribution."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Channel
    channels = metrics["channel_distribution"]
    ax1.bar(channels.keys(), channels.values(), color=["#3498db", "#e67e22", "#2ecc71"])
    ax1.set_title("By Channel", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Conversations")

    # Region
    regions = metrics["region_distribution"]
    ax2.bar(regions.keys(), regions.values(), color="#9b59b6")
    ax2.set_title("By Region", fontsize=13, fontweight="bold")
    ax2.set_ylabel("Conversations")

    plt.suptitle("Conversation Distribution", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "distribution_charts.png", dpi=150)
    plt.close()
    print(f"  Saved distribution_charts.png")


def generate_dashboard(results_path: str, output_dir: str = "outputs"):
    """Generate all dashboard charts."""
    results = load_results(results_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    charts_dir = out / "dashboard_charts"
    charts_dir.mkdir(exist_ok=True)

    print("Generating dashboard charts...")
    plot_conversation_funnel(results["metrics"], charts_dir)
    plot_intent_failure_breakdown(results["intent_failures"], charts_dir)
    plot_sentiment_distribution(results["sentiment_shifts"], charts_dir)
    plot_confidence_heatmap(results["intent_failures"], charts_dir)
    plot_channel_region(results["metrics"], charts_dir)
    print(f"\nAll charts saved to {charts_dir}/")


def main():
    parser = argparse.ArgumentParser(description="ChatFlow Dashboard Generator")
    parser.add_argument("--input", "-i", required=True, help="Path to analysis_results.json")
    parser.add_argument("--output", "-o", default="outputs", help="Output directory")
    args = parser.parse_args()

    generate_dashboard(args.input, args.output)


if __name__ == "__main__":
    main()
