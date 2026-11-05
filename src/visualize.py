"""
Scientific Visualization Module
Topic: Vector Databases in Retrieval-Augmented Generation (RAG)
Authors: Aibar Alauov, Balkybek Baurzhan
Adheres to the 3 Core Rules of Scientific Visualization:
1. Explicit Axes & Physical Units
2. Uncertainty Quantification (95% Bootstrap CI Bands & Whiskers)
3. Accessibility & Printability (Colorblind-safe palettes, dual encoding with line styles/markers)
4. Dual Export: Vector (.pdf) and Raster (>= 300 DPI .png)
"""

import sys
from pathlib import Path
from typing import Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def set_scientific_style():
    """Configures matplotlib with publication-ready scientific plotting parameters."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def plot_fig1_latency_vs_load(
    summary_df: pd.DataFrame,
    output_dir: str = "reports/figures",
    dpi: int = 300,
) -> Tuple[Path, Path]:
    """
    Generates Figure 1: Query Latency (ms) vs. Concurrency Load.
    Includes:
    - Explicit axes with physical units
    - Shaded 95% Bootstrap Confidence Intervals (uncertainty quantification)
    - Colorblind-safe palette (sns.color_palette("colorblind"))
    - Dual series encoding: distinct markers (o, s) and line styles (--, -)
    - SLA Guardrail reference line (200 ms)
    - Logarithmic Y-axis scaling for order-of-magnitude span
    """
    set_scientific_style()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(8.5, 5.5), constrained_layout=True)
    cb_palette = sns.color_palette("colorblind")

    # Method configurations for accessibility & monochrome printability
    configs = {
        "Baseline_Flat": {
            "label": "Baseline: Exact Flat Index (L2 / Cosine)",
            "color": cb_palette[0],       # Deep Blue
            "linestyle": "--",            # Dashed
            "marker": "o",                # Circle
            "markersize": 7,
        },
        "Proposed_HNSW_SQ8": {
            "label": "Proposed: HNSW + SQ8 Index",
            "color": cb_palette[2],       # Green / Teal
            "linestyle": "-",             # Solid
            "marker": "s",                # Square
            "markersize": 7,
        },
    }

    for method, cfg in configs.items():
        subset = summary_df[summary_df["method"] == method].sort_values("concurrency_level")
        if subset.empty:
            continue

        x = subset["concurrency_level"].to_numpy()
        y_median = subset["median_latency_ms"].to_numpy()
        ci_low = subset["ci95_lower_ms"].to_numpy()
        ci_high = subset["ci95_upper_ms"].to_numpy()

        # Plot median line with markers
        ax1.plot(
            x,
            y_median,
            label=cfg["label"],
            color=cfg["color"],
            linestyle=cfg["linestyle"],
            marker=cfg["marker"],
            markersize=cfg["markersize"],
            linewidth=2.2,
            alpha=0.95,
        )

        # Plot shaded 95% Confidence Interval band (Core Rule 2: Uncertainty Quantification)
        ax1.fill_between(
            x,
            ci_low,
            ci_high,
            color=cfg["color"],
            alpha=0.22,
            label=f"95% Bootstrap CI ({method.split('_')[0]})",
        )

    # Core Rule 1: Explicit Axes & Physical Units + Logarithmic Scale
    ax1.set_xlabel("Client Concurrency Load (Concurrent Clients / Workers)", fontweight="bold")
    ax1.set_ylabel("Query Latency (ms) [Log Scale]", fontweight="bold")
    ax1.set_yscale("log")
    ax1.set_ylim(bottom=8, top=2500)

    # SLA Guardrail line
    sla_line = ax1.axhline(
        y=200.0,
        color=cb_palette[3],  # Crimson / Red
        linestyle=":",
        linewidth=2.0,
        label="SLA Guardrail Threshold (200 ms)",
    )

    # Annotate SLA Violation region
    ax1.text(
        x=1.1,
        y=220.0,
        s="SLA Violation Zone (>200 ms)",
        color=cb_palette[3],
        fontsize=9,
        fontweight="semibold",
    )

    ax1.set_xticks([1, 5, 10, 20, 50, 100])
    ax1.set_xticklabels(["1", "5", "10", "20", "50", "100"])
    ax1.set_title(
        "Figure 1: Query Latency vs. Concurrency Load in RAG Pipeline\n"
        "(Evaluation of Exact Flat Search vs. HNSW+SQ8 Graph Index with 95% CI)",
        fontweight="bold",
        pad=12,
    )
    ax1.legend(loc="upper left", frameon=True, framealpha=0.92, facecolor="white")

    pdf_path = out_dir / "fig1_latency_vs_load.pdf"
    png_path = out_dir / "fig1_latency_vs_load.png"

    # Core Rule 4: Dual Export (Vector .pdf and >=300 DPI .png)
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.savefig(png_path, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return pdf_path, png_path


def plot_fig2_distribution_boxplot(
    df_benchmark: pd.DataFrame,
    output_dir: str = "reports/figures",
    dpi: int = 300,
) -> Tuple[Path, Path]:
    """
    Generates Figure 2: Statistical Dispersion Boxplot of Latencies across Load Levels.
    Includes:
    - Boxplots showing Median, IQR, Whiskers (1.5 IQR), and Outliers
    - Grouped comparison: Baseline vs Proposed
    - SLA threshold line at 200 ms
    - Colorblind-safe palette
    - Dual export: .pdf and .png (300 DPI)
    """
    set_scientific_style()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9.5, 6.0), constrained_layout=True)
    cb_palette = sns.color_palette("colorblind")
    palette_dict = {
        "Baseline_Flat": cb_palette[0],
        "Proposed_HNSW_SQ8": cb_palette[2],
    }

    # Prepare plotting dataframe with human-readable labels
    df_plot = df_benchmark.copy()
    df_plot["Method_Label"] = df_plot["method"].map({
        "Baseline_Flat": "Baseline (Flat)",
        "Proposed_HNSW_SQ8": "Proposed (HNSW-SQ8)"
    })

    method_palette = {
        "Baseline (Flat)": cb_palette[0],
        "Proposed (HNSW-SQ8)": cb_palette[2],
    }

    sns.boxplot(
        data=df_plot,
        x="concurrency_level",
        y="latency_ms",
        hue="Method_Label",
        palette=method_palette,
        ax=ax,
        width=0.65,
        fliersize=3.0,
        linewidth=1.2,
        showmeans=True,
        meanprops={"marker": "^", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 6},
    )

    # Core Rule 1: Explicit Axes & Units
    ax.set_xlabel("Client Concurrency Load (Concurrent Clients)", fontweight="bold")
    ax.set_ylabel("Query Latency (ms) [Log Scale]", fontweight="bold")
    ax.set_yscale("log")
    ax.set_ylim(bottom=10, top=3000)

    # Guardrail SLA line
    ax.axhline(
        y=200.0,
        color=cb_palette[3],
        linestyle=":",
        linewidth=2.0,
        label="SLA Guardrail (200 ms)",
    )

    ax.set_title(
        "Figure 2: Empirical Latency Distribution & Dispersion across Concurrency Tiers\n"
        "(Boxplots show Median, IQR, 1.5x IQR Whiskers, Mean [^], and Outliers)",
        fontweight="bold",
        pad=12,
    )
    ax.legend(title="Vector Index Architecture", loc="upper left", frameon=True, framealpha=0.92)

    pdf_path = out_dir / "fig2_distribution_boxplot.pdf"
    png_path = out_dir / "fig2_distribution_boxplot.png"

    # Core Rule 4: Dual Export (.pdf and >= 300 DPI .png)
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.savefig(png_path, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return pdf_path, png_path


def run_visualizations(
    processed_csv_path: str = "data/processed/metrics_summary.csv",
    raw_data_path: str = "data/raw/test_run.jsonl",
    output_dir: str = "reports/figures",
    dpi: int = 300,
):
    """Executes full visualization pipeline producing all mandatory figures."""
    from src.process_data import filter_warmup_phase, load_raw_benchmark_data

    print(f"[1/3] Loading data for visualization...")
    summary_df = pd.read_csv(processed_csv_path)
    df_raw = load_raw_benchmark_data(raw_data_path)
    _, df_benchmark = filter_warmup_phase(df_raw)

    print(f"[2/3] Rendering Figure 1: Latency vs. Concurrency with 95% CI bands...")
    fig1_pdf, fig1_png = plot_fig1_latency_vs_load(summary_df, output_dir=output_dir, dpi=dpi)
    print(f"      Saved: {fig1_pdf} and {fig1_png} ({dpi} DPI)")

    print(f"[3/3] Rendering Figure 2: Latency Distribution Boxplot...")
    fig2_pdf, fig2_png = plot_fig2_distribution_boxplot(df_benchmark, output_dir=output_dir, dpi=dpi)
    print(f"      Saved: {fig2_pdf} and {fig2_png} ({dpi} DPI)")
    print("Scientific visualization pipeline completed successfully!")


if __name__ == "__main__":
    run_visualizations()
