#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the two composite diagnostic figures used in the manuscript.

By default the script reads the completed analysis stored in
extended_analysis/validated_outputs and writes the figures to
manuscript_figures. A different completed run can be supplied with --input.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "extended_analysis" / "validated_outputs"
DEFAULT_OUTPUT = ROOT / "manuscript_figures"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9.5,
    "axes.unicode_minus": True,
})


def save(fig: plt.Figure, output: Path, stem: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / f"{stem}.png", dpi=600, bbox_inches="tight")
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def read_table(folder: Path, name: str) -> pd.DataFrame:
    path = folder / "tables" / name
    if not path.exists():
        raise FileNotFoundError(f"Missing analysis table: {path}")
    return pd.read_csv(path)


def build_extended_validation(folder: Path, output: Path) -> None:
    windows = read_table(folder, "training_window_performance_summary.csv")
    horizons = read_table(folder, "extended_horizon_metrics.csv")

    model_order = [
        "ENSEMBLE",
        "ETS_AIC_ARCHIVED",
        "LGBM",
        "LSTM",
        "TFT",
    ]
    group_order = ["18_to_30_targets", "54_to_60_targets", "84_targets"]
    group_labels = ["18–30", "54–60", "84"]

    fig, axes = plt.subplots(2, 2, figsize=(10.3, 7.3), constrained_layout=True)

    for ax, country, label in zip(axes[0], ["SPAIN", "FRANCE"], ["(a)", "(b)"]):
        subset = windows[(windows.country == country) & windows.model.isin(model_order)].copy()
        for model in model_order:
            group = subset[subset.model == model].set_index("window_group").reindex(group_order)
            if group["mean_RMSE_MW"].notna().any():
                name = group["model_label"].dropna().iloc[0]
                ax.plot(group_labels, group["mean_RMSE_MW"] / 1000.0, marker="o", label=name)
        ax.set_title(f"{label} {country.title()}: training-window performance")
        ax.set_xlabel("Supervised targets available at origin")
        ax.set_ylabel("Mean 24-month RMSE [GW]")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7.5, ncol=2)

    for ax, country, label in zip(axes[1], ["SPAIN", "FRANCE"], ["(c)", "(d)"]):
        subset = horizons[(horizons.country == country) & (horizons.model == "ENSEMBLE")].copy()
        subset["origin"] = pd.to_datetime(subset["origin"])
        for origin, group in subset.groupby("origin"):
            group = group.sort_values("evaluation_horizon_months")
            ax.plot(
                group["evaluation_horizon_months"],
                group["RMSE_MW"] / 1000.0,
                marker="o",
                linewidth=1.1,
                label=origin.strftime("%Y-%m"),
                alpha=0.85,
            )

        balanced = subset[subset.evaluation_horizon_months.isin([24, 36])]
        counts = balanced.groupby("evaluation_horizon_months")["origin"].nunique()
        valid_h = counts[counts == 4].index
        balanced_mean = (
            balanced[balanced.evaluation_horizon_months.isin(valid_h)]
            .groupby("evaluation_horizon_months", as_index=False)["RMSE_MW"]
            .mean()
            .sort_values("evaluation_horizon_months")
        )
        ax.plot(
            balanced_mean["evaluation_horizon_months"],
            balanced_mean["RMSE_MW"] / 1000.0,
            color="black",
            marker="D",
            linewidth=2.0,
            label="Balanced mean (4 origins)",
        )
        ax.set_title(f"{label} {country.title()}: extended recursive horizons")
        ax.set_xlabel("Evaluation horizon [months]")
        ax.set_ylabel("Ensemble RMSE [GW]")
        ax.set_xticks([24, 30, 36, 42, 48, 54])
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7.2, ncol=2)

    save(fig, output, "extended_validation")


def build_uncertainty_diagnostics(folder: Path, output: Path) -> None:
    scales = read_table(folder, "terminal_residual_scale_sensitivity.csv")
    inflation = read_table(folder, "future_residual_inflation_sensitivity.csv")
    structure = read_table(folder, "conditional_interval_and_structural_range.csv")

    fig, axes = plt.subplots(2, 2, figsize=(10.3, 7.3), constrained_layout=True)

    for ax, country, label in zip(axes[0], ["SPAIN", "FRANCE"], ["(a)", "(b)"]):
        group = scales[scales.country == country].sort_values("mean_width_MW")
        x = group["mean_width_MW"] / 1000.0
        y = group["coverage_pct"]
        ax.plot(x, y, marker="o")
        for _, row in group.iterrows():
            ax.annotate(
                f"{row['absolute_scale']:.3g}",
                (row["mean_width_MW"] / 1000.0, row["coverage_pct"]),
                xytext=(4, 3),
                textcoords="offset points",
                fontsize=7.2,
            )
        selected = group[group.is_selected.astype(bool)]
        ax.scatter(
            selected["mean_width_MW"] / 1000.0,
            selected["coverage_pct"],
            s=55,
            facecolors="none",
            edgecolors="black",
            linewidths=1.2,
            label="Selected scale",
            zorder=3,
        )
        ax.axhline(90.0, linestyle="--", linewidth=1.0, color="gray", label="Nominal 90%")
        ax.set_title(f"{label} {country.title()}: terminal residual-scale sensitivity")
        ax.set_xlabel("Mean P5–P95 width [GW]")
        ax.set_ylabel("Coverage [%]")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7.5)

    variants = ["FROZEN_AFTER_24", "LINEAR_TO_1_25", "LINEAR_TO_1_50"]
    variant_labels = ["Frozen at h=24", "Linear to 1.25×", "Linear to 1.50×"]
    subset = inflation[(inflation.year == 2045) & inflation.variant.isin(variants)].copy()
    x = np.arange(len(variants))
    width = 0.36
    for offset, country in [(-width / 2, "SPAIN"), (width / 2, "FRANCE")]:
        values = (
            subset[subset.country == country]
            .set_index("variant")
            .reindex(variants)["width_MW"]
            .to_numpy() / 1000.0
        )
        axes[1, 0].bar(x + offset, values, width, label=country.title())
    axes[1, 0].set_title("(c) Sensitivity to residual dispersion after month 24")
    axes[1, 0].set_ylabel("2045 conditional P5–P95 width [GW]")
    axes[1, 0].set_xticks(x, variant_labels, rotation=15, ha="right")
    axes[1, 0].grid(True, axis="y", alpha=0.25)
    axes[1, 0].legend(fontsize=7.5)

    subset = structure[structure.year == 2045].set_index("country").reindex(["SPAIN", "FRANCE"])
    x = np.arange(2)
    axes[1, 1].bar(
        x - width / 2,
        subset["turnover_conditional_width_MW"] / 1000.0,
        width,
        label="Conditional statistical width",
    )
    axes[1, 1].bar(
        x + width / 2,
        subset["scenario_P50_range_MW"] / 1000.0,
        width,
        label="Range across scenario P50 values",
    )
    axes[1, 1].set_title("(d) Statistical interval and structural scenario spread")
    axes[1, 1].set_ylabel("2045 spread [GW]")
    axes[1, 1].set_xticks(x, ["Spain", "France"])
    axes[1, 1].grid(True, axis="y", alpha=0.25)
    axes[1, 1].legend(fontsize=7.5)

    save(fig, output, "uncertainty_diagnostics")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="Completed extended-analysis folder containing tables/.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Destination for PNG and PDF files.")
    args = parser.parse_args()

    build_extended_validation(args.input.resolve(), args.output.resolve())
    build_uncertainty_diagnostics(args.input.resolve(), args.output.resolve())
    print(f"Figures written to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
