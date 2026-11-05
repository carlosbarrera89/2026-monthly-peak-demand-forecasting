#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Additional diagnostics for the monthly peak-demand forecasting package.

The default run uses the archived predictions and simulations. It does not
change the main experiment, retrain the neural models, or replace any values
reported by run_forecasting_final.py.

Use --long-horizon-retraining only in an environment that contains every
package listed in requirements_retraining.txt. That optional mode fits the
same three learners at later historical origins and evaluates paths longer
than 24 months where observations are available.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
import traceback
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
COUNTRIES = ("spain", "france")
HORIZON_BANDS = ((1, 6), (7, 12), (13, 18), (19, 24))
FUTURE_INFLATION_ENDPOINTS = {
    "FROZEN_AFTER_24": 1.00,
    "LINEAR_TO_1_25": 1.25,
    "LINEAR_TO_1_50": 1.50,
}
LONG_HORIZON_ORIGINS = (
    "2021-06-01",
    "2021-12-01",
    "2022-06-01",
    "2022-12-01",
)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.unicode_minus": True,
})


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    temp.replace(path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False, encoding="utf-8", date_format="%Y-%m-%d")
    temp.replace(path)


def point_metrics(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    valid = np.isfinite(actual) & np.isfinite(prediction)
    if not valid.any():
        return {"n": 0, "RMSE_MW": np.nan, "MAE_MW": np.nan,
                "WAPE_pct": np.nan, "bias_MW": np.nan}
    actual = actual[valid]
    prediction = prediction[valid]
    error = prediction - actual
    denominator = float(np.abs(actual).sum())
    return {
        "n": int(len(actual)),
        "RMSE_MW": float(np.sqrt(np.mean(error ** 2))),
        "MAE_MW": float(np.mean(np.abs(error))),
        "WAPE_pct": float(100.0 * np.abs(error).sum() / denominator) if denominator else np.nan,
        "bias_MW": float(np.mean(error)),
    }


def interval_score(actual: np.ndarray, lower: np.ndarray, upper: np.ndarray,
                   alpha: float = 0.10) -> np.ndarray:
    actual = np.asarray(actual, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    width = upper - lower
    return (width
            + (2.0 / alpha) * (lower - actual) * (actual < lower)
            + (2.0 / alpha) * (actual - upper) * (actual > upper))


def sample_crps(samples: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """CRPS for equally weighted samples without forming an S by S matrix."""
    samples = np.asarray(samples, dtype=float)
    actual = np.asarray(actual, dtype=float)
    if samples.ndim != 2 or samples.shape[1] != len(actual):
        raise ValueError("CRPS samples and observations are not aligned.")
    ordered = np.sort(samples, axis=0)
    n = ordered.shape[0]
    ranks = (2.0 * np.arange(1, n + 1) - n - 1.0)[:, None]
    first = np.mean(np.abs(samples - actual[None, :]), axis=0)
    second = np.sum(ranks * ordered, axis=0) / (n * n)
    return first - second


def probabilistic_metrics(samples: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    q5, q50, q95 = np.quantile(samples, [0.05, 0.50, 0.95], axis=0)
    actual = np.asarray(actual, dtype=float)
    covered = (actual >= q5) & (actual <= q95)
    score = interval_score(actual, q5, q95)
    crps = sample_crps(samples, actual)
    return {
        "n": int(len(actual)),
        "coverage_pct": float(100.0 * covered.mean()),
        "mean_width_MW": float(np.mean(q95 - q5)),
        "interval_score_MW": float(np.mean(score)),
        "CRPS_MW": float(np.mean(crps)),
        "below_P5": int(np.sum(actual < q5)),
        "above_P95": int(np.sum(actual > q95)),
        "P50_RMSE_MW": float(np.sqrt(np.mean((q50 - actual) ** 2))),
    }


def annual_sample_quantiles(samples: np.ndarray, dates: Iterable[pd.Timestamp]) -> pd.DataFrame:
    dates = pd.DatetimeIndex(pd.to_datetime(list(dates)))
    if samples.shape[1] != len(dates):
        raise ValueError("Monthly samples and dates are not aligned.")
    rows = []
    for year in sorted(set(dates.year)):
        mask = dates.year == year
        annual_max = samples[:, mask].max(axis=1)
        q5, q50, q95 = np.quantile(annual_max, [0.05, 0.50, 0.95])
        rows.append({
            "year": int(year),
            "P5_MW": float(q5),
            "P50_MW": float(q50),
            "P95_MW": float(q95),
            "width_MW": float(q95 - q5),
            "mean_MW": float(np.mean(annual_max)),
            "sd_MW": float(np.std(annual_max, ddof=1)),
        })
    return pd.DataFrame(rows)


def horizon_band(horizon: int) -> str:
    for start, end in HORIZON_BANDS:
        if start <= horizon <= end:
            return f"h{start:02d}_h{end:02d}"
    return "outside_range"


def save_figure(fig: plt.Figure, folder: Path, stem: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(folder / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def model_label(name: str) -> str:
    return {
        "LGBM": "LightGBM",
        "ETS_damped": "Damped ETS",
        "SeasonalNaive": "Seasonal naive",
        "SARIMA_predeclared": "Prespecified SARIMA",
        "ETS_AIC_ARCHIVED": "AIC-selected ETS",
        "SARIMA_AIC_ARCHIVED": "AIC-selected SARIMA",
        "ENSEMBLE": "Inverse-RMSE ensemble",
    }.get(name, name)


def training_window_analysis(tables: Path, figures: Path) -> None:
    detail_rows = []
    for country in COUNTRIES:
        calendar = pd.read_csv(ROOT / "frozen_statistical" / country / "validation_calendar.csv",
                               parse_dates=["origin"])
        paths = pd.read_csv(ROOT / "frozen_statistical" / country / "statistical_benchmark_paths.csv",
                            parse_dates=["origin", "ds"])
        paths["origin"] = pd.to_datetime(paths["origin"])
        paths = paths.merge(calendar[["phase", "origin", "train_n_months", "supervised_n"]],
                            on=["phase", "origin"], how="left", validate="many_to_one")
        for keys, group in paths.groupby(["phase", "origin", "train_n_months", "supervised_n", "model"], dropna=False):
            phase, origin, train_n, supervised_n, model = keys
            row = {
                "country": country.upper(),
                "phase": phase,
                "origin": origin,
                "train_n_months": int(train_n),
                "supervised_n": int(supervised_n),
                "model": model,
                "model_label": model_label(model),
            }
            row.update(point_metrics(group["y_true"], group["pred_MW"]))
            detail_rows.append(row)
    detail = pd.DataFrame(detail_rows).sort_values(["country", "supervised_n", "model"])
    detail["window_group"] = pd.cut(
        detail["supervised_n"], bins=[0, 30, 60, np.inf],
        labels=["18_to_30_targets", "54_to_60_targets", "84_targets"],
        include_lowest=True,
    ).astype(str)
    write_csv(tables / "training_window_performance_by_origin.csv", detail)

    summary = (detail.groupby(["country", "window_group", "model", "model_label"], observed=True)
               .agg(n_origins=("origin", "nunique"),
                    supervised_n_min=("supervised_n", "min"),
                    supervised_n_max=("supervised_n", "max"),
                    mean_RMSE_MW=("RMSE_MW", "mean"),
                    median_RMSE_MW=("RMSE_MW", "median"),
                    mean_WAPE_pct=("WAPE_pct", "mean"))
               .reset_index())
    write_csv(tables / "training_window_performance_summary.csv", summary)

    selected_models = ["LSTM", "LGBM", "TFT", "ENSEMBLE", "ETS_damped", "SeasonalNaive"]
    for country in ("SPAIN", "FRANCE"):
        fig, ax = plt.subplots(figsize=(8.0, 5.2))
        subset = detail[(detail.country == country) & detail.model.isin(selected_models)]
        for model, group in subset.groupby("model"):
            group = group.sort_values("supervised_n")
            ax.plot(group.supervised_n, group.RMSE_MW, marker="o", label=model_label(model))
        ax.set_xlabel("Supervised training targets")
        ax.set_ylabel("Complete-path RMSE [MW]")
        ax.set_title(f"{country.title()}: forecasting error and training-window length")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, ncol=2)
        save_figure(fig, figures, f"{country.lower()}_training_window_performance")


def horizon_error_analysis(tables: Path, figures: Path) -> None:
    all_paths = []
    for country in COUNTRIES:
        frame = pd.read_csv(ROOT / "frozen_statistical" / country / "statistical_benchmark_paths.csv",
                            parse_dates=["origin", "ds"])
        frame["country"] = country.upper()
        frame["horizon_band"] = frame["horizon"].map(horizon_band)
        all_paths.append(frame)
    paths = pd.concat(all_paths, ignore_index=True)

    horizon_rows = []
    for keys, group in paths.groupby(["country", "phase", "model", "horizon"]):
        country, phase, model, horizon = keys
        row = {"country": country, "phase": phase, "model": model,
               "model_label": model_label(model), "horizon": int(horizon)}
        row.update(point_metrics(group.y_true, group.pred_MW))
        horizon_rows.append(row)
    by_horizon = pd.DataFrame(horizon_rows)
    write_csv(tables / "error_by_horizon_and_phase.csv", by_horizon)

    band_rows = []
    for keys, group in paths.groupby(["country", "phase", "model", "horizon_band"]):
        country, phase, model, band = keys
        row = {"country": country, "phase": phase, "model": model,
               "model_label": model_label(model), "horizon_band": band}
        row.update(point_metrics(group.y_true, group.pred_MW))
        band_rows.append(row)
    by_band = pd.DataFrame(band_rows)
    write_csv(tables / "error_by_horizon_band.csv", by_band)

    selected_models = ["LSTM", "LGBM", "TFT", "ENSEMBLE", "ETS_damped", "SeasonalNaive"]
    overall = []
    for keys, group in paths.groupby(["country", "model", "horizon"]):
        country, model, horizon = keys
        row = {"country": country, "model": model, "horizon": int(horizon)}
        row.update(point_metrics(group.y_true, group.pred_MW))
        overall.append(row)
    overall = pd.DataFrame(overall)
    write_csv(tables / "error_by_horizon_all_origins.csv", overall)
    for country in ("SPAIN", "FRANCE"):
        fig, ax = plt.subplots(figsize=(8.0, 5.2))
        subset = overall[(overall.country == country) & overall.model.isin(selected_models)]
        for model, group in subset.groupby("model"):
            group = group.sort_values("horizon")
            ax.plot(group.horizon, group.RMSE_MW, marker="o", markersize=3,
                    label=model_label(model))
        ax.set_xlabel("Forecast horizon [months]")
        ax.set_ylabel("RMSE [MW]")
        ax.set_title(f"{country.title()}: retrospective error by forecast horizon")
        ax.set_xticks([1, 3, 6, 12, 18, 24])
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, ncol=2)
        save_figure(fig, figures, f"{country.lower()}_error_by_horizon")


def conditional_interval_analysis(tables: Path, figures: Path) -> None:
    rows = []
    for country in COUNTRIES:
        audit = pd.read_csv(ROOT / "frozen_statistical" / country / "prequential_probability_predictions.csv",
                            parse_dates=["origin", "ds"])
        terminal = pd.read_csv(ROOT / "frozen_statistical" / country / "holdout_recursive_predictions.csv",
                               parse_dates=["origin", "ds"])
        audit = audit[audit.variant.eq("SELECTED_BASELINE")].copy()
        terminal = terminal[terminal.variant.eq("SELECTED_BASELINE")].copy()
        audit["evaluation_period"] = "PREQUENTIAL_AUDIT"
        terminal["evaluation_period"] = "TERMINAL_RETROSPECTIVE"
        combined = pd.concat([audit, terminal], ignore_index=True)
        combined["horizon_band"] = combined.horizon.map(horizon_band)
        combined["interval_width_MW"] = combined.P95_MW - combined.P5_MW
        combined["interval_score_MW"] = interval_score(
            combined.y_true.to_numpy(), combined.P5_MW.to_numpy(), combined.P95_MW.to_numpy())
        for keys, group in combined.groupby(["evaluation_period", "horizon_band"]):
            period, band = keys
            rows.append({
                "country": country.upper(),
                "evaluation_period": period,
                "horizon_band": band,
                "n": int(len(group)),
                "coverage_pct": float(100.0 * group.covered.mean()),
                "below_P5": int(group.below_P5.sum()),
                "above_P95": int(group.above_P95.sum()),
                "mean_width_MW": float(group.interval_width_MW.mean()),
                "interval_score_MW": float(group.interval_score_MW.mean()),
                "CRPS_MW": float(group.CRPS_MW.mean()),
            })
    result = pd.DataFrame(rows)
    write_csv(tables / "conditional_interval_calibration_by_horizon_band.csv", result)

    for country in ("SPAIN", "FRANCE"):
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        subset = result[result.country.eq(country)].copy()
        order = [f"h{a:02d}_h{b:02d}" for a, b in HORIZON_BANDS]
        for period, group in subset.groupby("evaluation_period"):
            group = group.set_index("horizon_band").reindex(order).reset_index()
            ax.plot(order, group.coverage_pct, marker="o", label=period.replace("_", " ").title())
        ax.axhline(90.0, linestyle="--", linewidth=1, label="Nominal 90%")
        ax.set_xlabel("Forecast-horizon band")
        ax.set_ylabel("Observed coverage [%]")
        ax.set_title(f"{country.title()}: conditional interval coverage by horizon")
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        save_figure(fig, figures, f"{country.lower()}_coverage_by_horizon_band")


def residual_scale_sensitivity(tables: Path, figures: Path) -> None:
    terminal_rows = []
    future_rows = []
    base_checks = []
    for country in COUNTRIES:
        selected = json.loads((ROOT / "frozen_statistical" / country /
                               "probability_selection_frozen.json").read_text(encoding="utf-8"))["selected"]
        if selected["family"] != "HORIZON_SCALE":
            raise ValueError(f"{country}: the archived residual sensitivity expects HORIZON_SCALE.")
        selected_scale = float(selected["scale"])

        holdout_frame = pd.read_csv(ROOT / "frozen_statistical" / country /
                                    "holdout_recursive_predictions.csv")
        with np.load(ROOT / "frozen_statistical" / country / "holdout_monthly_simulations.npz",
                     allow_pickle=False) as data:
            samples = data["samples_MW"].astype(float)
        deterministic = holdout_frame["pred_ENS"].to_numpy(float)
        actual = holdout_frame["y_true"].to_numpy(float)
        log_ratio = np.log(samples / deterministic[None, :])
        scales = sorted(set(round(x, 6) for x in
                            [selected_scale * 0.85, selected_scale, selected_scale * 1.15, 1.0]))
        for absolute_scale in scales:
            adjusted = deterministic[None, :] * np.exp(log_ratio * (absolute_scale / selected_scale))
            row = {
                "country": country.upper(),
                "absolute_scale": absolute_scale,
                "relative_to_selected": absolute_scale / selected_scale,
                "selected_scale": selected_scale,
                "is_selected": bool(abs(absolute_scale - selected_scale) < 1e-10),
                "interpretation": "diagnostic_only_not_a_reselection",
            }
            row.update(probabilistic_metrics(adjusted, actual))
            terminal_rows.append(row)

        future_frame = pd.read_csv(ROOT / "frozen_statistical" / country /
                                   "monthly_peak_demand_scenarios.csv", parse_dates=["ds"])
        with np.load(ROOT / "frozen_statistical" / country / "future_monthly_simulations.npz",
                     allow_pickle=False) as data:
            future_samples = data["samples_MW"].astype(float)
            future_dates = pd.to_datetime(data["dates"].astype(str))
        future_det = future_frame["pred_ENS"].to_numpy(float)
        future_log_ratio = np.log(future_samples / future_det[None, :])
        horizons = np.arange(1, future_samples.shape[1] + 1)
        for variant, endpoint in FUTURE_INFLATION_ENDPOINTS.items():
            inflation = np.ones_like(horizons, dtype=float)
            later = horizons > 24
            inflation[later] = 1.0 + (endpoint - 1.0) * (horizons[later] - 24) / (horizons[-1] - 24)
            adjusted = future_det[None, :] * np.exp(future_log_ratio * inflation[None, :])
            annual = annual_sample_quantiles(adjusted, future_dates)
            annual.insert(0, "variant", variant)
            annual.insert(0, "country", country.upper())
            annual["end_multiplier_at_h240"] = endpoint
            future_rows.append(annual)

        frozen = pd.concat(future_rows, ignore_index=True)
        frozen = frozen[(frozen.country == country.upper()) & (frozen.variant == "FROZEN_AFTER_24")]
        expected = pd.read_csv(ROOT / "frozen_statistical" / country /
                               "annual_peak_demand_scenarios.csv")
        check = frozen.merge(expected[["year", "P5_MW", "P50_MW", "P95_MW"]], on="year",
                             suffixes=("_calc", "_archive"))
        differences = []
        for col in ("P5_MW", "P50_MW", "P95_MW"):
            differences.append(np.max(np.abs(check[f"{col}_calc"] - check[f"{col}_archive"])))
        base_checks.append({"country": country.upper(), "max_abs_difference_MW": float(max(differences))})

    terminal = pd.DataFrame(terminal_rows)
    future = pd.concat(future_rows, ignore_index=True)
    write_csv(tables / "terminal_residual_scale_sensitivity.csv", terminal)
    write_csv(tables / "future_residual_inflation_sensitivity.csv", future)
    write_csv(tables / "future_residual_inflation_reproduction_check.csv", pd.DataFrame(base_checks))

    for country in ("SPAIN", "FRANCE"):
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        subset = terminal[terminal.country.eq(country)].sort_values("absolute_scale")
        ax.plot(subset.absolute_scale, subset.coverage_pct, marker="o", label="Coverage")
        ax.axhline(90.0, linestyle="--", linewidth=1, label="Nominal 90%")
        selected = subset[subset.is_selected]
        if not selected.empty:
            ax.axvline(float(selected.absolute_scale.iloc[0]), linestyle=":", linewidth=1,
                       label="Selected scale")
        ax.set_xlabel("Residual scale multiplier")
        ax.set_ylabel("Terminal coverage [%]")
        ax.set_title(f"{country.title()}: terminal coverage sensitivity")
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        save_figure(fig, figures, f"{country.lower()}_terminal_scale_sensitivity")

        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        subset = future[future.country.eq(country)]
        for variant, group in subset.groupby("variant"):
            ax.plot(group.year, group.width_MW / 1000.0, marker="o", markersize=3,
                    label=variant.replace("_", " ").title())
        ax.set_xlabel("Calendar year")
        ax.set_ylabel("Annual P5-P95 width [GW]")
        ax.set_title(f"{country.title()}: long-horizon residual-scale sensitivity")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        save_figure(fig, figures, f"{country.lower()}_future_interval_inflation")


def alternative_baseline_analysis(tables: Path, figures: Path) -> None:
    monthly_rows = []
    annual_rows = []
    summary_rows = []
    stability_rows = []
    for country in COUNTRIES:
        future = pd.read_csv(ROOT / "frozen_statistical" / country /
                             "monthly_peak_demand_scenarios.csv", parse_dates=["ds"])
        increment = pd.read_csv(ROOT / "verified_outputs" / country / "TURNOVER" /
                                "monthly_peak_scenarios.csv", parse_dates=["ds"])
        if not pd.DatetimeIndex(future.ds).equals(pd.DatetimeIndex(increment.ds)):
            raise ValueError(f"{country}: structural increments and statistical paths are misaligned.")
        weights = pd.read_csv(ROOT / "frozen_statistical" / country / "ensemble_weights.csv")
        weight = dict(zip(weights.model, weights.weight))
        paths = {
            "LSTM": future.pred_LSTM.to_numpy(float),
            "LGBM": future.pred_LGBM.to_numpy(float),
            "TFT": future.pred_TFT.to_numpy(float),
            "INVERSE_RMSE_TRI_MODEL": future.pred_ENS.to_numpy(float),
            "EQUAL_WEIGHT_TRI_MODEL": future[["pred_LSTM", "pred_LGBM", "pred_TFT"]].mean(axis=1).to_numpy(float),
        }
        for pair in (("LSTM", "LGBM"), ("LSTM", "TFT"), ("LGBM", "TFT")):
            denominator = weight[pair[0]] + weight[pair[1]]
            paths[f"{pair[0]}_{pair[1]}"] = (
                weight[pair[0]] * paths[pair[0]] + weight[pair[1]] * paths[pair[1]]) / denominator
        dates = pd.DatetimeIndex(future.ds)
        structural = increment.structural_increment_MW.to_numpy(float)
        for name, baseline in paths.items():
            total = baseline + structural
            monthly_rows.append(pd.DataFrame({
                "country": country.upper(), "ds": dates, "baseline": name,
                "statistical_baseline_MW": baseline,
                "structural_increment_MW": structural,
                "turnover_total_MW": total,
            }))
            previous = None
            for year in sorted(set(dates.year)):
                mask = dates.year == year
                base_values = baseline[mask]
                total_values = total[mask]
                annual_rows.append({
                    "country": country.upper(), "year": int(year), "baseline": name,
                    "statistical_annual_max_MW": float(base_values.max()),
                    "turnover_annual_max_MW": float(total_values.max()),
                    "structural_effect_on_annual_max_MW": float(total_values.max() - base_values.max()),
                    "critical_month": int(dates[mask][int(np.argmax(total_values))].month),
                })
                corr = np.nan
                rmse = np.nan
                if previous is not None:
                    corr = float(np.corrcoef(previous, total_values)[0, 1])
                    rmse = float(np.sqrt(np.mean((previous - total_values) ** 2)))
                stability_rows.append({
                    "country": country.upper(), "baseline": name, "year": int(year),
                    "profile_correlation_previous_year": corr,
                    "profile_RMSE_previous_year_MW": rmse,
                    "annual_mean_MW": float(total_values.mean()),
                    "annual_max_MW": float(total_values.max()),
                    "annual_min_MW": float(total_values.min()),
                })
                previous = total_values.copy()
        annual_country = pd.DataFrame([x for x in annual_rows if x["country"] == country.upper()])
        final = annual_country[annual_country.year.eq(2045)].copy()
        reference = float(final.loc[final.baseline.eq("INVERSE_RMSE_TRI_MODEL"),
                                    "turnover_annual_max_MW"].iloc[0])
        final["difference_from_reference_MW"] = final.turnover_annual_max_MW - reference
        summary_rows.extend(final.to_dict("records"))

        fig, ax = plt.subplots(figsize=(8.0, 5.2))
        show = annual_country[annual_country.baseline.isin([
            "LSTM", "LGBM", "TFT", "INVERSE_RMSE_TRI_MODEL",
            "EQUAL_WEIGHT_TRI_MODEL", "LGBM_TFT"])]
        for name, group in show.groupby("baseline"):
            ax.plot(group.year, group.turnover_annual_max_MW / 1000.0,
                    marker="o", markersize=3, label=name.replace("_", " ").title())
        ax.set_xlabel("Calendar year")
        ax.set_ylabel("Deterministic annual peak under turnover [GW]")
        ax.set_title(f"{country.title()}: identical structural increment across statistical baselines")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, ncol=2)
        save_figure(fig, figures, f"{country.lower()}_alternative_baselines_turnover")

    monthly = pd.concat(monthly_rows, ignore_index=True)
    annual = pd.DataFrame(annual_rows)
    summary = pd.DataFrame(summary_rows)
    stability = pd.DataFrame(stability_rows)
    write_csv(tables / "alternative_baselines_monthly_turnover.csv", monthly)
    write_csv(tables / "alternative_baselines_annual_turnover.csv", annual)
    write_csv(tables / "alternative_baselines_2045_summary.csv", summary)
    write_csv(tables / "future_profile_stability_by_baseline.csv", stability)


def structural_range_analysis(tables: Path, figures: Path) -> None:
    rows = []
    for country in COUNTRIES:
        scenario_frames = []
        base = ROOT / "verified_outputs" / country
        for path in sorted(base.glob("*/annual_peak_scenarios.csv")):
            frame = pd.read_csv(path)
            frame["scenario_folder"] = path.parent.name
            scenario_frames.append(frame)
        all_scenarios = pd.concat(scenario_frames, ignore_index=True)
        turnover = all_scenarios[all_scenarios.scenario.eq("TURNOVER")].set_index("year")
        for year, group in all_scenarios.groupby("year"):
            turn = turnover.loc[year]
            rows.append({
                "country": country.upper(),
                "year": int(year),
                "n_structural_scenarios": int(group.scenario.nunique()),
                "scenario_P50_min_MW": float(group.P50_MW.min()),
                "scenario_P50_max_MW": float(group.P50_MW.max()),
                "scenario_P50_range_MW": float(group.P50_MW.max() - group.P50_MW.min()),
                "turnover_P5_MW": float(turn.P5_MW),
                "turnover_P50_MW": float(turn.P50_MW),
                "turnover_P95_MW": float(turn.P95_MW),
                "turnover_conditional_width_MW": float(turn.P95_MW - turn.P5_MW),
                "structural_range_to_conditional_width": float(
                    (group.P50_MW.max() - group.P50_MW.min()) / (turn.P95_MW - turn.P5_MW)),
                "interpretation": "structural_range_has_no_probability_weight",
            })
    result = pd.DataFrame(rows)
    write_csv(tables / "conditional_interval_and_structural_range.csv", result)
    for country in ("SPAIN", "FRANCE"):
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        subset = result[result.country.eq(country)]
        ax.plot(subset.year, subset.turnover_conditional_width_MW / 1000.0,
                marker="o", label="Turnover conditional P5-P95 width")
        ax.plot(subset.year, subset.scenario_P50_range_MW / 1000.0,
                marker="s", label="Range of scenario P50 values")
        ax.set_xlabel("Calendar year")
        ax.set_ylabel("Spread [GW]")
        ax.set_title(f"{country.title()}: statistical band and structural scenario range")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        save_figure(fig, figures, f"{country.lower()}_statistical_and_structural_spread")


def covariate_role_table(tables: Path) -> None:
    rows = [
        {"feature": "mon_sin", "source": "calendar", "role": "annual periodicity, sine component",
         "future_treatment": "known deterministically", "measured_exogenous": False},
        {"feature": "mon_cos", "source": "calendar", "role": "annual periodicity, cosine component",
         "future_treatment": "known deterministically", "measured_exogenous": False},
        {"feature": "trend", "source": "calendar", "role": "elapsed-time component for gradual level change",
         "future_treatment": "continued deterministically", "measured_exogenous": False},
        {"feature": "gdp_idx", "source": "World Bank", "role": "slow macroeconomic scale",
         "future_treatment": "log Theil-Sen trend using admissible history", "measured_exogenous": True},
        {"feature": "population_idx", "source": "World Bank", "role": "slow demographic scale",
         "future_treatment": "log Theil-Sen trend using admissible history", "measured_exogenous": True},
        {"feature": "ipi", "source": "Eurostat", "role": "monthly industrial activity and seasonal variation",
         "future_treatment": "seasonal repetition or damped seasonal ETS", "measured_exogenous": True},
        {"feature": "hdd", "source": "Eurostat", "role": "heating-related monthly weather variation",
         "future_treatment": "month-specific median of latest admissible ten years", "measured_exogenous": True},
        {"feature": "cdd", "source": "Eurostat", "role": "cooling-related monthly weather variation",
         "future_treatment": "month-specific median of latest admissible ten years", "measured_exogenous": True},
    ]
    write_csv(tables / "covariate_roles.csv", pd.DataFrame(rows))


def load_core_for_retraining():
    sys.path.insert(0, str(ROOT))
    import run_forecasting_final as core
    core.stat_load_probability_stack()
    core.stat_require_real_triad()
    return core


def seasonal_naive(history: np.ndarray, horizon: int) -> np.ndarray:
    values = list(np.asarray(history, dtype=float))
    result = []
    for _ in range(horizon):
        result.append(float(values[-12]))
        values.append(result[-1])
    return np.asarray(result)


def long_horizon_retraining(tables: Path, figures: Path, countries: Iterable[str]) -> None:
    """Fit the original three learners at later origins and evaluate 36-54 month paths."""
    core = load_core_for_retraining()
    cfg = json.loads((ROOT / "data" / "statistical_configuration.json").read_text(encoding="utf-8"))["config"]
    repo = core.stat_DriverRepository.from_folder(ROOT / "data" / "statistical" / "exogenous_cache", cfg)
    detail_rows = []
    metric_rows = []
    for country in countries:
        demand = core.stat_load_demand(core.demand_file(country), country, cfg)
        demand = demand.sort_values("ds").reset_index(drop=True)
        weight_table = pd.read_csv(ROOT / "frozen_statistical" / country.lower() / "ensemble_weights.csv")
        weights = dict(zip(weight_table.model, weight_table.weight))
        for origin_text in LONG_HORIZON_ORIGINS:
            origin = pd.Timestamp(origin_text)
            if origin not in set(demand.ds):
                continue
            end = demand.ds.max()
            horizon = (end.year - origin.year) * 12 + end.month - origin.month
            if horizon < 36:
                continue
            features, _ = repo.features(country.upper(), demand.ds.min(), origin, end)
            training = demand[demand.ds <= origin].merge(features, on="ds", how="left", validate="one_to_one")
            future = features[features.ds > origin].reset_index(drop=True)
            log_file = tables.parent / "logs" / f"{country.lower()}_{origin:%Y%m}_libraries.log"
            log_file.parent.mkdir(parents=True, exist_ok=True)
            bundle = core.stat_fit_triad(training, country.upper(), cfg, log_file)
            paths = core.stat_recursive_paths(bundle, future)
            paths["ENSEMBLE"] = sum(weights[name] * paths[name] for name in ("LSTM", "LGBM", "TFT"))
            paths["EQUAL_WEIGHT"] = np.mean(np.vstack([paths["LSTM"], paths["LGBM"], paths["TFT"]]), axis=0)
            actual = demand.set_index("ds").reindex(future.ds).y_mw.to_numpy(float)
            paths["PERSISTENCE"] = np.repeat(training.y_mw.iloc[-1], horizon)
            paths["SEASONAL_NAIVE"] = seasonal_naive(training.y_mw.to_numpy(float), horizon)
            for model, prediction in paths.items():
                for h, (date, y, p) in enumerate(zip(future.ds, actual, prediction), start=1):
                    detail_rows.append({"country": country.upper(), "origin": origin, "ds": date,
                                        "horizon": h, "model": model, "y_true": y, "pred_MW": p,
                                        "supervised_n": len(training) - cfg["lookback"]})
                for end_h in (24, 36, 42, 48, 54):
                    if end_h <= horizon:
                        row = {"country": country.upper(), "origin": origin, "model": model,
                               "evaluation_horizon_months": end_h,
                               "supervised_n": len(training) - cfg["lookback"]}
                        row.update(point_metrics(actual[:end_h], prediction[:end_h]))
                        metric_rows.append(row)
            del bundle
    detail = pd.DataFrame(detail_rows)
    metrics = pd.DataFrame(metric_rows)
    write_csv(tables / "extended_horizon_recursive_predictions.csv", detail)
    write_csv(tables / "extended_horizon_metrics.csv", metrics)
    if metrics.empty:
        return
    for country in sorted(metrics.country.unique()):
        fig, ax = plt.subplots(figsize=(8.0, 5.2))
        subset = metrics[(metrics.country == country) &
                         metrics.model.isin(["LSTM", "LGBM", "TFT", "ENSEMBLE", "EQUAL_WEIGHT",
                                            "PERSISTENCE", "SEASONAL_NAIVE"])]
        summary = (subset.groupby(["model", "evaluation_horizon_months"])
                   .RMSE_MW.mean().reset_index())
        for model, group in summary.groupby("model"):
            ax.plot(group.evaluation_horizon_months, group.RMSE_MW, marker="o",
                    label=model.replace("_", " ").title())
        ax.set_xlabel("Retrospective path length [months]")
        ax.set_ylabel("Mean RMSE across available origins [MW]")
        ax.set_title(f"{country.title()}: extended retrospective recursive paths")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, ncol=2)
        save_figure(fig, figures, f"{country.lower()}_extended_horizon_metrics")


def package_run(run: Path, destination: Path) -> None:
    temp = destination.with_suffix(".tmp.zip")
    if temp.exists():
        temp.unlink()
    with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(run.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(run.parent))
    temp.replace(destination)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None,
                        help="Output folder. The default is extended_analysis/results/run_<timestamp>.")
    parser.add_argument("--long-horizon-retraining", action="store_true",
                        help="Fit the same three learners at later historical origins for 36-54 month diagnostics.")
    parser.add_argument("--countries", nargs="+", default=["SPAIN", "FRANCE"],
                        help="Countries for the optional retraining mode.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow replacement of a nonempty explicit output folder.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    stamp = datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    run = args.output.resolve() if args.output else HERE / "results" / stamp
    if run.exists() and any(run.iterdir()):
        if not args.overwrite:
            raise ValueError(f"Output folder is not empty: {run}")
        shutil.rmtree(run)
    tables = run / "tables"
    figures = run / "figures"
    run.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    write_json(run / "execution_status.json", {"status": "RUNNING", "long_horizon_retraining": args.long_horizon_retraining})
    try:
        training_window_analysis(tables, figures)
        horizon_error_analysis(tables, figures)
        conditional_interval_analysis(tables, figures)
        residual_scale_sensitivity(tables, figures)
        alternative_baseline_analysis(tables, figures)
        structural_range_analysis(tables, figures)
        covariate_role_table(tables)
        if args.long_horizon_retraining:
            countries = [c.upper() for c in args.countries]
            long_horizon_retraining(tables, figures, countries)
        elapsed = time.monotonic() - started
        try:
            output_label = str(run.relative_to(ROOT))
        except ValueError:
            output_label = str(run)
        status = {
            "status": "COMPLETED",
            "elapsed_seconds": elapsed,
            "long_horizon_retraining": bool(args.long_horizon_retraining),
            "main_experiment_modified": False,
            "archived_predictions_reused": True,
            "output_folder": output_label,
            "n_tables": len(list(tables.glob("*.csv"))),
            "n_figures_png": len(list(figures.glob("*.png"))),
            "n_figures_pdf": len(list(figures.glob("*.pdf"))),
        }
        write_json(run / "execution_status.json", status)
        report = [
            "EXTENDED ANALYSIS",
            "",
            "The main forecasting experiment was not changed.",
            "The default diagnostics reuse archived predictions, simulations and scenario outputs.",
            f"Long-horizon retraining executed: {args.long_horizon_retraining}",
            f"Tables: {status['n_tables']}",
            f"Figures: {status['n_figures_png']} PNG and {status['n_figures_pdf']} PDF",
            f"Elapsed seconds: {elapsed:.2f}",
            "",
            "The residual-scale alternatives are diagnostics, not a new hyperparameter selection.",
            "The structural scenario range has no probability weights and is not merged with the conditional statistical band.",
        ]
        (run / "REPORT.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
        (HERE / "LATEST_RUN.txt").write_text(output_label + "\n", encoding="utf-8")
        package_run(run, HERE / "EXTENDED_ANALYSIS_TO_SHARE.zip")
        print(f"Extended analysis completed: {run}")
    except Exception as exc:
        detail = traceback.format_exc()
        write_json(run / "execution_status.json", {"status": "FAILED", "error": str(exc), "traceback": detail})
        (run / "ERROR.txt").write_text(detail, encoding="utf-8")
        package_run(run, HERE / "EXTENDED_ANALYSIS_TO_SHARE.zip")
        raise


if __name__ == "__main__":
    main()
