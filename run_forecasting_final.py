#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Monthly peak-demand forecasting and conditional physical scenarios.

Run this file to reproduce the archived release experiment. Use --retrain for a new
fit and --statistical-only for countries without a physical configuration.
Inputs and outputs are resolved relative to this file. See README.txt.
"""
from __future__ import annotations

import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import argparse
import ast
import copy
import gc
import hashlib
import importlib
import importlib.metadata
import json
import logging
import platform
import re
import shutil
import sys
import time
import traceback
import warnings
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
except ImportError as exc:
    raise RuntimeError('A required library is missing. See requirements.txt or use the environment recorded for the archived release.') from exc

MODE = 'REPRODUCE_RELEASE'
COUNTRIES = ['SPAIN', 'FRANCE']
STATISTICAL_ONLY = False
ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data/structural'
REF = ROOT / 'frozen_statistical'
VERSION = 'forecasting-final-release-1.2'
EXECUTION_MODE = MODE
EPS = 1e-7
CHECKS: list[dict[str, Any]] = []

# Statistical baseline: data, training, recursive prediction and errors.
# The stat_ prefix separates these functions from the physical accounting.
import random
import types
from contextlib import contextmanager, redirect_stdout, redirect_stderr

stat_CONFIG = {'countries': 'AUTO',
 'forecast_years': 20,
 'minimum_history_months': 120,
 'lookback': 24,
 'tabular_lags': 12,
 'evaluation_horizon': 24,
 'terminal_holdout_months': 24,
 'seed': 42,
 'num_threads': 1,
 'device': 'cpu',
 'lstm_hidden': 64,
 'lstm_layers': 1,
 'lstm_dropout': 0.1,
 'lstm_epochs': 150,
 'lstm_batch_size': 16,
 'lstm_learning_rate': 0.001,
 'lgbm_rounds': 800,
 'lgbm_num_leaves': 31,
 'lgbm_min_data_in_leaf': 10,
 'lgbm_learning_rate': 0.05,
 'tft_epochs': 20,
 'tft_batch_size': 32,
 'tft_hidden': 16,
 'tft_attention_heads': 2,
 'tft_hidden_continuous': 16,
 'tft_dropout': 0.1,
 'tft_learning_rate': 0.001,
 'annual_reference_lag_years': 1,
 'annual_release_year_end_offset': 1,
 'monthly_release_lag_months': 3,
 'macro_trend_years': 15,
 'climatology_years': 10,
 'mc_tune_simulations': 1500,
 'mc_final_simulations': 5000,
 'mc_blocks': [1, 3, 6, 12],
 'mc_scale_min': 0.5,
 'mc_scale_max': 2.5,
 'mc_scale_step': 0.05,
 'nominal_coverage': 0.9,
 'coverage_tolerance': 0.03,
 'dependence_acf_threshold': 0.3,
 'dependence_lb_fraction': 0.5,
 'save_figures': True,
 'save_model_parameters': True,
 'show_final_figures': True,
 'resume_completed_origins': True,
 'reuse_verified_forecasts': True,
 'probability_families': ['HORIZON_SCALE', 'CYCLIC_SCALE', 'CYCLIC_LOCATION_SCALE'],
 'probability_prior_months': 12.0,
 'probability_month_bandwidth': 2.0,
 'probability_horizon_bandwidth': 6.0,
 'inner_min_target_months': 24,
 'inner_validation_months': 6,
 'parsimony_relative_tolerance': 0.01}
stat_MODELS = ("LSTM", "LGBM", "TFT")
stat_FEATURES = ["mon_sin", "mon_cos", "trend", "gdp_idx", "population_idx", "ipi", "hdd", "cdd"]
stat_ANNUAL_DRIVERS = ("gdp_const2015_usd", "population")
stat_MONTHLY_DRIVERS = ("ipi", "hdd", "cdd")
stat_VERSION = "forecasting_statistical_baseline_1.0.0"
stat_SOURCE_DETERMINISTIC_VERSION = "monthly_triad_core_1.0.1"
try:
    stat_ROOT = Path(__file__).resolve().parent
except NameError:
    stat_ROOT = Path.cwd()
stat_np = stat_pd = stat_torch = stat_nn = stat_lgb = stat_plt = None
stat_StandardScaler = stat_TheilSenRegressor = stat_ExponentialSmoothing = None
stat_acf = stat_acorr_ljungbox = None
stat_STACK = {}
stat_LOG = logging.getLogger("forecasting_baseline")


def stat_write_json(obj, path):
    def default(value):
        if hasattr(value, "tolist"):
            return value.tolist()
        if hasattr(value, "item"):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        if hasattr(value, "isoformat"):
            return value.isoformat()
        raise TypeError(type(value).__name__)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=default), encoding="utf-8")
    temp.replace(path)


def stat_write_csv(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False, encoding="utf-8", date_format="%Y-%m-%d")
    temp.replace(path)


def stat_environment_versions():
    rows = [{"package": "Python", "version": platform.python_version()},
            {"package": "platform", "version": platform.platform()},
            {"package": "executable", "version": sys.executable}]
    for name in ("numpy", "pandas", "scipy", "scikit-learn", "statsmodels", "lightgbm", "torch",
                 "lightning", "pytorch-forecasting", "matplotlib"):
        try:
            ver = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            ver = "NOT_INSTALLED"
        rows.append({"package": name, "version": ver})
    return stat_pd.DataFrame(rows) if stat_pd is not None else rows


def stat_seed_all(seed, threads=1):
    random.seed(seed)
    stat_np.random.seed(seed)
    stat_torch.manual_seed(seed)
    stat_torch.set_num_threads(threads)
    if stat_torch.cuda.is_available():
        stat_torch.cuda.manual_seed_all(seed)
    # CPU training with a fixed seed.
    stat_torch.use_deterministic_algorithms(True, warn_only=True)


def stat_finite(values, name, positive=False):
    a = stat_np.asarray(values, dtype=float)
    if not stat_np.isfinite(a).all() or (positive and (a <= 0).any()):
        raise ValueError(f"{name}: non-finite or nonpositive values. No clipping is applied.")
    return a


def stat_month_start(date):
    return stat_pd.Timestamp(date).to_period("M").to_timestamp()


def stat_month_end(date):
    return stat_month_start(date) + stat_pd.offsets.MonthEnd(0)


def stat_months_between(later, earlier):
    later, earlier = stat_pd.Timestamp(later), stat_pd.Timestamp(earlier)
    return (later.year - earlier.year) * 12 + later.month - earlier.month


class stat_DriverRepository:
    """Local driver data and availability dates for each forecast origin."""
    def __init__(self, annual, monthly, cfg):
        self.annual = annual.copy()
        self.monthly = monthly.copy()
        self.monthly["ds"] = stat_pd.to_datetime(self.monthly.ds).dt.to_period("M").dt.to_timestamp()
        self.cfg = cfg
        self._macro = {}
        self._ipi = {}
        if self.annual.duplicated(["country", "year"]).any():
            raise ValueError("Duplicate rows in the annual macroeconomic cache.")
        if self.monthly.duplicated(["country", "ds"]).any():
            raise ValueError("Duplicate rows in the monthly exogenous cache.")

    @classmethod
    def from_folder(cls, folder, cfg):
        return cls(stat_pd.read_csv(folder / "exogenous_annual_worldbank.csv"),
                   stat_pd.read_csv(folder / "exogenous_monthly_eurostat.csv"), cfg)

    def annual_known(self, country, col, origin):
        d = self.annual.loc[self.annual.country.eq(country), ["year", col]].copy()
        offset = self.cfg["annual_release_year_end_offset"]
        eligible = d.year.map(lambda y: stat_pd.Timestamp(int(y) + offset, 12, 31)) <= stat_month_end(origin)
        d = d.loc[eligible].dropna().sort_values("year").reset_index(drop=True)
        if len(d) < 5 or not stat_np.all(stat_np.diff(d.year.to_numpy(int)) == 1):
            raise ValueError(f"{country}/{col}: at least five consecutive observed years are required at {origin}.")
        stat_finite(d[col], f"{country}/{col}", positive=True)
        return d

    def monthly_known(self, country, col, origin):
        d = self.monthly.loc[self.monthly.country.eq(country), ["ds", col]].copy()
        lag = self.cfg["monthly_release_lag_months"]
        eligible = d.ds + stat_pd.offsets.MonthEnd(lag + 1) <= stat_month_end(origin)
        d = d.loc[eligible].dropna().sort_values("ds").reset_index(drop=True)
        if d.empty or not stat_pd.DatetimeIndex(d.ds).equals(stat_pd.date_range(d.ds.min(), d.ds.max(), freq="MS")):
            raise ValueError(f"{country}/{col}: available data are empty or contain internal gaps at {origin}.")
        v = stat_finite(d[col], f"{country}/{col}", positive=(col == "ipi"))
        if (v < 0).any():
            raise ValueError(f"{country}/{col}: a negative observation was found.")
        return d

    def macro_predict(self, country, col, origin, years):
        req = stat_np.asarray(years, int)
        key = (country, col, str(stat_month_start(origin)))
        if key not in self._macro:
            d = self.annual_known(country, col, origin)
            last = d.tail(self.cfg["macro_trend_years"])
            model = stat_TheilSenRegressor(random_state=self.cfg["seed"], fit_intercept=True)
            model.fit(last.year.to_numpy(float).reshape(-1, 1), stat_np.log(last[col].to_numpy(float)))
            self._macro[key] = (d, model)
        d, model = self._macro[key]
        if len(req) and req.min() < int(d.year.min()):
            raise ValueError(f"{country}/{col}: observations before the first cached year cannot be invented.")
        with stat_np.errstate(over="raise", invalid="raise"):
            pred = stat_np.exp(model.predict(req.reshape(-1, 1)))
        obs = dict(zip(d.year.astype(int), d[col].astype(float)))
        pred = stat_finite([obs.get(int(y), p) for y, p in zip(req, pred)], col, positive=True)
        return pred, {"country": country, "driver": col, "origin": stat_month_start(origin),
                      "method": "TheilSen_log_trend_max15y", "last_observed_year": int(d.year.max()),
                      "n_observed_years": len(d), "n_fit_years": min(len(d), self.cfg["macro_trend_years"]),
                      "assumed_release_last": stat_pd.Timestamp(int(d.year.max()) + self.cfg["annual_release_year_end_offset"], 12, 31)}

    def ipi_predict(self, country, origin, requested):
        req = stat_pd.DatetimeIndex(requested)
        key = (country, str(stat_month_start(origin)))
        if key not in self._ipi:
            d = self.monthly_known(country, "ipi", origin)
            values = d.ipi.to_numpy(float)
            if len(values) < 12:
                raise ValueError("IPI requires at least twelve available months.")
            method, score, fit = "seasonal_repetition", float("nan"), None
            candidates = []
            if len(values) >= 36:
                nv = min(24, max(12, len(values) // 5))
                tr, va = values[:-nv], values[-nv:]
                candidates.append((stat_rmse(va, stat_seasonal_repetition(tr, nv)), "seasonal_repetition"))
                try:
                    ef = stat_ExponentialSmoothing(tr, trend="add", damped_trend=True,
                                              seasonal="add", seasonal_periods=12,
                                              initialization_method="estimated").fit(optimized=True, use_brute=True)
                    ep = stat_finite(ef.forecast(nv), "IPI internal ETS", positive=True)
                    candidates.append((stat_rmse(va, ep), "ipi_damped_ets"))
                except (ValueError, RuntimeError, FloatingPointError, stat_np.linalg.LinAlgError) as exc:
                    stat_LOG.warning("%s IPI: ETS candidate is inadmissible: %s", country, exc)
                score, method = min(candidates)
                if method == "ipi_damped_ets":
                    fit = stat_ExponentialSmoothing(values, trend="add", damped_trend=True,
                                               seasonal="add", seasonal_periods=12,
                                               initialization_method="estimated").fit(optimized=True, use_brute=True)
            self._ipi[key] = (d, method, score, fit)
        d, method, score, fit = self._ipi[key]
        last = stat_pd.Timestamp(d.ds.max())
        h = stat_np.asarray([stat_months_between(t, last) for t in req], int)
        max_h = max(0, int(h.max())) if len(h) else 0
        if max_h:
            forecast = fit.forecast(max_h) if fit is not None else stat_seasonal_repetition(d.ipi, max_h)
            forecast = stat_finite(forecast, "IPI forecast", positive=True)
        else:
            forecast = stat_np.empty(0)
        obs = dict(zip(d.ds, d.ipi.astype(float)))
        # Assign each forecast to its calendar date and actual horizon.
        pred = [obs.get(t, stat_np.nan) if hh <= 0 else forecast[hh - 1] for t, hh in zip(req, h)]
        return stat_finite(pred, "aligned IPI", positive=True), {
            "country": country, "driver": "ipi", "origin": stat_month_start(origin), "method": method,
            "inner_validation_RMSE": score, "last_observed_month": last,
            "max_calendar_horizon": max_h,
            "assumed_release_last": last + stat_pd.offsets.MonthEnd(self.cfg["monthly_release_lag_months"] + 1)}

    def weather_predict(self, country, col, origin, requested):
        known = self.monthly_known(country, col, origin)
        recent = known[known.ds > known.ds.max() - stat_pd.DateOffset(years=self.cfg["climatology_years"])]
        med = recent.groupby(recent.ds.dt.month)[col].median()
        if set(med.index) != set(range(1, 13)):
            raise ValueError(f"{country}/{col}: climatology is not available for all twelve months.")
        obs = dict(zip(known.ds, known[col].astype(float)))
        if len(requested) and min(requested) < known.ds.min():
            raise ValueError(f"{country}/{col}: the date precedes the first observation; zero filling is not allowed.")
        pred = [obs.get(stat_pd.Timestamp(t), float(med.loc[t.month])) for t in requested]
        return stat_finite(pred, col), {"country": country, "driver": col, "origin": stat_month_start(origin),
                                   "method": "past10y_monthly_median_climatology",
                                   "last_observed_month": known.ds.max(),
                                   "assumed_release_last": known.ds.max() + stat_pd.offsets.MonthEnd(self.cfg["monthly_release_lag_months"] + 1)}

    def features(self, country, start, origin, end):
        dates = stat_pd.date_range(stat_month_start(start), stat_month_start(end), freq="MS")
        x = stat_pd.DataFrame({"ds": dates})
        x["mon_sin"] = stat_np.sin(2 * stat_np.pi * (dates.month - 1) / 12)
        x["mon_cos"] = stat_np.cos(2 * stat_np.pi * (dates.month - 1) / 12)
        x["trend"] = [stat_months_between(t, start) / 12 for t in dates]
        rows = []
        refs = dates.year.to_numpy(int) - self.cfg["annual_reference_lag_years"]
        for col, feature in (("gdp_const2015_usd", "gdp_idx"), ("population", "population_idx")):
            values, meta = self.macro_predict(country, col, origin, refs)
            known = self.annual_known(country, col, origin)
            by = 2015 if 2015 in known.year.values else int(known.year.min())
            base = float(known.loc[known.year.eq(by), col].iloc[0])
            x[feature] = 100 * values / base
            meta["index_base_year"] = by
            meta["reference_lag_years"] = self.cfg["annual_reference_lag_years"]
            rows.append(meta)
        x["ipi"], meta = self.ipi_predict(country, origin, dates)
        rows.append(meta)
        for col in ("hdd", "cdd"):
            x[col], meta = self.weather_predict(country, col, origin, dates)
            rows.append(meta)
        stat_finite(x[stat_FEATURES], "exogenous feature matrix")
        diag = stat_pd.DataFrame(rows)
        diag["as_of"] = stat_month_end(origin)
        diag["availability_policy"] = "assumed_conservative_release_dates_single_revised_snapshot"
        return x, diag


def stat_seasonal_repetition(history, n):
    """Project IPI from its own history, not electricity demand."""
    work = list(stat_np.asarray(history, float))
    if len(work) < 12:
        raise ValueError("Twelve values are needed to repeat the seasonal IPI pattern.")
    result = []
    for _ in range(n):
        result.append(float(work[-12]))
        work.append(result[-1])
    return stat_np.asarray(result)


def stat_make_protocol(demand, cfg):
    """Build the calibration, audit and terminal forecast origins."""
    n, h = len(demand), cfg["evaluation_horizon"]
    holdout_origin = n - cfg["terminal_holdout_months"] - 1
    prob_last = holdout_origin - h
    prob_indices = [prob_last - 6, prob_last - 4, prob_last - 2, prob_last]
    weight_last = prob_indices[0] - h
    weight_first = max(cfg["lookback"] + 12 - 1, weight_last - 12)
    weight_indices = list(range(weight_last, weight_first - 1, -3))[::-1]
    if len(weight_indices) < 3:
        raise ValueError("This calendar does not provide three weight-calibration origins with enough training data.")
    rows = []
    for phase, indices in (("weight_calibration", weight_indices), ("probability_tuning", prob_indices),
                           ("terminal_holdout", [holdout_origin])):
        for i in indices:
            rows.append({"phase": phase, "origin": demand.ds.iloc[i], "train_n_months": i + 1,
                         "supervised_n": i + 1 - cfg["lookback"],
                         "target_start": demand.ds.iloc[i + 1], "target_end": demand.ds.iloc[i + h],
                         "n_horizons": h, "uses_for_selection": phase != "terminal_holdout"})
    out = stat_pd.DataFrame(rows)
    assert out[out.phase.eq("weight_calibration")].target_end.max() <= out[out.phase.eq("probability_tuning")].origin.min()
    assert out[out.phase.eq("probability_tuning")].target_end.max() <= out[out.phase.eq("terminal_holdout")].origin.min()
    return out


def stat_supervised(demand_values, x, lookback, lags):
    y, x = stat_finite(demand_values, "supervised targets"), stat_finite(x, "supervised features")
    if len(y) != len(x) or x.ndim != 2 or not 1 <= lags <= lookback:
        raise ValueError("Supervised inputs and targets are misaligned.")
    seq, tab, nxt, target = [], [], [], []
    for t in range(lookback, len(y)):
        seq.append(stat_np.column_stack([y[t-lookback:t], x[t-lookback:t]]))
        tab.append(stat_np.r_[y[t-stat_np.arange(1, lags+1)], x[t]])
        nxt.append(x[t])
        target.append(y[t])
    if not target:
        raise ValueError("No supervised windows are available.")
    return tuple(stat_np.asarray(a, stat_np.float32) for a in (seq, tab, nxt, target))


def stat_new_lstm(input_dim, next_dim, cfg):
    class LevelLSTM(stat_nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = stat_nn.LSTM(input_size=input_dim, hidden_size=cfg["lstm_hidden"],
                                   num_layers=cfg["lstm_layers"], batch_first=True,
                                   dropout=cfg["lstm_dropout"] if cfg["lstm_layers"] > 1 else 0.0)
            self.head = stat_nn.Sequential(stat_nn.Linear(cfg["lstm_hidden"] + next_dim, 64), stat_nn.ReLU(),
                                      stat_nn.Dropout(cfg["lstm_dropout"]), stat_nn.Linear(64, 1))

        def forward(self, sequence, known_next):
            encoded, _ = self.encoder(sequence)
            return self.head(stat_torch.cat([encoded[:, -1, :], known_next], dim=-1)).squeeze(-1)
    return LevelLSTM()


def stat_train_lstm(seq, next_x, target, cfg):
    stat_seed_all(cfg["seed"], cfg["num_threads"])
    model = stat_new_lstm(seq.shape[2], next_x.shape[1], cfg).to(cfg["device"])
    tensors = [stat_torch.as_tensor(a, dtype=stat_torch.float32) for a in (seq, next_x, target)]
    dataset = stat_torch.utils.data.TensorDataset(*tensors)
    loader = stat_torch.utils.data.DataLoader(dataset, batch_size=cfg["lstm_batch_size"], shuffle=True,
                                        num_workers=0, generator=stat_torch.Generator().manual_seed(cfg["seed"]))
    optimizer = stat_torch.optim.Adam(model.parameters(), lr=cfg["lstm_learning_rate"])
    trace = []
    for epoch in range(cfg["lstm_epochs"]):
        model.train()
        total, count = 0.0, 0
        for xs, xn, yt in loader:
            xs, xn, yt = xs.to(cfg["device"]), xn.to(cfg["device"]), yt.to(cfg["device"])
            optimizer.zero_grad()
            loss = stat_nn.functional.mse_loss(model(xs, xn), yt)
            if not stat_torch.isfinite(loss):
                raise FloatingPointError("LSTM: non-finite training loss.")
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu()) * len(yt)
            count += len(yt)
        trace.append({"epoch": epoch + 1, "training_MSE_standardized_level": total / count})
    model.eval()
    return model, stat_pd.DataFrame(trace)


def stat_train_lgbm(tab, target, cfg):
    seed = cfg["seed"]
    params = dict(objective="regression", metric="rmse", learning_rate=cfg["lgbm_learning_rate"],
                  num_leaves=cfg["lgbm_num_leaves"], min_data_in_leaf=cfg["lgbm_min_data_in_leaf"],
                  feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, verbosity=-1,
                  seed=seed, feature_fraction_seed=seed, bagging_seed=seed, data_random_seed=seed,
                  deterministic=True, force_col_wise=True, num_threads=cfg["num_threads"])
    return stat_lgb.train(params, stat_lgb.Dataset(tab, label=target), num_boost_round=cfg["lgbm_rounds"])


@contextmanager
def stat_third_party_output(log_path):
    """Capture the training library output in a separate log."""
    with Path(log_path).open("a", encoding="utf-8") as handle:
        with redirect_stdout(handle), redirect_stderr(handle):
            yield


def stat_train_tft(frame, country, cfg, library_log):
    if not stat_STACK:
        raise ImportError("TFT is required, but pytorch-forecasting has not been loaded.")
    stat_seed_all(cfg["seed"], cfg["num_threads"])
    f = frame.copy().reset_index(drop=True)
    f["time_idx"] = stat_np.arange(len(f), dtype=int)
    f["group"] = str(country)
    f["target"] = f.y_mw.astype(float)
    training = stat_STACK["dataset"](
        f, time_idx="time_idx", target="target", group_ids=["group"],
        min_encoder_length=cfg["lookback"], max_encoder_length=cfg["lookback"],
        min_prediction_length=1, max_prediction_length=1,
        time_varying_known_reals=["time_idx"] + stat_FEATURES,
        time_varying_unknown_reals=["target"],
        target_normalizer=stat_STACK["normalizer"](groups=["group"]),
        add_relative_time_idx=True, add_target_scales=True, add_encoder_length=True,
    )
    loader = training.to_dataloader(train=True, batch_size=cfg["tft_batch_size"], num_workers=0)
    tft = stat_STACK["tft"].from_dataset(
        training, learning_rate=cfg["tft_learning_rate"], hidden_size=cfg["tft_hidden"],
        attention_head_size=cfg["tft_attention_heads"], dropout=cfg["tft_dropout"],
        hidden_continuous_size=cfg["tft_hidden_continuous"], output_size=1,
        loss=stat_STACK["loss"](), log_interval=-1, reduce_on_plateau_patience=None,
    )
    # Epoch counts are fixed before evaluation. Weight-calibration and test
    # periods are not used to choose when to stop training.
    def adam_without_scheduler(self):
        return stat_torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
    tft.configure_optimizers = types.MethodType(adam_without_scheduler, tft)
    with stat_third_party_output(library_log):
        trainer = stat_STACK["pl"].Trainer(
            max_epochs=cfg["tft_epochs"], accelerator="gpu" if cfg["device"] == "cuda" else "cpu",
            devices=1, enable_checkpointing=False, logger=False, enable_progress_bar=False,
            enable_model_summary=False, log_every_n_steps=0, deterministic=True,
        )
        trainer.fit(tft, train_dataloaders=loader)
    tft.eval()
    return tft, training


class stat_ModelBundle:
    def __init__(self, country, cfg, frame, ys, xs, ts, lstm, lgbm, tft, tft_dataset, trace, library_log):
        self.country, self.cfg, self.frame = country, cfg, frame
        self.ys, self.xs, self.ts = ys, xs, ts
        self.lstm, self.lgbm, self.tft, self.tft_dataset = lstm, lgbm, tft, tft_dataset
        self.trace, self.library_log = trace, library_log


def stat_fit_triad(frame, country, cfg, library_log):
    f = frame.copy().reset_index(drop=True)
    if len(f) < cfg["lookback"] + 12:
        raise ValueError("Insufficient training data for the three models.")
    ys = stat_StandardScaler().fit(f[["y_mw"]].to_numpy(float))
    xs = stat_StandardScaler().fit(f[stat_FEATURES].to_numpy(float))
    seq, tab, nxt, y = stat_supervised(f.y_mw, f[stat_FEATURES], cfg["lookback"], cfg["tabular_lags"])
    ts = stat_StandardScaler().fit(y.reshape(-1, 1))
    ss = seq.copy()
    ss[:, :, 0] = ys.transform(seq[:, :, 0].reshape(-1, 1)).reshape(seq.shape[:2])
    ss[:, :, 1:] = xs.transform(seq[:, :, 1:].reshape(-1, len(stat_FEATURES))).reshape(seq.shape[0], cfg["lookback"], len(stat_FEATURES))
    nx = xs.transform(nxt).astype(stat_np.float32)
    yn = ts.transform(y.reshape(-1, 1)).ravel().astype(stat_np.float32)
    stat_LOG.info("  %s: LSTM (%d epochs; %d windows)", country, cfg["lstm_epochs"], len(y))
    lstm, trace = stat_train_lstm(ss, nx, yn, cfg)
    stat_LOG.info("  %s: LightGBM (%d trees maximum)", country, cfg["lgbm_rounds"])
    lgbm = stat_train_lgbm(tab, y, cfg)
    stat_LOG.info("  %s: TFT (%d epochs)", country, cfg["tft_epochs"])
    tft, td = stat_train_tft(f, country, cfg, library_log)
    return stat_ModelBundle(country, cfg, f, ys, xs, ts, lstm, lgbm, tft, td, trace, library_log)


def stat_recursive_paths(bundle, future):
    """Roll each learner forward using its own previously predicted levels.
    """
    cfg, hist = bundle.cfg, bundle.frame
    dates = stat_pd.DatetimeIndex(future.ds)
    expected = stat_pd.date_range(hist.ds.max() + stat_pd.offsets.MonthBegin(1), periods=len(future), freq="MS")
    if not dates.equals(expected):
        raise ValueError("Future dates have gaps or overlap the training period.")
    # Exclude the target from future covariates.
    future_x = future[["ds"] + stat_FEATURES].copy()
    xall = stat_np.vstack([hist[stat_FEATURES].to_numpy(float), future_x[stat_FEATURES].to_numpy(float)])
    all_dates = stat_pd.DatetimeIndex(list(hist.ds) + list(dates))
    start, length, lags = len(hist), cfg["lookback"], cfg["tabular_lags"]
    results = {}
    for name in stat_MODELS:
        y = stat_np.r_[hist.y_mw.to_numpy(float), stat_np.zeros(len(future))]
        pred = []
        for t in range(start, len(y)):
            if name == "LSTM":
                seq = stat_np.column_stack([bundle.ys.transform(y[t-length:t].reshape(-1, 1)).ravel(),
                                       bundle.xs.transform(xall[t-length:t])]).astype(stat_np.float32)
                nxt = bundle.xs.transform(xall[t:t+1]).astype(stat_np.float32)
                with stat_torch.inference_mode():
                    z = bundle.lstm(stat_torch.from_numpy(seq[None]).to(cfg["device"]),
                                    stat_torch.from_numpy(nxt).to(cfg["device"])).detach().cpu().numpy()
                value = float(bundle.ts.inverse_transform(z.reshape(-1, 1))[0, 0])
            elif name == "LGBM":
                tab = stat_np.r_[y[t-stat_np.arange(1, lags+1)], xall[t]].astype(stat_np.float32)[None, :]
                value = float(bundle.lgbm.predict(tab)[0])
            else:
                ctx = stat_pd.DataFrame(xall[t-length:t+1], columns=stat_FEATURES)
                ctx["ds"] = all_dates[t-length:t+1]
                ctx["time_idx"] = stat_np.arange(t-length, t+1, dtype=int)
                ctx["group"] = str(bundle.country)
                ctx["target"] = y[t-length:t+1]
                ctx.loc[len(ctx)-1, "target"] = 0.0   # decoder placeholder; not supplied as a known covariate
                ds = stat_STACK["dataset"].from_dataset(bundle.tft_dataset, ctx, predict=True, stop_randomization=True)
                loader = ds.to_dataloader(train=False, batch_size=1, num_workers=0)
                with stat_third_party_output(bundle.library_log):
                    prediction = bundle.tft.predict(loader, trainer_kwargs={
                        "accelerator": "gpu" if cfg["device"] == "cuda" else "cpu", "devices": 1,
                        "logger": False, "enable_progress_bar": False, "enable_model_summary": False})
                value = float(prediction.detach().cpu().numpy().reshape(-1)[0])
            if not stat_np.isfinite(value):
                raise FloatingPointError(f"{bundle.country}/{name}: non-finite forecast at {all_dates[t]}.")
            y[t] = value
            pred.append(value)
        results[name] = stat_np.asarray(pred)
        stat_LOG.info("  %s: %s path complete (%d months)", bundle.country, name, len(pred))
    return results


def stat_save_final_model_parameters(bundle, folder):
    folder.mkdir(parents=True, exist_ok=True)
    bundle.lgbm.save_model(str(folder / "LightGBM.txt"))
    stat_torch.save(bundle.lstm.state_dict(), folder / "LSTM_state_dict.pt")
    stat_torch.save(bundle.tft.state_dict(), folder / "TFT_state_dict.pt")
    stat_write_json({"country": bundle.country, "features": stat_FEATURES, "config": bundle.cfg,
                "scalers": {name: {"mean": s.mean_, "scale": s.scale_} for name, s in
                            (("historical_demand", bundle.ys), ("covariates", bundle.xs), ("target", bundle.ts))},
                "note": "Parameter snapshots, not a separate executable. Data + code + seed reproduce training."},
               folder / "model_metadata.json")
    stat_write_csv(bundle.trace, folder / "LSTM_training_loss.csv")


def stat_rmse(y, p):
    y, p = stat_finite(y, "RMSE actual"), stat_finite(p, "RMSE predicted")
    if y.shape != p.shape or y.size == 0:
        raise ValueError("RMSE requires nonempty arrays with matching shapes.")
    return float(stat_np.sqrt(stat_np.mean((y - p) ** 2)))


def stat_inverse_rmse_weights(pred, cfg):
    hmax = cfg["evaluation_horizon"]
    if pred.empty or not pred.phase.eq("weight_calibration").all():
        raise ValueError("Weights may use only the weight_calibration phase.")
    if pred.country.nunique() != 1:
        raise ValueError("Do not mix countries when calculating weights.")
    for origin, g in pred.groupby("origin"):
        if sorted(g.horizon.tolist()) != list(range(1, hmax+1)):
            raise ValueError("All horizons 1..24 are required at every weight-calibration origin.")
        expected = [stat_month_start(origin) + stat_pd.DateOffset(months=int(h)) for h in g.horizon]
        if not stat_np.array_equal(stat_pd.to_datetime(g.ds).to_numpy(), stat_pd.DatetimeIndex(expected).to_numpy()):
            raise ValueError("Calibration dates do not match the forecast horizon.")
    scores = stat_np.asarray([stat_rmse(pred.y_true, pred[f"pred_{name}"]) for name in stat_MODELS])
    inverse = 1 / (scores + 1e-12)
    w = inverse / inverse.sum()
    weights = dict(zip(stat_MODELS, w))
    rows = [{"country": pred.country.iloc[0], "model": name, "RMSE_full_path_MW": float(score),
             "weight": float(weight), "n_origin_horizon_pairs": len(pred),
             "n_unique_target_months": pred.ds.nunique(), "n_origins": pred.origin.nunique(),
             "target": "DIRECT_LEVEL", "weights_source": "code_inverse_RMSE_full_h1_to_h24"}
            for name, score, weight in zip(stat_MODELS, scores, w)]
    return weights, stat_pd.DataFrame(rows)


def stat_combine_paths(paths, weights):
    if set(paths) != set(stat_MODELS) or set(weights) != set(stat_MODELS):
        raise ValueError("All three models are required.")
    w = stat_finite([weights[n] for n in stat_MODELS], "weights")
    if (w < 0).any() or not stat_np.isclose(w.sum(), 1.0, atol=1e-12):
        raise ValueError("The weights do not form a convex combination.")
    values = stat_np.vstack([stat_finite(paths[n], n) for n in stat_MODELS])
    return w @ values


def stat_ensemble_frame(pred, weights):
    out = pred.copy()
    out["pred_ENS"] = stat_combine_paths({name: out[f"pred_{name}"].to_numpy(float) for name in stat_MODELS}, weights)
    out["residual_MW"] = out.y_true - out.pred_ENS
    return out


def stat_error_correlations(pred):
    rows = []
    for phase, g in pred.groupby("phase"):
        residuals = stat_pd.DataFrame({n: g.y_true.to_numpy() - g[f"pred_{n}"].to_numpy() for n in stat_MODELS})
        corr = residuals.corr()
        for i, first in enumerate(stat_MODELS):
            for second in stat_MODELS[i+1:]:
                rows.append({"country": g.country.iloc[0], "phase": phase, "model_1": first, "model_2": second,
                             "error_correlation": float(corr.loc[first, second]), "n_pairs": len(g),
                             "note": "Descriptive only; overlapping paths are not independent."})
    return stat_pd.DataFrame(rows)


def stat_dependence_diagnostics(z, country, cfg):
    rows_acf, rows_lb = [], []
    for row, values in enumerate(z):
        if stat_np.std(values) <= stat_np.finfo(float).eps:
            coefficients = stat_np.zeros(13)
            tests = stat_pd.DataFrame({"lb_stat": [0.0]*3, "lb_pvalue": [1.0]*3}, index=[3, 6, 12])
        else:
            coefficients = stat_acf(values, nlags=12, fft=False)
            tests = stat_acorr_ljungbox(values, lags=[3, 6, 12], return_df=True)
        for lag in range(1, 13):
            rows_acf.append({"country": country, "origin_row": row, "lag": lag, "acf": float(coefficients[lag])})
        for lag, result in tests.iterrows():
            rows_lb.append({"country": country, "origin_row": row, "lag": int(lag),
                            "lb_stat": float(result.lb_stat), "p_value": float(result.lb_pvalue),
                            "significant_5pct": bool(result.lb_pvalue < 0.05)})
    ad, ld = stat_pd.DataFrame(rows_acf), stat_pd.DataFrame(rows_lb)
    am = ad.groupby("lag").acf.mean()
    max_acf = float(am.loc[am.index <= 6].abs().max())
    fraction = float(ld.significant_5pct.mean())
    dependent = max_acf >= cfg["dependence_acf_threshold"] or fraction >= cfg["dependence_lb_fraction"]
    summary = {"country": country, "mean_acf_lag1": float(am.loc[1]), "max_abs_mean_acf1_6": max_acf,
               "fraction_ljungbox_significant": fraction, "dependence_detected": bool(dependent),
               "IID_excluded": bool(dependent), "tests_not_independent": True}
    return ad, ld, summary


def stat_empirical_crps(samples, actual):
    """Compute empirical CRPS in O(S log S)."""
    s = stat_finite(samples, "CRPS samples")
    y = stat_finite(actual, "CRPS actual")
    if s.ndim != 2 or s.shape[1] != len(y):
        raise ValueError("Invalid array dimensions for CRPS.")
    ordered = stat_np.sort(s, axis=0)
    n = s.shape[0]
    weights = (2 * stat_np.arange(1, n+1) - n - 1)[:, None]
    return stat_np.mean(stat_np.abs(s - y[None, :]), axis=0) - stat_np.sum(weights * ordered, axis=0) / (n*n)


def stat_interval_metrics(actual, q5, q50, q95, cfg):
    y, lo, med, hi = (stat_np.asarray(v, float) for v in (actual, q5, q50, q95))
    if not (stat_np.all(lo <= med) and stat_np.all(med <= hi)):
        raise ValueError("The quantiles cross.")
    alpha = 1 - cfg["nominal_coverage"]
    width = hi - lo
    score = width + (2/alpha) * stat_np.maximum(lo-y, 0) + (2/alpha) * stat_np.maximum(y-hi, 0)
    coverage = float(stat_np.mean((y >= lo) & (y <= hi)))
    out = {"n": len(y), "PICP90_pct": 100*coverage, "MPIW90_MW": float(stat_np.mean(width)),
           "IntervalScore90_MW": float(stat_np.mean(score)),
           "coverage_error_pp": 100*abs(coverage - cfg["nominal_coverage"]),
           "P50_RMSE_MW": stat_rmse(y, med)}
    for q, p in ((0.05, lo), (0.50, med), (0.95, hi)):
        e = y-p
        out[f"pinball_P{int(q*100)}_MW"] = float(stat_np.mean(stat_np.maximum(q*e, (q-1)*e)))
    return out


def stat_probabilistic_metrics(samples, actual, cfg):
    q5, q50, q95 = stat_np.quantile(samples, [0.05, 0.50, 0.95], axis=0)
    out = stat_interval_metrics(actual, q5, q50, q95, cfg)
    out["CRPS_MW"] = float(stat_np.mean(stat_empirical_crps(samples, actual)))
    return out, (q5, q50, q95)


def stat_annualize(dates, samples, deterministic=None):
    dates = stat_pd.DatetimeIndex(dates)
    values = stat_finite(samples, "annual simulations")
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] != len(dates) or len(dates) == 0:
        raise ValueError("Dates and simulations do not match; at least two simulations are required.")
    if dates.has_duplicates or not dates.is_monotonic_increasing or dates.isna().any():
        raise ValueError("Annualization dates must be unique and chronological.")
    if dates.to_period("M").has_duplicates:
        raise ValueError("Annualization requires exactly one peak per month, not hourly data.")
    if deterministic is not None and stat_finite(deterministic, "annual deterministic path").shape != (len(dates),):
        raise ValueError("The deterministic path is misaligned during annualization.")
    rows, peaks = [], []
    for year in sorted(set(dates.year)):
        mask = dates.year == year
        sims = values[:, mask].max(axis=1)    # Take the maximum within each simulation first.
        q5, q50, q95 = stat_np.quantile(sims, [0.05, 0.50, 0.95])  # Calculate quantiles across those maxima.
        rows.append({"year": int(year), "months_in_forecast": int(mask.sum()),
                     "complete_calendar_year": bool(mask.sum() == 12),
                     "P5_MW": float(q5), "P50_MW": float(q50), "P95_MW": float(q95),
                     "mean_MC_MW": float(sims.mean()), "sd_MC_MW": float(sims.std(ddof=1)),
                     "deterministic_max_MW": float(stat_np.asarray(deterministic)[mask].max()) if deterministic is not None else stat_np.nan,
                     "n_simulations": values.shape[0]})
        peaks.append(sims)
    return stat_pd.DataFrame(rows), stat_np.column_stack(peaks)


def stat_driver_backtests(repo, country, protocol, cfg):
    """Evaluate the driver forecasts separately from the demand model."""
    development_end = stat_pd.Timestamp(protocol.loc[protocol.phase.eq("terminal_holdout"), "origin"].iloc[0])
    records = []
    for col in stat_ANNUAL_DRIVERS:
        truth = repo.annual[repo.annual.country.eq(country)].dropna(subset=[col]).set_index("year")[col]
        for year in range(int(truth.index.min()) + 8, development_end.year):
            origin = stat_pd.Timestamp(year, 12, 1)
            known = repo.annual_known(country, col, origin)
            for h in (1, 3, 5):
                target_year = year + h
                if target_year > development_end.year or target_year not in truth.index:
                    continue
                p, info = repo.macro_predict(country, col, origin, [target_year])
                records.append({"country": country, "driver": col, "origin": origin,
                                "target": str(target_year), "horizon": h, "horizon_unit": "years",
                                "actual": float(truth.loc[target_year]), "prediction": float(p[0]),
                                "last_observed_period": int(known.year.max()), "method": info["method"]})
    truth_monthly = repo.monthly[repo.monthly.country.eq(country)].set_index("ds")
    # Drivers share development origins, but each fit uses its own history.
    for origin in sorted(protocol.loc[~protocol.phase.eq("terminal_holdout"), "origin"].unique()):
        origin = stat_pd.Timestamp(origin)
        dates = stat_pd.date_range(origin + stat_pd.offsets.MonthBegin(1), periods=24, freq="MS")
        for col in stat_MONTHLY_DRIVERS:
            if col == "ipi":
                predictions, info = repo.ipi_predict(country, origin, dates)
            else:
                predictions, info = repo.weather_predict(country, col, origin, dates)
            for h, (date, p) in enumerate(zip(dates, predictions), 1):
                if date > development_end or date not in truth_monthly.index:
                    continue
                actual = truth_monthly.loc[date, col]
                if stat_pd.isna(actual):
                    continue
                records.append({"country": country, "driver": col, "origin": origin,
                                "target": str(date.date()), "horizon": h, "horizon_unit": "months",
                                "actual": float(actual), "prediction": float(p),
                                "last_observed_period": str(stat_pd.Timestamp(info["last_observed_month"]).date()),
                                "method": info["method"]})
    detail = stat_pd.DataFrame(records)
    rows = []
    for (driver, unit, method), group in detail.groupby(["driver", "horizon_unit", "method"]):
        selected = [1, 3, 5] if unit == "years" else [1, 3, 6, 12, 24]
        for h in selected:
            sub = group[group.horizon.eq(h)]
            if sub.empty:
                continue
            e = sub.prediction.to_numpy() - sub.actual.to_numpy()
            denominator = float(sub.actual.abs().sum())
            rows.append({"country": country, "driver": driver, "method": method, "horizon_unit": unit,
                         "horizon": h, "n": len(sub), "RMSE_native_units": stat_rmse(sub.actual, sub.prediction),
                         "MAE_native_units": float(stat_np.abs(e).mean()), "Bias_native_units": float(e.mean()),
                         "WAPE_pct": 100*float(stat_np.abs(e).sum())/denominator if denominator else stat_np.nan,
                         "evaluation_status": "development_only_own_history_forecasts"})
    return detail, stat_pd.DataFrame(rows)


def stat_prediction_frame(country, phase, origin, dates, paths, demand):
    frame = stat_pd.DataFrame({"country": country, "phase": phase, "origin": stat_month_start(origin),
                          "ds": stat_pd.DatetimeIndex(dates), "horizon": stat_np.arange(1, len(dates)+1)})
    truth = demand.set_index("ds").y_mw
    frame["y_true"] = [truth.get(stat_pd.Timestamp(t), stat_np.nan) for t in dates]
    for name in stat_MODELS:
        frame[f"pred_{name}"] = paths[name]
    return frame


stat_DETERMINISTIC_FUNCTIONS = [
    "seed_all", "finite", "month_start", "month_end", "months_between",
    "DriverRepository", "seasonal_repetition", "make_protocol", "supervised",
    "new_lstm", "train_lstm", "train_lgbm", "train_tft", "ModelBundle",
    "fit_triad", "recursive_paths", "inverse_rmse_weights", "combine_paths",
]
stat_DETERMINISTIC_CONFIG_KEYS = [
    "forecast_years", "minimum_history_months", "lookback", "tabular_lags",
    "evaluation_horizon", "terminal_holdout_months", "seed", "num_threads", "device",
    "lstm_hidden", "lstm_layers", "lstm_dropout", "lstm_epochs", "lstm_batch_size",
    "lstm_learning_rate", "lgbm_rounds", "lgbm_num_leaves", "lgbm_min_data_in_leaf",
    "lgbm_learning_rate", "tft_epochs", "tft_batch_size", "tft_hidden",
    "tft_attention_heads", "tft_hidden_continuous", "tft_dropout", "tft_learning_rate",
    "annual_reference_lag_years", "annual_release_year_end_offset",
    "monthly_release_lag_months", "macro_trend_years", "climatology_years",
]


def stat_load_probability_stack():
    """Load the statistical postprocessing dependencies."""
    global stat_np, stat_pd, stat_plt, stat_StandardScaler, stat_TheilSenRegressor, stat_ExponentialSmoothing, stat_acf, stat_acorr_ljungbox
    import numpy as _np
    import pandas as _pd
    import matplotlib.pyplot as _plt
    from sklearn.preprocessing import StandardScaler as _SS
    from sklearn.linear_model import TheilSenRegressor as _TS
    from statsmodels.tsa.holtwinters import ExponentialSmoothing as _ES
    from statsmodels.tsa.stattools import acf as _acf
    from statsmodels.stats.diagnostic import acorr_ljungbox as _lb
    stat_np, stat_pd, stat_plt = _np, _pd, _plt
    stat_StandardScaler, stat_TheilSenRegressor, stat_ExponentialSmoothing = _SS, _TS, _ES
    stat_acf, stat_acorr_ljungbox = _acf, _lb
    stat_plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "font.size": 11, "axes.unicode_minus": True})
    return stat_environment_versions()


def stat_require_real_triad():
    """Load the LSTM, LightGBM and TFT training dependencies."""
    global stat_torch, stat_nn, stat_lgb, stat_STACK
    if stat_STACK:
        return
    try:
        import torch as _torch
        import torch.nn as _nn
        import lightgbm as _lgb
        import lightning.pytorch as pl
        from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
        from pytorch_forecasting.data import GroupNormalizer
        from pytorch_forecasting.metrics import RMSE
        stat_torch, stat_nn, stat_lgb = _torch, _nn, _lgb
        stat_STACK = {"pl": pl, "dataset": TimeSeriesDataSet, "tft": TemporalFusionTransformer,
                 "normalizer": GroupNormalizer, "loss": RMSE}
    except Exception as exc:
        raise ImportError(
            "New data or countries require the real LSTM, LightGBM and TFT models. "
            "Install the dependencies in requirements_retraining.txt. "
            "TFT is not replaced, and a two-model ensemble is not allowed. "
            "Import error: " + repr(exc)) from exc


def demand_metadata(path: Path) -> dict[str, str]:
    """Read the five metadata fields at the start of a demand file."""
    values = {}
    with Path(path).open(encoding='utf-8-sig') as stream:
        for line in stream:
            text = line.strip()
            if not text:
                continue
            if not text.startswith('#'):
                break
            key, sep, value = text[1:].partition('=')
            if not sep:
                continue
            key, value = key.strip().lower(), value.strip()
            if key in values:
                raise ValueError(f'{path.name}: duplicate metadata field {key}.')
            values[key] = value
    required = ('country', 'source', 'frequency', 'variable', 'units')
    missing = [key for key in required if not values.get(key)]
    if missing:
        raise ValueError(f'{path.name}: missing metadata: {", ".join(missing)}.')
    if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', values['country']):
        raise ValueError(f'{path.name}: use letters, digits and underscores in country.')
    values['country'] = values['country'].upper()
    if values['frequency'].lower() != 'monthly':
        raise ValueError(f'{path.name}: frequency must be monthly.')
    if values['variable'] != 'monthly_peak_demand':
        raise ValueError(f'{path.name}: variable must be monthly_peak_demand.')
    if values['units'] != 'MW':
        raise ValueError(f'{path.name}: demand must be supplied in MW.')
    return {key: values[key] for key in required}


def demand_file(country: str) -> Path:
    """Find one TXT or CSV input for a country."""
    folder = ROOT / 'data/statistical'
    stem = f'{country.upper()}_monthly_peak_MW'.lower()
    matches = [p for p in folder.iterdir()
               if p.is_file() and p.stem.lower() == stem and p.suffix.lower() in ('.txt', '.csv')]
    if len(matches) != 1:
        raise ValueError(f'{country}: expected one {stem}.txt or .csv in {folder}; found {len(matches)}.')
    return matches[0]


def stat_load_demand(path, country, cfg):
    """Read monthly peaks in MW and check the calendar and metadata."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Missing file: {path}.')
    meta = demand_metadata(path)
    if meta['country'] != str(country).upper():
        raise ValueError(f'{path.name}: country metadata does not match {country}.')
    frame = pd.read_csv(path, sep=',' if path.suffix.lower() == '.csv' else '\t',
                        comment='#', encoding='utf-8-sig')
    if not {'ds', 'y_mw'}.issubset(frame.columns):
        raise ValueError(f'{path.name}: expected ds,y_mw (CSV) or ds TAB y_mw (TXT).')
    frame = frame[['ds', 'y_mw']].copy()
    frame['ds'] = pd.to_datetime(frame['ds'], errors='raise').dt.to_period('M').dt.to_timestamp()
    frame['y_mw'] = pd.to_numeric(frame['y_mw'], errors='raise')
    frame = frame.sort_values('ds').reset_index(drop=True)
    if frame.empty or frame['ds'].duplicated().any() or frame['ds'].isna().any():
        raise ValueError(f'{country}: empty series, missing dates or duplicate months.')
    expected = pd.date_range(frame['ds'].min(), frame['ds'].max(), freq='MS')
    if not pd.DatetimeIndex(frame['ds']).equals(expected):
        raise ValueError(f'{country}: missing months. Supply a complete monthly series.')
    values = frame['y_mw'].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError(f'{country}: demand must contain finite, positive MW values.')
    if len(frame) < cfg['minimum_history_months']:
        raise ValueError(f"{country}: {len(frame)} months; {cfg['minimum_history_months']} are required.")
    return frame


def export_observed_inputs(country: str, run: Path, cfg: dict) -> None:
    """Copy the measured series used by the run for later plotting."""
    source = demand_file(country)
    frame = stat_load_demand(source, country, cfg)
    out = run / country.lower()
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / 'observed_monthly_peak_MW.csv', index=False, date_format='%Y-%m-%d')
    jwrite(out / 'observed_input_metadata.json', demand_metadata(source))
    snapshot = run / 'observed_inputs' / source.name
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, snapshot)


def stat_target_month_weights(frame):
    """Give each observed month unit total weight; overlapping origins are not replications."""
    counts = frame.groupby("ds").ds.transform("size").to_numpy(float)
    if len(counts) == 0:
        raise ValueError("No target months are available for weighting.")
    return 1.0 / counts


def stat_available_residuals(frame, origin):
    """Keep only admissible information before calculating errors and moments."""
    allowed = ("weight_calibration", "probability_tuning")
    bad = set(frame.phase.unique()) - set(allowed)
    if bad:
        raise ValueError("The residual bank cannot contain holdout or future phases: " + str(sorted(bad)))
    cutoff = stat_month_start(origin)
    out = frame[(frame.ds <= cutoff) & (frame.origin < stat_out_cutoff(cutoff))].copy()
    if out.empty:
        raise ValueError("The available residual bank is empty.")
    out = out.sort_values(["origin", "horizon"]).reset_index(drop=True)
    stat_finite(out.y_true, "available residual observations", positive=True)
    stat_finite(out.pred_ENS, "available residual ensemble forecasts", positive=True)
    out["log_level_ratio"] = stat_np.log(out.y_true.to_numpy(float) / out.pred_ENS.to_numpy(float))
    out["target_month_weight"] = stat_target_month_weights(out)
    out["residual_as_of"] = cutoff
    assert out.ds.max() <= cutoff and (out.origin < out.ds).all()
    return out


def stat_out_cutoff(date):
    # Interpret the monthly label as its reference month.
    return stat_month_start(date)


def stat_cyclic_month_distance(a, b):
    distance = stat_np.abs(stat_np.asarray(a, float)[:, None] - stat_np.asarray(b, float)[None, :])
    return stat_np.minimum(distance, 12.0 - distance)


class stat_ConditionalErrorModel:
    """Partially pooled moments of log(y/yhat), not a separate demand predictor.

    HORIZON_SCALE: zero location, scale smoothed over the horizon.
    CYCLIC_SCALE: zero location, scale by cyclic month and horizon.
    CYCLIC_LOCATION_SCALE: location shrunk to zero and cyclic/horizon scale.
    Month and horizon enter through the same kernel for every country.
    """
    def __init__(self, bank, family, cfg):
        if family not in cfg["probability_families"]:
            raise ValueError("Unknown residual family.")
        self.family, self.cfg = family, cfg
        self.bank = bank.copy().sort_values(["origin", "horizon"]).reset_index(drop=True)
        self.r = stat_finite(self.bank.log_level_ratio, "log-ratios")
        self.w = stat_target_month_weights(self.bank)
        self.months = self.bank.ds.dt.month.to_numpy(float)
        self.horizons = self.bank.horizon.to_numpy(float)
        self.unique_months = int(self.bank.ds.nunique())
        self.prior = float(cfg["probability_prior_months"])
        mu_train = self.location(self.bank.ds, self.horizons)
        self.squared = (self.r - mu_train) ** 2
        self.pooled_variance = float(stat_np.average(self.squared, weights=self.w))
        self.degenerate = self.pooled_variance == 0.0
        sigma_train = self.scale(self.bank.ds, self.horizons)
        z = stat_np.divide(self.r - mu_train, sigma_train, out=stat_np.zeros_like(self.r), where=sigma_train > 0)
        self.z_center = float(stat_np.average(z, weights=self.w))
        self.z = stat_finite(z - self.z_center, "standardized residuals")
        self.bank["location_log"] = mu_train
        self.bank["sigma_log"] = sigma_train
        self.bank["z"] = self.z
        self.bank["target_month_weight"] = self.w
        self.block_cache = {}

    def kernel(self, dates, horizons):
        dates = stat_pd.DatetimeIndex(dates)
        h = stat_finite(horizons, "residual horizons", positive=True)
        if len(h) != len(dates) or not stat_np.all(h == h.astype(int)):
            raise ValueError("Residual dates and horizons are misaligned.")
        # Beyond 24 months, retain the scale of the last evaluated horizon.
        h = stat_np.minimum(h, 24.0)
        dh = (h[:, None] - self.horizons[None, :]) / self.cfg["probability_horizon_bandwidth"]
        k = stat_np.exp(-0.5 * dh**2)
        if self.family != "HORIZON_SCALE":
            dm = stat_cyclic_month_distance(dates.month.to_numpy(), self.months)
            k *= stat_np.exp(-0.5 * (dm/self.cfg["probability_month_bandwidth"])**2)
        return k * self.w[None, :]

    def location(self, dates, horizons):
        if self.family != "CYCLIC_LOCATION_SCALE":
            return stat_np.zeros(len(dates), float)
        k = self.kernel(dates, horizons)
        # Weight the shrinkage by distinct months rather than overlapping rows.
        return (k @ self.r) / (self.prior + k.sum(axis=1))

    def scale(self, dates, horizons):
        if self.degenerate:
            return stat_np.zeros(len(dates), float)
        k = self.kernel(dates, horizons)
        variance = (self.prior*self.pooled_variance + k @ self.squared) / (self.prior+k.sum(axis=1))
        return stat_np.sqrt(stat_finite(variance, "log-error variance", positive=True))

    def moments(self, dates, horizons):
        return self.location(dates, horizons), self.scale(dates, horizons)

    def blocks(self, length):
        length = int(length)
        if length in self.block_cache:
            return self.block_cache[length]
        values, weights, provenance = [], [], []
        for origin, group in self.bank.groupby("origin", sort=True):
            indices = group.index.to_numpy(int)
            hs = group.horizon.to_numpy(int)
            ds = stat_pd.DatetimeIndex(group.ds)
            for start in range(len(group)-length+1):
                ids = indices[start:start+length]
                if not stat_np.array_equal(stat_np.diff(hs[start:start+length]), stat_np.ones(length-1, int)):
                    continue
                window = ds[start:start+length]
                if not window.equals(stat_pd.date_range(window[0], periods=length, freq="MS")):
                    continue
                values.append(self.z[ids])
                weights.append(float(self.w[ids].mean()))
                provenance.append({"source_origin": origin, "source_start": window[0], "source_end": window[-1],
                                   "first_horizon": int(hs[start]), "last_horizon": int(hs[start+length-1]),
                                   "block_length": length})
        if not values:
            raise ValueError(f"No complete blocks of {length} months are available.")
        weights = stat_np.asarray(weights, float)
        weights /= weights.sum()
        result = (stat_np.vstack(values), weights, stat_pd.DataFrame(provenance))
        self.block_cache[length] = result
        return result

    def sample_z(self, n, horizon, block, seed):
        if n < 2 or horizon < 1:
            raise ValueError("Invalid Monte Carlo dimensions.")
        library, probabilities, _ = self.blocks(block)
        rng = stat_np.random.default_rng(seed)
        number = int(stat_np.ceil(horizon / block))
        selected = rng.choice(len(library), size=(n, number), replace=True, p=probabilities)
        # Blocks stay within an origin; concatenate them only after sampling.
        return library[selected].reshape(n, number*block)[:, :horizon]

    def parameters(self):
        return {"family": self.family, "n_rows": len(self.bank), "n_unique_target_months": self.unique_months,
                "n_origins": int(self.bank.origin.nunique()), "first_target": self.bank.ds.min(),
                "last_target": self.bank.ds.max(), "pooled_log_variance": self.pooled_variance,
                "z_weighted_center_removed": self.z_center, "degenerate_distribution": self.degenerate,
                "prior_unique_month_mass": self.prior,
                "month_bandwidth": self.cfg["probability_month_bandwidth"],
                "horizon_bandwidth": self.cfg["probability_horizon_bandwidth"],
                "calendar_conditioning_not_hemisphere_rule": True,
                "horizon_above24_assumption": "h_argument_fixed24_calendar_effect_continues",
                "demand_target": "DIRECT_LEVEL_MW", "error_variable": "log(y_observed/yhat_ensemble)"}


def stat_conditional_draws(model, pred, block, factor, n, seed):
    pred = pred.sort_values("horizon")
    if pred.origin.nunique() != 1 or pred.horizon.duplicated().any():
        raise ValueError("Each simulation must correspond to one forecast origin.")
    h = pred.horizon.to_numpy(int)
    det = stat_finite(pred.pred_ENS, "deterministic demand for Monte Carlo", positive=True)
    mu, sig = model.moments(pred.ds, h)
    noise = model.sample_z(n, int(h.max()), int(block), seed)[:, h-1]
    with stat_np.errstate(over="raise", under="raise", invalid="raise"):
        draws = det[None, :] * stat_np.exp(mu[None, :] + float(factor)*sig[None, :]*noise)
    return stat_finite(draws, "Monte Carlo level log-ratios", positive=True)


def stat_noise_quantile_brackets(noise):
    """Represent interpolated quantiles exactly at every scale.
    With interpolation, exp(quantile(z)) is not exactly quantile(exp(z)).
    Retain both order statistics to avoid that discrepancy.
    """
    ordered = stat_np.sort(noise, axis=0)
    rank = stat_np.asarray([.05, .50, .95]) * (len(noise)-1)
    lo, hi = stat_np.floor(rank).astype(int), stat_np.ceil(rank).astype(int)
    return ordered[lo], ordered[hi], (rank-lo)[:, None]


def stat_conditional_quantiles_from_brackets(det, mu, sigma, brackets, factor):
    low, high, fraction = brackets
    with stat_np.errstate(over="raise", under="raise", invalid="raise"):
        a = stat_np.asarray(det)[None, :] * stat_np.exp(mu[None, :] + factor*sigma[None, :]*low)
        b = stat_np.asarray(det)[None, :] * stat_np.exp(mu[None, :] + factor*sigma[None, :]*high)
    result = (1.0-fraction)*a + fraction*b
    return stat_finite(result, "Monte Carlo quantiles", positive=True)


def stat_inner_probability_folds(calibration_raw, cfg):
    """Time-respecting cross-validation of the postprocessor, excluding probability_tuning and holdout.

    Recompute weights at each split using already completed 24-month paths.
    Evaluate later errors from baseline paths issued before the split.
    Export their original horizons instead of relabeling them h1..24.
    """
    if not calibration_raw.phase.eq("weight_calibration").all():
        raise ValueError("Internal selection accepts only weight_calibration data.")
    raw = calibration_raw.copy().sort_values(["origin", "horizon"]).reset_index(drop=True)
    unique = stat_pd.DatetimeIndex(sorted(raw.ds.unique()))
    count, span = int(cfg["inner_min_target_months"]), int(cfg["inner_validation_months"])
    folds, audits = [], []
    for position in range(count-1, len(unique)-span, span):
        cutoff, last = unique[position], unique[position+span]
        complete = [origin for origin, group in raw.groupby("origin")
                    if group.ds.max() <= cutoff and len(group) == 24 and set(group.horizon) == set(range(1, 25))]
        if not complete:
            continue
        weights, wt = stat_inverse_rmse_weights(raw[raw.origin.isin(complete)], cfg)
        rebased = stat_ensemble_frame(raw, weights)
        train = stat_available_residuals(rebased, cutoff)
        valid = rebased[(rebased.ds > cutoff) & (rebased.ds <= last)].copy()
        if valid.empty or train.ds.nunique() < count:
            continue
        # At each split, use only paths issued before the cutoff.
        if not (valid.origin <= cutoff).all():
            raise ValueError("Internal cross-validation contains a forecast not yet issued at the cutoff.")
        info = {"fold": len(folds)+1, "calibration_as_of": cutoff, "validation_end": last,
                "fit_target_max": train.ds.max(), "validation_target_min": valid.ds.min(),
                "weight_fit_target_max": raw[raw.origin.isin(complete)].ds.max(),
                "weight_fit_complete_paths": len(complete), "train_rows": len(train),
                "train_unique_months": int(train.ds.nunique()), "validation_rows": len(valid),
                "validation_unique_months": int(valid.ds.nunique()),
                "validation_base_horizon_min": int(valid.horizon.min()),
                "validation_base_horizon_max": int(valid.horizon.max()),
                "validation_base_origins": int(valid.origin.nunique()),
                "scope": "residual_postprocessor_forward_validation_of_already_issued_base_paths",
                "not_new_end_to_end_24month_validation": True,
                "train_validation_target_overlap": bool(set(train.ds) & set(valid.ds))}
        for model in stat_MODELS:
            info[f"weight_{model}"] = weights[model]
        assert info["fit_target_max"] <= cutoff < info["validation_target_min"]
        folds.append((train, valid, info))
        audits.append(info)
    if not folds:
        raise ValueError("No admissible internal time split exists; the holdout will not be used for calibration. "
                         "Check the coverage of the development paths.")
    return folds, stat_pd.DataFrame(audits)


def stat_weighted_selection_metrics(actual, det, qs, dates):
    lo, med, hi = qs
    y, p = stat_finite(actual, "selection observations", positive=True), stat_finite(det, "selection deterministic forecasts", positive=True)
    if not ((lo <= med) & (med <= hi)).all():
        raise ValueError("The quantiles cross during selection.")
    temp = stat_pd.DataFrame({"ds": stat_pd.to_datetime(dates)})
    w = stat_target_month_weights(temp)
    w /= w.sum()
    below, above = y < lo, y > hi
    score = (hi-lo) + 20*stat_np.maximum(lo-y, 0) + 20*stat_np.maximum(y-hi, 0)
    coverage = float(stat_np.sum(w*(~below & ~above)))
    return {"weighted_PICP90_pct": 100*coverage,
            "coverage_error_pp": 100*abs(coverage-.90),
            "weighted_lower_miss_pct": 100*float(stat_np.sum(w*below)),
            "weighted_upper_miss_pct": 100*float(stat_np.sum(w*above)),
            "tail_imbalance_pp": 100*abs(float(stat_np.sum(w*below)-stat_np.sum(w*above))),
            "normalized_IntervalScore90": float(stat_np.sum(w*score/p)),
            "weighted_IntervalScore90_MW": float(stat_np.sum(w*score)),
            "weighted_MPIW90_MW": float(stat_np.sum(w*(hi-lo))),
            "n_pairs": len(y), "n_unique_target_months": temp.ds.nunique()}


def stat_choose_probability_candidate(grid, cfg, family=None):
    allowed = grid[grid.valid_candidate].copy()
    if family is not None:
        allowed = allowed[allowed.family.eq(family)]
    if allowed.empty:
        raise ValueError("No valid probabilistic candidates remain.")
    inside = allowed[allowed.coverage_error_pp <= 100*cfg["coverage_tolerance"] + 1e-10]
    feasible = len(inside) > 0
    if feasible:
        pool = inside
    else:
        distance = allowed.coverage_error_pp.min()
        pool = allowed[allowed.coverage_error_pp <= distance+1e-10]
    best = float(pool.normalized_IntervalScore90.min())
    close = pool[pool.normalized_IntervalScore90 <= best*(1+cfg["parsimony_relative_tolerance"])+1e-14]
    # Prefer the simpler option when its score is within 1% of the best admissible score.
    winner = close.sort_values(["family_complexity", "normalized_IntervalScore90", "tail_imbalance_pp",
                                "block_length", "scale_distance_from_one", "scale"]).iloc[0]
    result = winner.to_dict()
    result["coverage_tolerance_achieved_in_inner_cv"] = feasible
    result["selection_basis"] = "inner forward CV; weighted coverage then normalized interval score with 1pct parsimony"
    result["selection_uses_probability_audit"] = False
    result["selection_uses_terminal_holdout"] = False
    return result


def stat_select_probability_layer(calibration_raw, cfg, seed, folder):
    folds, audit = stat_inner_probability_folds(calibration_raw, cfg)
    stat_write_csv(audit, folder / "inner_probability_calendar.csv")
    weights, _ = stat_inverse_rmse_weights(calibration_raw, cfg)
    base = stat_ensemble_frame(calibration_raw, weights)
    bank = stat_available_residuals(base, base.ds.max())
    global_error = stat_ConditionalErrorModel(bank, "HORIZON_SCALE", cfg)
    z = global_error.bank.pivot(index="origin", columns="horizon", values="z").sort_index().to_numpy()
    ad, ld, dependence = stat_dependence_diagnostics(z, calibration_raw.country.iloc[0], cfg)
    stat_write_csv(ad, folder / "selection_residual_acf.csv")
    stat_write_csv(ld, folder / "selection_residual_ljungbox.csv")
    stat_write_json(dependence, folder / "selection_dependence_diagnostics.json")
    blocks = [int(b) for b in cfg["mc_blocks"] if b > 1 or not dependence["dependence_detected"]]
    scales = stat_np.round(stat_np.arange(cfg["mc_scale_min"], cfg["mc_scale_max"]+cfg["mc_scale_step"]*.5,
                                cfg["mc_scale_step"]), 6)
    records = []
    actual = stat_np.concatenate([valid.y_true.to_numpy(float) for _, valid, _ in folds])
    det = stat_np.concatenate([valid.pred_ENS.to_numpy(float) for _, valid, _ in folds])
    dates = stat_pd.DatetimeIndex(stat_np.concatenate([valid.ds.to_numpy() for _, valid, _ in folds]))
    for complexity, family in enumerate(cfg["probability_families"]):
        fitted = [stat_ConditionalErrorModel(train, family, cfg) for train, _, _ in folds]
        for block in blocks:
            pieces = []
            error = None
            try:
                for fold_index, (model, (_, valid, _)) in enumerate(zip(fitted, folds)):
                    # Preserve the original date order when combining groups.
                    valid = valid.reset_index(drop=True)
                    groups = []
                    for group_index, (_, group) in enumerate(valid.groupby("origin", sort=True)):
                        h = group.horizon.to_numpy(int)
                        noise = model.sample_z(cfg["mc_tune_simulations"], int(h.max()), block,
                                               seed + 10000*fold_index + 100*group_index + 17*block)[:, h-1]
                        mu, sigma = model.moments(group.ds, h)
                        groups.append((group.index.to_numpy(int), group.pred_ENS.to_numpy(float), mu, sigma,
                                       stat_noise_quantile_brackets(noise)))
                    pieces.append((len(valid), groups))
            except (ValueError, FloatingPointError) as exc:
                error = str(exc)
            for scale in scales:
                record = {"family": family, "family_complexity": complexity, "block_length": block,
                          "scale": float(scale), "scale_distance_from_one": abs(float(scale)-1),
                          "n_inner_folds": len(folds), "n_inner_unique_months": len(set(dates)),
                          "valid_candidate": error is None, "invalid_reason": error or ""}
                if error is None:
                    try:
                        quantiles = []
                        for count, groups in pieces:
                            q = stat_np.empty((3, count), float)
                            for index, point, mu, sigma, brackets in groups:
                                q[:, index] = stat_conditional_quantiles_from_brackets(point, mu, sigma, brackets, float(scale))
                            quantiles.append(q)
                        record.update(stat_weighted_selection_metrics(actual, det, stat_np.concatenate(quantiles, axis=1), dates))
                    except (ValueError, FloatingPointError) as exc:
                        record.update(valid_candidate=False, invalid_reason=str(exc))
                if not record["valid_candidate"]:
                    record.update(normalized_IntervalScore90=stat_np.inf, coverage_error_pp=stat_np.inf,
                                  tail_imbalance_pp=stat_np.inf)
                records.append(record)
    grid = stat_pd.DataFrame(records)
    selected = stat_choose_probability_candidate(grid, cfg)
    global_reference = stat_choose_probability_candidate(grid, cfg, family="HORIZON_SCALE")
    for label, winner in [("selected", selected), ("horizon_reference", global_reference)]:
        grid[label] = (grid.family.eq(winner["family"]) & grid.block_length.eq(winner["block_length"]) &
                       stat_np.isclose(grid.scale, winner["scale"], atol=1e-12))
    stat_write_csv(grid, folder / "probability_selection_grid.csv")
    decision = {"selected": selected, "horizon_reference": global_reference,
                "frozen_at": calibration_raw.ds.max(), "inner_folds": len(folds),
                "inner_target_months": len(set(dates)), "initial_dependence": dependence,
                "small_inner_validation_warning": len(set(dates)) < 24,
                "initial_bank_weights_fit_on_same_development_archive": True,
                "scope": "probability hyperparameters frozen before all prequential audit origins"}
    stat_write_json(decision, folder / "probability_selection_frozen.json")
    return selected, global_reference, decision

def stat_probability_records(pred, samples, cfg, variant, role):
    detail = pred.copy().sort_values("horizon").reset_index(drop=True)
    score, quantiles = stat_probabilistic_metrics(samples, detail.y_true.to_numpy(float), cfg)
    for name, values in zip(("P5_MW", "P50_MW", "P95_MW"), quantiles):
        detail[name] = values
    detail["CRPS_MW"] = stat_empirical_crps(samples, detail.y_true.to_numpy(float))
    detail["variant"] = variant
    detail["evaluation_role"] = role
    detail["below_P5"] = detail.y_true < detail.P5_MW
    detail["above_P95"] = detail.y_true > detail.P95_MW
    detail["covered"] = ~detail.below_P5 & ~detail.above_P95
    score.update(country=detail.country.iloc[0], phase=detail.phase.iloc[0], variant=variant,
                 origin=detail.origin.iloc[0], evaluation_role=role,
                 below_P5_count=int(detail.below_P5.sum()), above_P95_count=int(detail.above_P95.sum()),
                 n_unique_target_months=int(detail.ds.nunique()), n_origins=1)
    return detail, score


def stat_export_probability_state(model, selected, origin, directory):
    directory.mkdir(parents=True, exist_ok=True)
    stat_write_json({"as_of_origin": stat_month_start(origin), "selected_hyperparameters": selected,
                "fitted_moments": model.parameters()}, directory / "state.json")
    stat_write_csv(model.bank, directory / "available_residual_bank.csv")
    _, probabilities, provenance = model.blocks(int(selected["block_length"]))
    provenance = provenance.copy()
    provenance["sampling_probability"] = probabilities
    stat_write_csv(provenance, directory / "block_library_provenance.csv")
    # Export conditional error moments by month and horizon.
    dates = stat_pd.DatetimeIndex([stat_pd.Timestamp(2000, m, 1) for m in range(1, 13) for _ in range(24)])
    horizons = stat_np.tile(stat_np.arange(1, 25), 12)
    mu, sig = model.moments(dates, horizons)
    stat_write_csv(stat_pd.DataFrame({"calendar_month": dates.month, "base_horizon": horizons,
                           "location_log": mu, "sigma_log": sig,
                           "multiplicative_location_exp_mu": stat_np.exp(mu)}), directory / "calendar_horizon_moments.csv")


def stat_annual_validation(dates, draws, detail, variant):
    annual, simulations = stat_annualize(dates, draws, detail.pred_ENS.to_numpy())
    actual = detail.groupby(detail.ds.dt.year).y_true.max()
    annual["observed_peak_MW"] = annual.year.map(actual)
    annual["covered_P5_P95"] = annual.observed_peak_MW.between(annual.P5_MW, annual.P95_MW)
    annual["P50_error_MW"] = annual.P50_MW - annual.observed_peak_MW
    annual["CRPS_MW"] = stat_empirical_crps(simulations, annual.observed_peak_MW.to_numpy())
    annual["variant"] = variant
    annual["diagnostic_only_few_years"] = True
    annual["partial_year_not_a_full_annual_peak"] = ~annual.complete_calendar_year
    return annual




# Physical accounting and conditional scenarios.

def jload(p: Path) -> Any:
    return json.loads(p.read_text(encoding='utf-8'))


def jwrite(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str, allow_nan=False), encoding='utf-8')


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def check(name: str, condition: Any, detail: Any = '') -> None:
    ok = bool(condition)
    CHECKS.append({'test': name, 'passed': ok, 'detail': str(detail)})
    if not ok:
        raise AssertionError(f'{name}: {detail}')


def annualize(samples: np.ndarray, det: np.ndarray, dates: pd.DatetimeIndex,
              country: str, scenario: str) -> pd.DataFrame:
    if samples.ndim != 2 or samples.shape[1] != len(dates):
        raise ValueError('Sample-date dimensions disagree')
    rows = []
    for i, y in enumerate(sorted(set(dates.year)), 1):
        ix = np.flatnonzero(dates.year == y)
        if len(ix) != 12:
            raise ValueError(f'Incomplete calendar year {y}')
        a = samples[:, ix].max(axis=1)  # Each sample contributes its annual maximum before quantiles are calculated.
        q = np.quantile(a, [.05, .5, .95])
        rows.append(dict(year=int(y), study_year=i, country=country.upper(), scenario=scenario,
                         P5_MW=float(q[0]), P50_MW=float(q[1]), P95_MW=float(q[2]),
                         deterministic_max_MW=float(np.max(det[ix])),
                         n_simulations=samples.shape[0],
                         interpretation='conditional_on_structural_hypotheses_NOT_validated90pct',
                         source_information_date='2026-09-21'))
    return pd.DataFrame(rows)


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y, p = np.asarray(y, float), np.asarray(p, float)
    if len(y) != len(p) or not (np.isfinite(y).all() and np.isfinite(p).all()):
        raise ValueError('Invalid validation arrays')
    e = p-y
    return dict(RMSE_MW=float(np.sqrt(np.mean(e*e))), MAE_MW=float(np.mean(abs(e))),
                WAPE_pct=float(100*np.sum(abs(e))/np.sum(abs(y))), Bias_MW=float(e.mean()))


def stock_path(base: dict[str, float], target_b: float, target_p: float, rate: float,
               years: np.ndarray, mode: str, anchor: int = 2030) -> pd.DataFrame:
    """Annual passenger-car balance by technology, with explicit entries and retirements.

Replacement is a scenario assumption. Reject targets that are incompatible
with the available fleet rather than altering stocks to make them feasible."""
    if not 0 < rate < 1:
        raise ValueError('Replacement rate outside (0,1)')
    if list(years) != list(range(2025, int(years[-1])+1)):
        raise ValueError('Stock path requires consecutive years from 2025')
    fleet = float(base['fleet_total']); b0 = float(base['BEV']); p0 = float(base['PHEV'])
    if min(b0, p0, target_b, target_p) < 0 or max(b0+p0, target_b+target_p) > fleet:
        raise ValueError('Technology targets incompatible with total fleet')
    stock = np.array([b0, p0, fleet-b0-p0], float)
    rows = [dict(year=2025, fleet_total=fleet, BEV=stock[0], PHEV=stock[1], other=stock[2],
                 BEV_entries=0., PHEV_entries=0., other_entries=0., BEV_retired=0.,
                 PHEV_retired=0., other_retired=0., balance_error=0.)]
    shares_anchor = None
    for y in years[1:]:
        retired = rate * stock
        survivors = stock-retired
        births = float(retired.sum())
        if y <= anchor:
            t = (y-2025)/(anchor-2025)
            wanted = np.array([b0+(target_b-b0)*t, p0+(target_p-p0)*t, 0.])
            wanted[2] = fleet-wanted[0]-wanted[1]
            entries = wanted-survivors
            if entries.min() < -EPS or entries.sum() > births+EPS:
                raise ValueError(f'Infeasible stock target at {y}: required entries={entries}, replacements={births}')
            if y == anchor:
                shares_anchor = entries / births
        else:
            if mode == 'hold_stock':
                entries = retired
            elif mode in ('hold_entry_share', 'all_bev_entries_2035'):
                if shares_anchor is None:
                    raise ValueError('No valid anchor entry shares')
                shares = shares_anchor.copy()
                if mode == 'all_bev_entries_2035':
                    t = min((y-anchor)/5., 1.)  # date of the adoption assumption, not a power limit
                    shares = (1-t)*shares + t*np.array([1., 0., 0.])
                entries = births*shares
            else:
                raise ValueError(f'Unknown stock mode {mode}')
        new = survivors+entries
        err = float(np.max(abs(new-stock-entries+retired)))
        if new.min() < -EPS or abs(new.sum()-fleet) > EPS or entries.min() < -EPS:
            raise ValueError('Stock-flow conservation violated')
        rows.append(dict(year=int(y), fleet_total=fleet, BEV=float(new[0]), PHEV=float(new[1]), other=float(new[2]),
                         BEV_entries=float(entries[0]), PHEV_entries=float(entries[1]), other_entries=float(entries[2]),
                         BEV_retired=float(retired[0]), PHEV_retired=float(retired[1]), other_retired=float(retired[2]),
                         balance_error=err))
        stock = new
    return pd.DataFrame(rows)


def month_mid_time(dates: pd.DatetimeIndex) -> np.ndarray:
    """Decimal mid-month date used to interpolate year-end stocks."""
    return dates.year.to_numpy(float)-1+(dates.month.to_numpy(float)-.5)/12


def monthly_stock(path: pd.DataFrame, dates: pd.DatetimeIndex, col: str) -> np.ndarray:
    return np.interp(month_mid_time(dates), path.year, path[col])


def daily_shape_weights(dates: pd.DatetimeIndex, shape: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hours = dates.days_in_month.to_numpy(float)*24
    if len(shape) != 12 or np.any(shape < 0) or shape.sum() <= 0:
        raise ValueError('Invalid 12-month daily intensity shape')
    weights = np.zeros(len(dates))
    for y in sorted(set(dates.year)):
        ix = np.flatnonzero(dates.year == y)
        weights[ix] = shape[dates.month[ix]-1]*hours[ix]
        weights[ix] /= weights[ix].sum()
    return weights, hours


def hdd_weights(dates: pd.DatetimeIndex, exog: pd.DataFrame) -> np.ndarray:
    x = exog.copy(); x['ds'] = pd.to_datetime(x['ds'])
    hdd = x.set_index('ds').reindex(dates)['hdd'].to_numpy(float)
    if not np.isfinite(hdd).all() or np.any(hdd < 0):
        raise ValueError('Missing or negative HDD')
    w = np.zeros(len(dates))
    for y in sorted(set(dates.year)):
        ix = np.flatnonzero(dates.year == y)
        if hdd[ix].sum() <= 0:
            raise ValueError('No heating exposure: an HP heating scenario needs a supported seasonal allocation')
        w[ix] = hdd[ix]/hdd[ix].sum()
    return w


def cohort_utilization(age: np.ndarray, p: dict[str, Any]) -> np.ndarray:
    if p['data_center_ramp_years'] <= 2.5:
        raise ValueError('Data-center maturity must exceed 2.5years')
    # Utilization is a fraction of capacity, not a demand-clipping rule.
    u = np.interp(age, [0., 2.5, p['data_center_ramp_years']],
                  [p['data_center_initial_utilization'],p['data_center_utilization_age2_5'],p['data_center_mature_utilization']])
    u = np.where(age < 0, 0., u)
    if np.min(u) < 0 or np.max(u) > 1:
        raise ValueError('Capacity utilization outside [0,1]')
    return u


def project_components(country: str, dates: pd.DatetimeIndex, projects: pd.DataFrame,
                       p: dict[str, Any], base_table: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Convert capacity, realization and utilization into coincident load."""
    comp = {}; ledger = []; energies = []
    d = projects[projects.country.eq(country)].copy()
    for group, g in d.groupby('exclusive_group', sort=True):
        g = g.sort_values('year')
        if g['year'].duplicated().any() or g.component.nunique() != 1 or g.capacity_basis.nunique() != 1:
            raise ValueError(f'Duplicate/inconsistent capacity group {group}')
        r = g.iloc[0]; name = str(r.component)
        if name == 'residential' and not p['enable_residential_projects']:
            ledger.append(dict(group=group, component=name, status='excluded_primary_potential_overlap', source_id=r.source_id))
            continue
        if r.capacity_basis == 'already_realized_scenario_capacity' and not np.allclose(g.realization,1):
            raise ValueError('A realization factor was applied twice to an already-realized capacity')
        base = 0.
        if name == 'electrolyzers':
            q = base_table[base_table.country.eq(country)&base_table.component.eq(name)]
            if len(q)!=1: raise ValueError('Exactly one stated electrolyzer baseline required')
            base = float(q.base_capacity_GW.iloc[0])
            if country in p.get('h2_base_by_country_GW',{}):
                base = float(p['h2_base_by_country_GW'][country])
        capacities = g.capacity_GW.to_numpy(float)
        if base < 0 or np.any(capacities < base):
            raise ValueError('Incompatible baseline/target capacity')
        # Subtract existing capacity before applying the realization assumption.
        delivered = (capacities-base)*g.realization.to_numpy(float)*p['project_multiplier']
        delivered *= 1-p['project_already_embedded_fraction']
        if not 0 <= p['project_already_embedded_fraction'] <= 1:
            raise ValueError('Overlap fraction outside [0,1]')
        delay = p.get('project_delay_years',0)
        knots = np.r_[2025+delay, g.year.to_numpy(float)+delay]
        vals = np.r_[0., delivered]
        if np.any(np.diff(knots)<=0) or np.any(np.diff(vals)<-EPS):
            raise ValueError('Project cumulative capacity must have unique increasing knots')
        if r.ramp == 'cohort':
            # Distribute connections across the months in each declared interval.
            end_coords = dates.year.to_numpy(float)-1+dates.month.to_numpy(float)/12
            full = np.interp(end_coords,knots,vals,left=0.,right=vals[-1])
            flow = np.diff(np.r_[0.,full])
            if np.min(flow)<-EPS:raise ValueError('Negative commissioning flow')
            tmid = month_mid_time(dates)
            load = np.zeros(len(dates)); energy = np.zeros(len(dates))
            for j, add_gw in enumerate(flow):
                if add_gw <= 0: continue
                age = tmid-tmid[j]
                util = cohort_utilization(age,p)
                load += add_gw*1000*util*float(r.peak_fraction)
                energy += add_gw*1000*util*dates.days_in_month.to_numpy(float)*24
                ledger.append(dict(group=group, component=name, commissioning_month=str(dates[j].date()),
                                   delivered_GW=float(add_gw), capacity_basis=r.capacity_basis,
                                   source_id=r.source_id, status='assumed_monthly_commissioning_schedule'))
        else:
            cap = np.interp(month_mid_time(dates),knots,vals,left=0.,right=vals[-1])
            load = cap*1000*float(r.peak_fraction)
            energy = cap*1000*float(r.annual_utilization)*dates.days_in_month.to_numpy(float)*24
            for y, capx in zip(knots[1:],vals[1:]):
                ledger.append(dict(group=group, component=name, target_year=int(y), delivered_GW=float(capx),
                                   baseline_subtracted_GW=base, capacity_basis=r.capacity_basis, realization=float(r.realization),
                                   source_id=r.source_id, status='scenario_capacity_not_observed_peak'))
        comp[name] = load
        for dte,e in zip(dates,energy):
            energies.append(dict(ds=dte, component=name, increment_grid_MWh=float(e),
                                 energy_status='capacity_utilization_scenario_not_total_system_energy'))
    return comp,pd.DataFrame(ledger),pd.DataFrame(energies)


def physical_case(country: str, cconf: dict, config: dict, scenario: dict,
                  stocks: pd.DataFrame, dates: pd.DatetimeIndex, exog: pd.DataFrame,
                  projects: pd.DataFrame, capbases: pd.DataFrame) -> dict[str, Any]:
    p = copy.deepcopy(config['parameters']); p.update(scenario['overrides'])
    # Total fleet, BEV and PHEV stocks use the same passenger-car scope.
    b = stocks[stocks.country.eq(country)&stocks.year.eq(2025)]
    if len(b)!=1: raise ValueError(f'Unique2025 technology reference required: {country}')
    base = b.iloc[0][['fleet_total','BEV','PHEV']].to_dict()
    if country in p.get('alternative_base_by_country',{}):
        base = dict(p['alternative_base_by_country'][country])
    if cconf['policy_target_kind']=='allocated_passenger_plugin_count':
        target_total=cconf['aggregate_ev_target']*cconf['passenger_allocation']
        fraction=p.get('target_bev_fraction',cconf['target_bev_fraction'])
        target_b=target_total*fraction;target_p=target_total*(1-fraction)
    elif cconf['policy_target_kind']=='passenger_bev_share':
        target_b=cconf['target_bev_share']*base['fleet_total']
        target_p=base['PHEV']
    else: raise ValueError('Unknown target definition')
    if p['fleet_growth'] != 0:
        raise ValueError('Current protocol assumes fixed fleet; do not silently introduce a growth rate')
    path=stock_path(base,target_b,target_p,p['turnover_rate'],np.arange(2025,2046),scenario['stock_mode'],cconf['target_year'])
    check(f'{country}/{scenario["id"]}/stock_balance',path.balance_error.max()<EPS,path.balance_error.max())
    w,hours=daily_shape_weights(dates,np.array(config['ev_daily_energy_shape'],float))
    comp={};energies=[]
    for tech,param in [('BEV','bev_grid_MWh_per_year'),('PHEV','phev_grid_MWh_per_year')]:
        n=monthly_stock(path,dates,tech)
        delta=n-base[tech]
        grid=delta*p[param]*w
        comp['EV_'+tech]=grid/hours*p['ev_critical_to_month_mean']
        energies.extend(dict(ds=d,component='EV_'+tech,increment_grid_MWh=float(e),
                             energy_status='grid_energy_including_charging_losses_once') for d,e in zip(dates,grid))
    hp_ledger=pd.DataFrame()
    if cconf['heat_pumps_enabled']:
        pts=np.asarray(cconf['hp_stock_knots'],float)
        base_hp=float(np.interp(2025,pts[:,0],pts[:,1]))
        if 'hp_base_override' in p: base_hp=float(p['hp_base_override'])
        xp=np.r_[2025.,pts[pts[:,0]>2025,0]];fp=np.r_[base_hp,pts[pts[:,0]>2025,1]]
        n=np.interp(month_mid_time(dates),xp,fp)
        delta=n-base_hp
        if np.any(delta < 0):raise ValueError('Heat pump target below reference; requires explicit retirement ledger')
        j=float(p['hp_resistance_replacement_fraction'])
        if not 0<=j<=1:raise ValueError('Heat replacement fraction outside[0,1]')
        w_heat=hdd_weights(dates,exog)
        Q=delta*p['hp_thermal_MWh_per_dwelling_year']*p['hp_thermal_demand_factor']*w_heat
        if min(p['hp_seasonal_COP'],p['hp_cold_COP'])<=0:raise ValueError('Nonpositive COP')
        backup=p['hp_cold_backup_fraction']
        if not 0<=backup<=1:raise ValueError('Invalid backup heat fraction')
        cold_inverse=(1-backup)/p['hp_cold_COP']+backup
        thermal_peak=Q/hours*p['hp_thermal_peak_to_month_mean']
        comp['HP_electricity_new']=thermal_peak*cold_inverse
        comp['HP_resistance_removed']=-thermal_peak*j
        # Annual energy uses seasonal COP; peak demand uses cold-condition COP.
        newE=Q/p['hp_seasonal_COP'];removedE=-Q*j
        hp_ledger=pd.DataFrame(dict(ds=dates,stock_month_mean=n,base_2025=base_hp,increment_stock=delta,
                                   non_electric_replaced=delta*(1-j),resistance_replaced=delta*j,
                                   useful_heat_MWh=Q,new_HP_grid_MWh=newE,removed_Joule_MWh=removedE,
                                   net_grid_MWh=newE+removedE,new_HP_peak_MW=comp['HP_electricity_new'],
                                   removed_Joule_peak_MW=comp['HP_resistance_removed'],
                                   replacement_fraction_status='scenario_not_observed',base_status=cconf['hp_base_2025_status']))
        for name,arr in [('HP_electricity_new',newE),('HP_resistance_removed',removedE)]:
            energies.extend(dict(ds=d,component=name,increment_grid_MWh=float(e),energy_status='net_heating_accounting') for d,e in zip(dates,arr))
    pc,project_ledger,project_energy=project_components(country,dates,projects,p,capbases)
    comp.update(pc)
    additionality=float(p['additionality_fraction'])
    if not 0<=additionality<=1:raise ValueError('Additionality fraction outside[0,1]')
    frame=pd.DataFrame(comp,index=dates)
    # Retain negative contributions from displaced electric load.
    frame*=additionality
    energy=pd.concat([pd.DataFrame(energies),project_energy],ignore_index=True)
    energy['increment_grid_MWh']*=additionality
    return dict(components=frame,stocks=path,hp_ledger=hp_ledger,project_ledger=project_ledger,
                energy=energy,parameters=p,base=base,target_b=target_b,target_p=target_p)


def frozen_country(country: str, config: dict) -> tuple:
    with np.load(REF/country/'future_monthly_simulations.npz',allow_pickle=False) as z:
        samples=z['samples_MW'].astype(float);det=z['deterministic_MW'].astype(float)
        dates=pd.DatetimeIndex(pd.to_datetime(z['dates'].astype(str)))
    expected=pd.date_range(config['forecast_start'],config['forecast_end'],freq='MS')
    check(f'{country}/future_calendar',dates.equals(expected))
    check(f'{country}/sample_shape',samples.shape==(config['n_simulations'],len(expected)),samples.shape)
    check(f'{country}/base_finite_positive',np.isfinite(samples).all() and (samples>0).all())
    old=pd.read_csv(REF/country/'annual_peak_demand_scenarios.csv')
    recal=annualize(samples,det,dates,country,'STATISTICAL_BASELINE')
    check(f'{country}/annual_quantiles_reproduce_statistical_baseline',
          np.max(abs(recal[['P5_MW','P50_MW','P95_MW']].values-old[['P5_MW','P50_MW','P95_MW']].values))<EPS)
    # Recalculate weights from calibration paths, not the terminal block.
    cal=pd.read_csv(REF/country/'development_recursive_predictions.csv')
    cal=cal[cal.phase.eq('weight_calibration')]
    m=['LSTM','LGBM','TFT'];rm=np.array([np.sqrt(np.mean((cal['pred_'+k]-cal.y_true)**2)) for k in m])
    inv=1/(rm+1e-12);ww=inv/inv.sum()
    oldw=pd.read_csv(REF/country/'ensemble_weights.csv').set_index('model').loc[m,'weight'].values
    check(f'{country}/weights_recomputed_full_path',np.max(abs(ww-oldw))<1e-8,ww.tolist())
    hold=pd.read_csv(REF/country/'holdout_recursive_predictions.csv')
    wrecon=hold[['pred_'+k for k in m]].values@ww
    check(f'{country}/ensemble_reconstruction',np.max(abs(wrecon-hold.pred_ENS))<1e-5)
    exog=pd.read_csv(REF/country/'future_exogenous_features.csv')
    return samples,det,dates,exog,pd.DataFrame(dict(model=m,RMSE_full_path=rm,weight_recomputed=ww))


def stock_driver_diagnostics(country: str, stocks: pd.DataFrame, out: Path, rate: float) -> None:
    """Evaluate the fleet balance against observed history, without future targets."""
    d=stocks[stocks.country.eq(country)].sort_values('year').set_index('year');rows=[]
    for origin in d.index:
        known=d.loc[:origin]
        if len(known)<5:continue
        recent=known[['BEV','PHEV']].iloc[-4:]
        entries=recent.values[1:]-(1-rate)*recent.values[:-1]
        fleet=float(known.fleet_total.iloc[-1]);b=known[['BEV','PHEV']].iloc[-1].values.astype(float)
        annual_entries=entries.mean(axis=0)
        if (annual_entries<0).any() or annual_entries.sum()>rate*fleet:
            continue  # This origin cannot support the declared fixed-fleet balance.
        pred=b.copy()
        for step in range(1,6):
            pred=(1-rate)*pred+annual_entries
            if step not in [1,3,5] or origin+step not in d.index:continue
            for i,t in enumerate(['BEV','PHEV']):
                actual=float(d.loc[origin+step,t]);rows.append(dict(country=country,origin=int(origin),target_year=int(origin+step),horizon=step,technology=t,
                 actual_stock=actual,pred_entries_model=float(pred[i]),pred_persistence=float(b[i]),
                 protocol='latest_vintage_historical_only_not_validation_of_policy_path'))
    x=pd.DataFrame(rows);x.to_csv(out/'stock_driver_backtest_predictions.csv',index=False)
    scores=[]
    if len(x):
        for (tech,h),g in x.groupby(['technology','horizon']):
            for m in ['pred_entries_model','pred_persistence']:
                e=g[m]-g.actual_stock;scores.append(dict(technology=tech,horizon=h,method=m,n=len(g),
                    RMSE_vehicles=float(np.sqrt(np.mean(e*e))),MAE_vehicles=float(abs(e).mean()),
                    structural_policy_validation=False))
    pd.DataFrame(scores).to_csv(out/'stock_driver_backtest_scores.csv',index=False)



def run_country(country: str, cconf: dict, config: dict, scenarios: list,
                stocks: pd.DataFrame, projects: pd.DataFrame, capbases: pd.DataFrame,
                run_dir: Path) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    out=run_dir/country;out.mkdir(exist_ok=True)
    samples,det,dates,exog,weights=frozen_country(country,config)
    weights.to_csv(out/'weights_recomputed_frozen.csv',index=False)
    ann={'STATISTICAL_BASELINE':annualize(samples,det,dates,country,'STATISTICAL_BASELINE')}
    allrows=[];contribrows=[];reference=None;month_ref=None
    for scenario in scenarios:
        tag=scenario['id'];sd=out/tag;sd.mkdir()
        physical=physical_case(country,cconf,config,scenario,stocks,dates,exog,projects,capbases)
        cf=physical['components'];shift=cf.sum(axis=1).values
        ss=samples+shift;central=det+shift
        check(f'{country}/{tag}/finite',np.isfinite(ss).all() and np.isfinite(central).all())
        check(f'{country}/{tag}/positive_total_without_clipping',(ss>0).all() and (central>0).all())
        check(f'{country}/{tag}/additive_identity',np.max(abs((ss-samples)-shift))<EPS)
        if tag=='BASELINE_OVERLAP100':
            check(f'{country}/{tag}/exact_statistical_baseline',np.array_equal(ss,samples))
        a=annualize(ss,central,dates,country,tag);ann[tag]=a;allrows.append(a)
        a.to_csv(sd/'annual_peak_scenarios.csv',index=False)
        q=np.quantile(ss,[.05,.5,.95],axis=0)
        monthly=pd.DataFrame(dict(ds=dates,deterministic_MW=central,P5_MW=q[0],P50_MW=q[1],P95_MW=q[2],
                                  structural_increment_MW=shift,scenario=tag))
        monthly.to_csv(sd/'monthly_peak_scenarios.csv',index=False)
        cf.assign(total_increment_MW=shift).to_csv(sd/'monthly_component_accounting.csv',index_label='ds')
        physical['stocks'].to_csv(sd/'vehicle_stock_flow.csv',index=False)
        physical['energy'].to_csv(sd/'monthly_energy_ledger.csv',index=False)
        if len(physical['hp_ledger']):physical['hp_ledger'].to_csv(sd/'heating_net_ledger.csv',index=False)
        physical['project_ledger'].to_csv(sd/'capacity_cohort_ledger.csv',index=False)
        jwrite(sd/'resolved_assumptions.json',dict(scenario=scenario,parameters=physical['parameters'],
                   target_BEV=physical['target_b'],target_PHEV=physical['target_p'],base=physical['base'],
                   empirical_structural_validation=False,scenario_probabilities=None))
        cr=[];simulation_attribution=[]
        for y in sorted(set(dates.year)):
            ix=np.flatnonzero(dates.year==y);pos=ix[np.argmax(central[ix])]
            row={'year':int(y),'scenario':tag,'peak_month':str(dates[pos].date()),'baseline_at_selected_month_MW':float(det[pos]),
                 'total_deterministic_peak_MW':float(central[pos]),'total_increment_MW':float(shift[pos])}
            for c in cf:row[c+'_increment_MW']=float(cf.iloc[pos][c])
            cr.append(row)
            check(f'{country}/{tag}/{y}/component_sum',abs(row['baseline_at_selected_month_MW']+sum(row[c+'_increment_MW'] for c in cf)-row['total_deterministic_peak_MW'])<EPS)
            # Use each simulation's own peak month, then average the contributions.
            peakidx=ix[ss[:,ix].argmax(axis=1)];which=np.arange(ss.shape[0])
            simrow=dict(year=int(y),mean_total_peak_MW=float(ss[which,peakidx].mean()),
                        mean_base_at_same_peak_MW=float(samples[which,peakidx].mean()))
            for c in cf:simrow['mean_'+c+'_MW']=float(cf[c].values[peakidx].mean())
            err=abs(simrow['mean_total_peak_MW']-simrow['mean_base_at_same_peak_MW']-sum(simrow['mean_'+c+'_MW'] for c in cf))
            check(f'{country}/{tag}/{y}/simulation_attribution',err<EPS,err)
            simulation_attribution.append(simrow)
        cframe=pd.DataFrame(cr);cframe.to_csv(sd/'annual_deterministic_contributions.csv',index=False)
        pd.DataFrame(simulation_attribution).to_csv(sd/'annual_mean_attribution_same_peak.csv',index=False)
        if tag in ['MILESTONES','TURNOVER','FAST_RENEWAL'] or config['save_all_scenario_samples']:
            np.savez_compressed(sd/'future_monthly_simulations.npz',samples_MW=ss,dates=dates.strftime('%Y-%m-%d').to_numpy(dtype='U10'),deterministic_MW=central)
        if tag==config['reference_configuration']:
            reference=(physical,cframe);month_ref=monthly
        contribrows.extend(cr)
    totals=pd.concat(allrows,ignore_index=True)
    totals.to_csv(out/'all_conditional_annual_scenarios.csv',index=False)
    span=totals.groupby('year').agg(P50_min_MW=('P50_MW','min'),P50_max_MW=('P50_MW','max'),
          lowest_conditional_P5_MW=('P5_MW','min'),highest_conditional_P95_MW=('P95_MW','max')).reset_index()
    span['meaning']='span_over_tested_scenarios_NOT_probability_interval_or_complete_bound'
    span.to_csv(out/'structural_sensitivity_span.csv',index=False)
    if reference is None or month_ref is None:raise ValueError('Missing reference scenario')
    # Generate the article figures after both countries have finished.
    if config['run_driver_diagnostics']:stock_driver_diagnostics(country,stocks,out,config['parameters']['turnover_rate'])
    stats=pd.DataFrame()
    if config['run_statistical_benchmarks']:
        logging.info('Retrospective statistical comparisons: %s',country)
        stats=reproduce_benchmark_metrics(country,out,config)
    # Coverage refers to the statistical baseline, not the physical total.
    hm=pd.read_csv(REF/country/'holdout_probabilistic_metrics.csv')
    hm['scope_probability_evaluation']='BASELINE_ONLY_NOT_STRUCTURAL_SCENARIO';hm.to_csv(out/'baseline_holdout_probability_metrics.csv',index=False)
    return totals,span,stats


def unit_tests() -> None:
    """Small accounting tests; their values are never used as national input data."""
    for y,n in [(2027,8760),(2028,8784)]:
        d=pd.date_range(f'{y}-01-01',periods=12,freq='MS');w,h=daily_shape_weights(d,np.ones(12))
        check(f'hours_{y}',h.sum()==n);check(f'energy_{y}',abs(w.sum()-1)<1e-12)
        check(f'constant_power_{y}',np.max(abs((w*1000)/h-1000/n))<1e-12)
    b=dict(fleet_total=1000.,BEV=100.,PHEV=50.)
    for mode in ['hold_stock','hold_entry_share','all_bev_entries_2035']:
        s=stock_path(b,220.,90.,.05,np.arange(2025,2046),mode)
        check(mode+'_sum',np.max(abs(s[['BEV','PHEV','other']].sum(axis=1)-1000))<1e-9)
        check(mode+'_anchor',np.allclose(s.loc[s.year.eq(2030),['BEV','PHEV']].values,[220,90]))
        if mode=='hold_stock':check('hold_stock_const',np.ptp(s[s.year>=2030].BEV)<1e-8)
    try:stock_path(b,950.,45.,.01,np.arange(2025,2046),'hold_entry_share')
    except ValueError:check('infeasible_stock_rejected',True)
    else:check('infeasible_stock_rejected',False)
    check('heating_joule_saving_example',abs(6/3-6-(-4))<1e-12)
    check('net_heating_zero_boundary',abs(1/2-.5)<1e-12)
    check('PHEV_energy_not_BEV',1.3<2.6)
    d=pd.date_range('2026-01-01',periods=12,freq='MS')
    a=np.zeros((3,12));a[0,0]=10;a[1,1]=10;a[2,2]=10
    check('max_before_quantile_counterexample',np.quantile(a.max(axis=1),.5)!=np.max(np.quantile(a,.5,axis=0)))
    tmp=annualize(a+1,np.ones(12),d,'test','test')
    check('annualization_known_case',tmp.P50_MW.iloc[0]==11)
    p=jload(DATA/'configuration.json')['parameters']
    uu=cohort_utilization(np.array([-1,0,2.5,12.5,20]),p)
    check('cohort_not_before_connection',uu[0]==0)
    check('cohort_age_not_calendar',np.allclose(uu[1:],[.05,.2,.8,.8]))
    check('cohort_capacity_respected',uu.max()<=1)
    projects=pd.read_csv(DATA/'project_capacity_knots.csv')
    r=projects[projects.capacity_basis.eq('already_realized_scenario_capacity')]
    check('RTE_realization_not_applied_twice',np.all(r.realization==1))
    c,ledger,e=project_components('france',pd.date_range('2026-01-01',periods=240,freq='MS'),projects,p,pd.read_csv(DATA/'capacity_base_assumptions.csv'))
    check('RTE_industry_fully_delivered',abs(c['industry'][-1]-3400*.65)<1e-8)
    check('RTE_dc_fully_mature',abs(c['data_centers'][-1]-4300*.8)<1e-8)
    # Check the vehicle categories against the original data extract.
    raw=pd.read_csv(DATA/'eurostat_passenger_cars_raw_inherited.csv')
    clean=pd.read_csv(DATA/'passenger_car_stocks.csv')
    for row in clean.itertuples():
        geo='ES' if row.country=='spain' else 'FR';x=raw[raw.geo.eq(geo)&raw.time.eq(row.year)].set_index('mot_nrg').value
        check(f'{row.country}/{row.year}/PHEV_definition',abs(row.PHEV-float(x.ELC_PET_PI+x.ELC_DIE_PI))<1e-8)
        check(f'{row.country}/{row.year}/BEV_definition',abs(row.BEV-float(x.ELC))<1e-8)
    protocol=jload(DATA/'scenario_protocol.json')
    check('no_probability_for_structural_cases',protocol['probabilities'] is None)
    check('no_selection_on_future_peak',not protocol['selection_by_future_peak'])


def package_results(run: Path, destination: Path) -> None:
    temp=destination.with_suffix('.tmp.zip')
    with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(run.rglob('*')):
            if p.is_file():z.write(p,Path(run.name)/p.relative_to(run))
    temp.replace(destination)


# Release reproduction, verification and exports.


def generate_article_figures(run: Path) -> None:
    """Redraw the eleven numerical figures from the current run."""
    DATA = prepare_figure_data(run)
    FIG = run / 'article/Figures'
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','Liberation Serif','DejaVu Serif'],
                         'font.size':11,'axes.labelsize':12,'axes.titlesize':13,'legend.fontsize':9,
                         'xtick.labelsize':10,'ytick.labelsize':10,'savefig.dpi':350,'pdf.fonttype':42})

    def save(fig,name):
        for ext in ('png','pdf'):
            fig.savefig(FIG/f'{name}.{ext}',dpi=350,bbox_inches='tight',pad_inches=.06)
        plt.close(fig)

    def annual_axis(ax,title):
        ax.set(xlim=(2025.5,2045.5),xlabel='Calendar year',ylabel='Annual peak demand [GW]',title=title)
        ax.set_xticks([2026,2030,2035,2040,2045]);ax.grid(True,alpha=.2)

    # Each segment represents a 24-month evaluation path.
    cal=pd.read_csv(DATA/'spain_calendar.csv',parse_dates=['origin','target_start','target_end'])
    fig,ax=plt.subplots(figsize=(9.0,4.7))
    for phase,label,marker in [('weight_calibration','Weight calibration','o'),('probability_tuning','Prequential probability audit','s'),('terminal_holdout','Terminal retrospective evaluation','D')]:
        rows=cal[cal.phase.eq(phase)]
        xs, ys = [], []
        for idx, row in rows.iterrows():
            xs.extend([mdates.date2num(row.origin), mdates.date2num(row.target_start),
                       mdates.date2num(row.target_end), np.nan])
            ys.extend([idx, idx, idx, np.nan])
        ax.plot(xs, ys, linewidth=2.3, marker=marker, markevery=list(range(0,len(xs),4)),
                markersize=5, label=label)
    ax.set_yticks(np.arange(len(cal)),cal.origin.dt.strftime('%Y-%m'))
    ax.invert_yaxis();ax.set(ylabel='Forecast origin',xlabel='Target month',title='Chronological evaluation of the statistical baseline')
    ax.xaxis.set_major_locator(mdates.YearLocator());ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.grid(axis='x',alpha=.2);ax.legend(loc='upper center',bbox_to_anchor=(.5,-.15),ncol=3,frameon=False)
    fig.tight_layout();save(fig,'Fig1_validation_calendar')

    for c,name,num in [('spain','Spain',2),('france','France',3)]:
        h=pd.read_csv(DATA/f'{c}_holdout.csv',parse_dates=['ds'])
        fig,ax=plt.subplots(figsize=(9.0,4.2))
        ax.plot(h.ds,h.y_true/1000,linewidth=2.1,marker='o',markersize=3.5,label='Observed')
        for col,label,ls in [('pred_LSTM','LSTM',':'),('pred_LGBM','LightGBM','--'),('pred_TFT','TFT','-.'),('pred_ENS','Ensemble','-')]:
            ax.plot(h.ds,h[col]/1000,linewidth=2 if label=='Ensemble' else 1.2,linestyle=ls,label=label)
        ax.set(xlabel='Target month',ylabel='Monthly peak demand [GW]',title=f'{name}: recursive forecast issued in December 2023')
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3));ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.grid(alpha=.2);ax.legend(loc='upper center',bbox_to_anchor=(.5,1.01),ncol=5,frameon=False)
        fig.tight_layout();save(fig,f'Fig{num}_{c}_backtest')

    for c,name,num in [('spain','Spain',4),('france','France',5)]:
        hist=pd.read_csv(DATA/f'{c}_history.txt',sep='\t',comment='#',parse_dates=['ds'])
        d=pd.read_csv(DATA/f'{c}_TURNOVER_monthly_peak_scenarios.csv',parse_dates=['ds'])
        b=pd.read_csv(DATA/f'{c}_baseline_monthly.csv',parse_dates=['ds'])
        fig,ax=plt.subplots(figsize=(9.0,4.3))
        ax.plot(d.ds,d.P50_MW/1000,linewidth=1.25,label='Turnover scenario: P50')
        ax.fill_between(d.ds,d.P5_MW/1000,d.P95_MW/1000,alpha=.18,label='Conditional P5–P95')
        ax.plot(hist.ds,hist.y_mw/1000,linewidth=1.2,label='Observed monthly peaks')
        ax.plot(b.ds,b.P50_MW/1000,linestyle='--',linewidth=1.0,label='Statistical baseline: P50')
        ax.axvline(pd.Timestamp('2026-01-01'),linestyle=':',linewidth=1)
        ax.set(xlabel='Calendar year',ylabel='Monthly peak demand [GW]',title=f'{name}: observed peaks and conditional monthly scenarios',xlim=(pd.Timestamp('2015-01-01'),pd.Timestamp('2046-01-01')))
        ax.xaxis.set_major_locator(mdates.YearLocator(5));ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.grid(alpha=.2);ax.legend(loc='upper center',bbox_to_anchor=(.5,1.02),ncol=2,frameon=False)
        fig.tight_layout();save(fig,f'Fig{num}_{c}_monthly')

    for c,name,num in [('spain','Spain',6),('france','France',7)]:
        d=pd.read_csv(DATA/f'{c}_TURNOVER_annual.csv')
        fig,ax=plt.subplots(figsize=(9.0,4.4))
        ax.plot(d.year,d.P50_MW/1000,linewidth=2.1,marker='o',markersize=3,label='Turnover: P50')
        ax.fill_between(d.year,d.P5_MW/1000,d.P95_MW/1000,alpha=.18,label='Turnover: conditional P5–P95')
        for case,label,ls in [('MILESTONES','Milestones: P50','--'),('FAST_RENEWAL','Fast renewal: P50','-.')]:
            p=pd.read_csv(DATA/f'{c}_{case}_annual.csv');ax.plot(p.year,p.P50_MW/1000,linewidth=1.6,linestyle=ls,label=label)
        b=pd.read_csv(DATA/f'{c}_baseline_annual.csv')
        ax.plot(b.year,b.P50_MW/1000,linestyle=':',linewidth=1.6,label='Statistical baseline: P50')
        annual_axis(ax,f'{name}: conditional annual peak-demand scenarios')
        ax.legend(loc='upper center',bbox_to_anchor=(.5,1.03),ncol=2,frameon=False)
        fig.tight_layout();save(fig,f'Fig{num}_{c}_annual')

    for c,name,num in [('spain','Spain',8),('france','France',9)]:
        d=pd.read_csv(DATA/f'{c}_TURNOVER_annual_deterministic_contributions.csv').set_index('year').loc[[2030,2035,2045]]
        fig,ax=plt.subplots(figsize=(9.0,4.5))
        groups=[('EV_BEV_increment_MW','Battery electric cars'),('EV_PHEV_increment_MW','Plug-in hybrid cars'),
                ('HP_new_electricity_MW','New heat pumps'),('HP_displaced_resistance_MW','Displaced resistance heating'),
                ('industry_increment_MW','Industry'),('electrolyzers_increment_MW','Electrolyzers'),
                ('data_centers_increment_MW','Data centers'),('ports_increment_MW','Ports'),('rail_increment_MW','Rail')]
        # Separate new electricity demand from displaced electric heating.
        aliases={'HP_new_electricity_MW':['HP_electricity_new_increment_MW','heat_pumps_new_increment_MW','HP_gross_increment_MW','HP_new_increment_MW'],
                 'HP_displaced_resistance_MW':['HP_resistance_removed_increment_MW','heat_pumps_displaced_increment_MW','HP_displaced_increment_MW','HP_removed_resistance_MW']}
        pos=np.zeros(3);neg=np.zeros(3);x=np.arange(3)
        for key,label in groups:
            if key not in d:
                key=next((a for a in aliases.get(key,[]) if a in d),None)
            if key is None:continue
            v=d[key].to_numpy()/1000;bottom=np.where(v>=0,pos,neg)
            ax.bar(x,v,bottom=bottom,label=label,width=.52)
            pos+=np.where(v>0,v,0);neg+=np.where(v<0,v,0)
        ax.plot(x,d.total_increment_MW/1000,marker='D',linestyle='none',markersize=6,label='Net structural increment')
        ax.axhline(0,linewidth=.8)
        ax.set(xticks=x,xticklabels=['2030','2035','2045'],xlabel='Calendar year',ylabel='Increment at the annual critical month [GW]',title=f'{name}: physical contributions in the turnover scenario')
        ax.grid(axis='y',alpha=.2);ax.legend(loc='upper center',bbox_to_anchor=(.5,-.16),ncol=3,frameon=False)
        fig.tight_layout();save(fig,f'Fig{num}_{c}_contributions')

    for c,name,num in [('spain','Spain',10),('france','France',11)]:
        labels={'MILESTONES':'Vehicle stocks held after 2030','TURNOVER':'Reference turnover', 'FAST_RENEWAL':'Faster BEV replacement',
                'MANAGED_CHARGING':'Lower EV coincidence', 'HIGH_COINCIDENCE':'Higher EV coincidence',
                'EV_LOW_ENERGY':'Lower vehicle electricity use','EV_HIGH_ENERGY':'Higher vehicle electricity use',
                'HEAT_FOSSIL_ONLY':'No resistance-heating replacement','HEAT_HALF_JOULE':'50% resistance-heating replacement',
                'COLD_BACKUP':'Cold-weather COP and backup','BASELINE_OVERLAP50':'50% additionality','PROJECTS_DELAY3':'Delayed projects'}
        # Keep every case in the CSV and show a readable selection in the figure.
        actual={p.stem[len(c)+1:-len('_annual')]:p for p in DATA.glob(f'{c}_*_annual.csv')}
        selected=[]
        for key,label in labels.items():
            if key in actual:
                q=pd.read_csv(actual[key]);v=float(q.loc[q.year.eq(2045),'P50_MW'].iloc[0])/1000;selected.append((label,v))
        fig,ax=plt.subplots(figsize=(9.0,4.4))
        base=float(pd.read_csv(DATA/f'{c}_TURNOVER_annual.csv').query('year==2045').P50_MW.iloc[0])/1000
        ys=np.arange(len(selected));xs=np.array([v for _,v in selected])
        ax.hlines(ys,np.minimum(xs,base),np.maximum(xs,base),linewidth=3)
        ax.plot(xs,ys,'o',markersize=6);ax.axvline(base,linestyle='--',linewidth=1)
        ax.set_yticks(ys,[label for label,_ in selected]);ax.invert_yaxis()
        ax.set(xlabel='Conditional annual P50 in 2045 [GW]',title=f'{name}: structural sensitivities, not probability intervals')
        ax.grid(axis='x',alpha=.2)
        fig.tight_layout();save(fig,f'Fig{num}_{c}_sensitivity')
    for filename in ['Fig1.png', 'Fig2_workflow_LEVEL_structural.png', 'Fig2_workflow_LEVEL_structural.pdf']:
        shutil.copy2(ROOT / 'data/illustrations' / filename, FIG / filename)
    check('thirteen_article_figures', len(list(FIG.glob('*.png'))) == 13)


def verify_inputs() -> dict[str, str]:
    """Check that the inputs still match the verified release archive."""
    manifest = jload(ROOT / 'frozen_input_manifest.json')
    for rel, expected in manifest.items():
        path = (ROOT / rel).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            raise ValueError(f'A reproduction input is missing: {rel}')
        if sha256(path) != expected:
            raise ValueError(f'Input changed: {rel}. This run cannot be certified against the verified release archive.')
    check('input_integrity', True, len(manifest))
    return manifest


def compare_numeric(actual: pd.DataFrame, expected: pd.DataFrame, columns: list[str],
                    keys: list[str], name: str, atol: float = 1e-6) -> float:
    """Compare values rather than CSV separator or date formatting."""
    a = actual.sort_values(keys).reset_index(drop=True)
    e = expected.sort_values(keys).reset_index(drop=True)
    check(name + '/rows', len(a) == len(e), (len(a), len(e)))
    for key in keys:
        check(name + '/key/' + key, a[key].astype(str).tolist() == e[key].astype(str).tolist())
    av, ev = a[columns].to_numpy(float), e[columns].to_numpy(float)
    check(name + '/finite', np.isfinite(av).all() and np.isfinite(ev).all())
    difference = float(np.max(np.abs(av - ev))) if av.size else 0.0
    check(name + '/values', np.allclose(av, ev, atol=atol, rtol=2e-12), difference)
    return difference


def read_raw_paths(country: str, cfg: dict, run_dir: Path, retrain: bool) -> tuple:
    """Load archived component paths or fit a new set of models."""
    source = ROOT / 'frozen_statistical' / country
    target = demand_file(country)
    demand = stat_load_demand(target, country.upper(), cfg)
    protocol = stat_make_protocol(demand, cfg)
    history = demand.set_index('ds').y_mw
    raw_parts = []
    covariate_records = []
    if retrain:
        stat_require_real_triad()
        annual = pd.read_csv(ROOT / 'data/statistical/exogenous_cache/exogenous_annual_worldbank.csv')
        monthly = pd.read_csv(ROOT / 'data/statistical/exogenous_cache/exogenous_monthly_eurostat.csv')
        repo = stat_DriverRepository(annual, monthly, cfg)
        cases = [(r.phase, r.origin, 24) for r in protocol.itertuples()]
        cases.append(('future_forecast', demand.ds.max(), 240))
        for phase, origin, horizon in cases:
            logging.info('%s: training real LSTM, LightGBM and TFT models at origin %s', country, origin)
            cov, availability = repo.features(country.upper(), demand.ds.min(), origin,
                                             origin + pd.DateOffset(months=horizon))
            hist = demand[demand.ds <= origin].merge(cov, on='ds', validate='one_to_one')
            future = cov[cov.ds > origin].copy()
            tag = f'{phase}_{origin:%Y%m}'
            model = stat_fit_triad(hist, country.upper(), cfg, run_dir / f'{country}_TFT.log')
            paths = stat_recursive_paths(model, future)
            frame = stat_prediction_frame(country.upper(), phase, origin, future.ds, paths, demand)
            raw_parts.append(frame)
            out = run_dir / 'training_record' / country / tag
            out.mkdir(parents=True, exist_ok=True)
            model.trace.to_csv(out / 'LSTM_loss.csv', index=False)
            cov.to_csv(out / 'covariates.csv', index=False)
            availability.to_csv(out / 'availability.csv', index=False)
            if phase == 'future_forecast':
                stat_save_final_model_parameters(model, out / 'parameters')
            covariate_records.append((tag, cov))
            del model
            gc.collect()
    else:
        for name in ['development_recursive_predictions.csv', 'holdout_recursive_predictions.csv',
                     'monthly_peak_demand_scenarios.csv']:
            raw_parts.append(pd.read_csv(source / name, parse_dates=['origin', 'ds']))
        # Reuse the archived component forecasts.
        for path in sorted((source / 'covariates_by_origin').glob('*.csv')):
            covariate_records.append((path.stem, pd.read_csv(path, parse_dates=['ds'])))
    raw = pd.concat(raw_parts, ignore_index=True)
    core_columns = ['country', 'phase', 'origin', 'ds', 'horizon', 'y_true',
                    'pred_LSTM', 'pred_LGBM', 'pred_TFT']
    raw = raw[core_columns].copy()
    raw['origin'] = pd.to_datetime(raw.origin)
    raw['ds'] = pd.to_datetime(raw.ds)
    if raw.duplicated(['phase', 'origin', 'horizon']).any():
        raise ValueError('Duplicate predictions were found for the same origin and horizon.')
    # Match observed targets to their calendar months.
    actual = history.reindex(raw.ds).to_numpy(float)
    present = raw.phase.ne('future_forecast').to_numpy()
    check(country + '/validation_observations',
          np.array_equal(raw.loc[present, 'y_true'].to_numpy(float), actual[present]))
    raw['y_true'] = actual
    for (phase, origin), g in raw.groupby(['phase', 'origin']):
        g = g.sort_values('horizon')
        length = 240 if phase == 'future_forecast' else 24
        check(f'{country}/{phase}/{origin}/horizons',
              np.array_equal(g.horizon.to_numpy(), np.arange(1, length + 1)))
        check(f'{country}/{phase}/{origin}/dates',
              pd.DatetimeIndex(g.ds).equals(pd.date_range(origin + pd.DateOffset(months=1), periods=length, freq='MS')))
    return raw, demand, protocol, covariate_records


def regenerate_statistical_baseline(country: str, destination: Path, retrain: bool) -> None:
    """Estimate weights and error parameters, then draw the baseline paths."""
    cfg = jload(ROOT / 'data/statistical_configuration.json')['config']
    out = destination / country
    out.mkdir(parents=True, exist_ok=True)
    raw, demand, protocol, covariates = read_raw_paths(country, cfg, destination, retrain)
    raw.to_csv(out / 'raw_three_model_paths.csv', index=False)
    for tag, cov in covariates:
        target = out / 'covariates_by_origin' / (tag + '.csv')
        target.parent.mkdir(parents=True, exist_ok=True)
        cov.to_csv(target, index=False)
    cal_raw = raw[raw.phase.eq('weight_calibration')].copy()
    weights, weight_table = stat_inverse_rmse_weights(cal_raw, cfg)
    weight_table.to_csv(out / 'ensemble_weights.csv', index=False)
    selected, control, decision = stat_select_probability_layer(cal_raw, cfg, cfg['seed'] + 1000, out)
    if not retrain:
        expected = jload(ROOT / 'frozen_statistical' / country / 'probability_selection_frozen.json')['selected']
        for k in ['family', 'block_length', 'scale']:
            check(f'{country}/probability_selection/{k}', selected[k] == expected[k], selected[k])
    logging.info('%s: weights %s; bootstrap %s, block %s, scale %.2f',
                 country, weights, selected['family'], selected['block_length'], selected['scale'])
    development = [stat_ensemble_frame(cal_raw, weights)]
    choices = {'SELECTED_BASELINE': selected, 'HORIZON_SCALE_BASELINE': control}
    audit_records, audit_scores = [], []
    for i, row in enumerate(protocol[protocol.phase.eq('probability_tuning')].itertuples()):
        current = raw[raw.phase.eq(row.phase) & raw.origin.eq(row.origin)].sort_values('horizon')
        pred = stat_ensemble_frame(current, weights)
        bank = stat_available_residuals(pd.concat(development, ignore_index=True), row.origin)
        check(f'{country}/audit_{row.origin:%Y%m}/time_cutoff', bank.ds.max() < pred.ds.min())
        for variant, choice in choices.items():
            model = stat_ConditionalErrorModel(bank, choice['family'], cfg)
            samples = stat_conditional_draws(model, pred, int(choice['block_length']), choice['scale'],
                                             cfg['mc_final_simulations'], cfg['seed'] + 12000 + i * 1000)
            detail, score = stat_probability_records(pred, samples, cfg, variant, 'prequential_audit')
            detail['calibration_as_of'] = row.origin
            detail['residual_fit_last_target'] = bank.ds.max()
            audit_records.append(detail)
            audit_scores.append(score)
            if variant == 'SELECTED_BASELINE':
                stat_export_probability_state(model, choice, row.origin,
                                              out / 'probability_states' / f'audit_{row.origin:%Y%m}')
        development.append(pred)
    development = pd.concat(development, ignore_index=True)
    development.to_csv(out / 'development_recursive_predictions.csv', index=False)
    audit = pd.concat(audit_records, ignore_index=True)
    audit.to_csv(out / 'prequential_probability_predictions.csv', index=False)
    pd.DataFrame(audit_scores).to_csv(out / 'prequential_probability_metrics.csv', index=False)
    terminal_origin = protocol.loc[protocol.phase.eq('terminal_holdout'), 'origin'].iat[0]
    bank = stat_available_residuals(development, terminal_origin)
    # Reuse the pre-terminal error bank for the long-horizon simulation.
    check(country + '/holdout_excluded_from_bank', bank.ds.max() <= terminal_origin)
    hold = stat_ensemble_frame(raw[raw.phase.eq('terminal_holdout')].sort_values('horizon'), weights)
    future = stat_ensemble_frame(raw[raw.phase.eq('future_forecast')].sort_values('horizon'), weights)
    hold_scores = []
    for variant, choice in choices.items():
        model = stat_ConditionalErrorModel(bank, choice['family'], cfg)
        hs = stat_conditional_draws(model, hold, int(choice['block_length']), choice['scale'],
                                    cfg['mc_final_simulations'], cfg['seed'] + 24000)
        detail, score = stat_probability_records(hold, hs, cfg, variant, 'terminal_retrospective')
        hold_scores.append(score)
        fs = stat_conditional_draws(model, future, int(choice['block_length']), choice['scale'],
                                    cfg['mc_final_simulations'], cfg['seed'] + 36000)
        monthly = future.drop(columns=['y_true']).copy()
        monthly['ensemble_deterministic_MW'] = monthly.pred_ENS
        monthly['P5_MW'], monthly['P50_MW'], monthly['P95_MW'] = np.quantile(fs, [.05, .5, .95], axis=0)
        monthly['variant'] = variant
        annual, annual_samples = stat_annualize(monthly.ds, fs, monthly.pred_ENS)
        annual['country'] = country.upper()
        annual['variant'] = variant
        annual.insert(1, 'study_year', np.arange(1, len(annual) + 1))
        folder = out if variant == 'SELECTED_BASELINE' else out / 'horizon_only_control'
        folder.mkdir(exist_ok=True)
        monthly.to_csv(folder / 'monthly_peak_demand_scenarios.csv', index=False)
        annual.to_csv(folder / 'annual_peak_demand_scenarios.csv', index=False)
        if variant == 'SELECTED_BASELINE':
            detail.to_csv(out / 'holdout_recursive_predictions.csv', index=False)
            stat_export_probability_state(model, choice, terminal_origin,
                                          out / 'probability_states/frozen_for_holdout_and_future')
            for label, matrix, frame in [('holdout', hs, detail), ('future', fs, future)]:
                np.savez_compressed(out / f'{label}_monthly_simulations.npz', samples_MW=matrix,
                                    dates=frame.ds.dt.strftime('%Y-%m-%d').to_numpy(dtype='U10'),
                                    deterministic_MW=frame.pred_ENS.to_numpy(float))
                if not retrain:
                    with np.load(ROOT / 'frozen_statistical' / country / f'{label}_monthly_simulations.npz', allow_pickle=False) as z:
                        difference = float(np.max(np.abs(matrix - z['samples_MW'])))
                    check(f'{country}/{label}/recalculated_samples', difference < 1e-6, difference)
            aa = stat_annual_validation(detail.ds, hs, detail, variant)
            aa.to_csv(out / 'holdout_annual_peak_diagnostics.csv', index=False)
            pivot = model.bank.pivot(index='origin', columns='horizon', values='z').sort_index()
            if not pivot.isna().any().any():
                ac, lb, dep = stat_dependence_diagnostics(pivot.to_numpy(), country, cfg)
                ac.to_csv(out / 'final_residual_acf.csv', index=False)
                lb.to_csv(out / 'final_residual_ljungbox.csv', index=False)
                jwrite(out / 'final_residual_dependence.json', dep)
    pd.DataFrame(hold_scores).to_csv(out / 'holdout_probabilistic_metrics.csv', index=False)
    calendar = protocol.copy()
    calendar['selection_history'] = calendar.uses_for_selection
    calendar['uses_for_selection'] = calendar.phase.eq('weight_calibration')
    calendar['baseline_role'] = calendar.phase.replace({'probability_tuning': 'prequential_audit_no_hyperparameter_selection'})
    calendar.to_csv(out / 'validation_calendar.csv', index=False)
    stat_error_correlations(pd.concat([development, hold], ignore_index=True)).to_csv(out / 'base_model_error_correlations.csv', index=False)
    final_cov = next((x for tag, x in covariates if tag.startswith('future_forecast')), None)
    if final_cov is None:
        raise ValueError('Covariates for the final forecast origin are missing.')
    final_cov[final_cov.ds > demand.ds.max()].to_csv(out / 'future_exogenous_features.csv', index=False)
    jwrite(out / 'statistical_run_provenance.json', {
        'new_neural_training': retrain, 'weights_recomputed': True,
        'probability_candidates_reselected_from_archived_development': True,
        'monte_carlo_resampled': True, 'n_simulations': cfg['mc_final_simulations'],
        'future_seed': cfg['seed'] + 36000, 'holdout_seed': cfg['seed'] + 24000,
        'model_paths_source': 'new_training' if retrain else 'archived_real_LSTM_LightGBM_TFT_predictions',
        'structural_coverage_claim': False})


def reproduce_benchmark_metrics(country: str, out: Path, config: dict) -> pd.DataFrame:
    """Recalculate the reported scores from predictions without rerunning the optimizer."""
    archive = pd.read_csv(ROOT / 'frozen_statistical' / country / 'statistical_benchmark_paths.csv',
                          parse_dates=['origin', 'ds'])
    hist = stat_load_demand(demand_file(country), country,
                            jload(ROOT / 'data/statistical_configuration.json')['config']).set_index('ds').y_mw
    development = pd.read_csv(REF / country / 'development_recursive_predictions.csv', parse_dates=['origin', 'ds'])
    terminal = pd.read_csv(REF / country / 'holdout_recursive_predictions.csv', parse_dates=['origin', 'ds'])
    paths = pd.concat([development, terminal], ignore_index=True)
    result = []
    for (phase, origin), g in paths.groupby(['phase', 'origin'], sort=True):
        g = g.sort_values('horizon')
        truth = hist.reindex(g.ds).to_numpy(float)
        check(f'{country}/benchmark/{origin:%Y%m}/observations', np.array_equal(truth, g.y_true.to_numpy(float)))
        values = {name: g['pred_' + name].to_numpy(float) for name in ['LSTM', 'LGBM', 'TFT']}
        values['ENSEMBLE'] = g.pred_ENS.to_numpy(float)
        before = hist.loc[:origin].to_numpy(float)
        values['Persistence'] = np.full(len(g), before[-1])
        values['SeasonalNaive'] = np.tile(before[-12:], (len(g) + 11) // 12)[:len(g)]
        for model in ['ETS_damped', 'SARIMA_predeclared', 'ETS_AIC_ARCHIVED', 'SARIMA_AIC_ARCHIVED']:
            old = archive[archive.origin.eq(origin) & archive.model.eq(model)].set_index('ds').reindex(g.ds)
            if len(old) != len(g) or old.pred_MW.isna().any():
                raise ValueError(f'Incomplete benchmark: {country}/{origin}/{model}')
            check(f'{country}/{origin:%Y%m}/{model}/same_targets', np.array_equal(old.y_true.to_numpy(float), truth))
            values[model] = old.pred_MW.to_numpy(float)
        for model, pred in values.items():
            for i, r in enumerate(g.itertuples()):
                result.append(dict(country=country, phase=phase, origin=origin, ds=r.ds,
                                   horizon=int(r.horizon), model=model, y_true=float(truth[i]), pred_MW=float(pred[i])))
    result = pd.DataFrame(result)
    result.to_csv(out / 'statistical_benchmark_paths.csv', index=False)
    records = []
    for (phase, model), g in result.groupby(['phase', 'model']):
        for scope, h in [('full_path', 0), ('endpoint', 1), ('endpoint', 3), ('endpoint', 6), ('endpoint', 12), ('endpoint', 24)]:
            subset = g if h == 0 else g[g.horizon.eq(h)]
            records.append(dict(country=country, phase=phase, model=model, scope=scope, horizon=h,
                                n_pairs=len(subset), n_unique_months=subset.ds.nunique(), n_origins=subset.origin.nunique(),
                                **metrics(subset.y_true, subset.pred_MW)))
    scores = pd.DataFrame(records)
    scores.to_csv(out / 'statistical_benchmark_metrics.csv', index=False)
    if EXECUTION_MODE == 'REPRODUCE_RELEASE':
        expected = pd.read_csv(ROOT / 'verified_outputs' / country / 'statistical_benchmark_metrics.csv')
        compare_numeric(scores, expected, ['RMSE_MW', 'MAE_MW', 'WAPE_pct', 'Bias_MW'],
                        ['country', 'phase', 'model', 'scope', 'horizon'], country + '/verified_metrics')
    shutil.copy2(ROOT / 'frozen_statistical' / country / 'statistical_benchmark_fit_diagnostics.csv',
                 out / 'statistical_benchmark_fit_diagnostics_archived.csv')
    bad = result[result.model.eq('SARIMA_AIC_ARCHIVED') & result.country.eq('spain') &
                 result.origin.eq(pd.Timestamp('2019-06-01'))].copy()
    if not bad.empty:
        bad['status'] = 'failed_archived_fit_not_evidence_of_forecast_superiority'
        bad.to_csv(out / 'known_benchmark_failure.csv', index=False)
    return scores


def verify_scenarios(run: Path) -> pd.DataFrame:
    """Check all sensitivities against the run supporting the manuscript."""
    rows = []
    for c in ['spain', 'france']:
        for case in jload(DATA / 'scenario_protocol.json')['scenarios']:
            tag = case['id']
            new = pd.read_csv(run / c / tag / 'annual_peak_scenarios.csv')
            old = pd.read_csv(ROOT / 'verified_outputs' / c / tag / 'annual_peak_scenarios.csv')
            error = compare_numeric(new, old, ['P5_MW', 'P50_MW', 'P95_MW', 'deterministic_max_MW'],
                                    ['year'], f'{c}/{tag}/annual_verified')
            for name, keys in [('monthly_peak_scenarios.csv', ['ds']),
                               ('annual_deterministic_contributions.csv', ['year']),
                               ('vehicle_stock_flow.csv', ['year'])]:
                a = pd.read_csv(run / c / tag / name)
                e = pd.read_csv(ROOT / 'verified_outputs' / c / tag / name)
                cols = [x for x in e.select_dtypes(include='number').columns if x in a and x not in keys]
                compare_numeric(a, e, cols, keys, f'{c}/{tag}/{name}')
            rows.append(dict(country=c, scenario=tag, max_annual_difference_MW=error))
    frame = pd.DataFrame(rows)
    frame.to_csv(run / 'scenario_reproduction_checks.csv', index=False)
    return frame


def write_planning_tables(run: Path, annual: pd.DataFrame, scenarios: list, reference: str) -> None:
    annual.to_csv(run / 'all_countries_all_conditional_scenarios.csv', index=False)
    for case in scenarios:
        tag = case['id']
        d = annual[annual.scenario.eq(tag)]
        wide = d.pivot(index=['year', 'study_year'], columns='country', values=['P5_MW', 'P50_MW', 'P95_MW'])
        wide.columns = [f'{country}_{q}' for q, country in wide.columns]
        wide = wide.reset_index()
        wide['scenario'] = tag
        wide['structural_probability'] = 'not_assigned'
        dest = run / 'planning_tables' / tag
        dest.mkdir(parents=True, exist_ok=True)
        wide.to_csv(dest / 'annual_peak_demand_input_GEP_TEP_all_scenarios.csv', index=False)
        median = wide[['year', 'study_year'] + [x for x in wide if x.endswith('_P50_MW')] + ['scenario']]
        median.to_csv(dest / 'annual_peak_demand_P50_MW.csv', index=False)
        if tag == reference:
            wide.to_csv(run / 'annual_peak_demand_input_GEP_TEP_all_scenarios.csv', index=False)
            median.to_csv(run / 'annual_peak_demand_P50_MW.csv', index=False)


def prepare_figure_data(run: Path) -> Path:
    """Write the data used by the manuscript figures."""
    dest = run / 'article/figure_data'
    dest.mkdir(parents=True, exist_ok=True)
    names = {'holdout_recursive_predictions.csv': 'holdout.csv', 'validation_calendar.csv': 'calendar.csv',
             'monthly_peak_demand_scenarios.csv': 'baseline_monthly.csv',
             'annual_peak_demand_scenarios.csv': 'baseline_annual.csv'}
    for country in ['spain', 'france']:
        for source, target in names.items():
            shutil.copy2(REF / country / source, dest / f'{country}_{target}')
        history = stat_load_demand(demand_file(country), country,
                                   jload(ROOT / 'data/statistical_configuration.json')['config'])
        history.to_csv(dest / f'{country}_history.txt', sep='\t', index=False, date_format='%Y-%m-%d')
        for case in jload(DATA / 'scenario_protocol.json')['scenarios']:
            tag = case['id']
            for old, new in [('annual_peak_scenarios.csv', 'annual.csv'),
                             ('monthly_peak_scenarios.csv', 'monthly_peak_scenarios.csv'),
                             ('annual_deterministic_contributions.csv', 'annual_deterministic_contributions.csv')]:
                shutil.copy2(run / country / tag / old, dest / f'{country}_{tag}_{new}')
    return dest


def export_article_tables(run: Path) -> None:
    """Generate the numerical tables and check their rounding against the manuscript."""
    source = '\n'.join(jload(ROOT / 'verified_outputs/table_templates.json')['tables'])
    blocks = re.findall(r'\\begin\{table\}.*?\\end\{table\}', source, flags=re.S)
    output = run / 'article/tables'
    output.mkdir(parents=True, exist_ok=True)
    table_blocks = {}
    table_numbers = {}
    for number, block in enumerate(blocks, 1):
        label = re.search(r'\\label\{([^}]+)\}', block).group(1)
        table_blocks[label] = block
        table_numbers[label] = number
        if number <= 5:
            # The literature review and input definitions are documentary text.
            (output / f'Table_{number:02d}_{label.split(":")[-1]}.tex').write_text(block + '\n', encoding='utf-8')
    model_names = {'LSTM': 'LSTM', 'LightGBM': 'LGBM', 'TFT': 'TFT', 'Ensemble': 'ENSEMBLE',
                   'Persistence': 'Persistence', 'Seasonal naive': 'SeasonalNaive',
                   'Damped ETS': 'ETS_damped', 'Prespecified SARIMA': 'SARIMA_predeclared',
                   'AIC-selected ETS (archived)': 'ETS_AIC_ARCHIVED',
                   'AIC-selected SARIMA (archived)': 'SARIMA_AIC_ARCHIVED'}
    scores = pd.concat([pd.read_csv(run / c / 'statistical_benchmark_metrics.csv') for c in ['spain', 'france']], ignore_index=True)
    checks = []

    def replace_row(label: str, prefix: str, computed: list[float]) -> None:
        block = table_blocks[label]
        match = re.search(r'^' + re.escape(prefix) + r'\s*&\s*(.*?)\\\\\s*$', block, flags=re.M)
        if match is None:
            raise ValueError(f'Row {prefix!r} was not found in {label}.')
        printed = [x.strip() for x in match.group(1).split('&')]
        if len(printed) != len(computed):
            raise ValueError(f'Column counts differ in {label}/{prefix}.')
        rendered = []
        for old, value in zip(printed, computed):
            decimals = len(old.split('.')[1]) if '.' in old else 0
            tolerance = 0.5 * 10**(-decimals) + 1e-9
            difference = abs(float(old) - float(value))
            checks.append(dict(table=label, row=prefix, printed=old, recalculated=float(value),
                               rounding_tolerance=tolerance, difference=difference,
                               passed=difference <= tolerance))
            rendered.append(f'{value:.{decimals}f}')
        replacement = prefix + ' & ' + ' & '.join(rendered) + r' \\'
        table_blocks[label] = block[:match.start()] + replacement + block[match.end():]

    terminal_rows = []
    for printed, model in model_names.items():
        vals = []
        for country in ['spain', 'france']:
            r = scores[(scores.country == country) & (scores.phase == 'terminal_holdout') &
                       (scores.model == model) & (scores.scope == 'full_path')].iloc[0]
            vals.extend(float(r[k]) for k in ['RMSE_MW', 'MAE_MW', 'WAPE_pct'])
            terminal_rows.append(dict(country=country, model=model, **{k: float(r[k]) for k in ['RMSE_MW', 'MAE_MW', 'WAPE_pct', 'Bias_MW']}))
        replace_row('tab:terminal', printed, vals)
    pd.DataFrame(terminal_rows).to_csv(output / 'Table_06_terminal_metrics.csv', index=False)
    endpoint_rows = []
    for country in ['spain', 'france']:
        for horizon in [1, 3, 6, 12, 24]:
            vals = []
            for model in ['LSTM', 'LGBM', 'TFT', 'ENSEMBLE', 'ETS_damped', 'SeasonalNaive']:
                r = scores[(scores.country == country) & (scores.phase == 'probability_tuning') &
                           (scores.model == model) & (scores.scope == 'endpoint') & (scores.horizon == horizon)].iloc[0]
                vals.append(float(r.RMSE_MW))
                endpoint_rows.append(dict(country=country, horizon=horizon, model=model, RMSE_MW=float(r.RMSE_MW)))
            replace_row('tab:horizons', country.title() + f' & {horizon}', vals)
    pd.DataFrame(endpoint_rows).to_csv(output / 'Table_07_endpoint_RMSE.csv', index=False)
    prob_rows = []
    for country in ['spain', 'france']:
        for label, filename in [('2021-origin audit', 'prequential_probability_metrics.csv'),
                                ('Terminal 2024--2025', 'holdout_probabilistic_metrics.csv')]:
            d = pd.read_csv(REF / country / filename)
            d = d[d.variant.eq('SELECTED_BASELINE')]
            cols = ['PICP90_pct', 'MPIW90_MW', 'IntervalScore90_MW', 'CRPS_MW']
            vals = [float(d[k].mean()) for k in cols]
            replace_row('tab:probability', country.title() + ' & ' + label, vals)
            prob_rows.append(dict(country=country, period=label, **dict(zip(cols, vals))))
    pd.DataFrame(prob_rows).to_csv(output / 'Table_08_probability_metrics.csv', index=False)
    annuals = {c: pd.read_csv(run / c / 'TURNOVER/annual_peak_scenarios.csv').set_index('year') for c in ['spain', 'france']}
    final_rows = []
    for year in range(2026, 2046):
        vals = [float(annuals[c].loc[year, k]) / 1000 for c in ['spain', 'france'] for k in ['P5_MW', 'P50_MW', 'P95_MW']]
        replace_row('tab:annual', str(year), vals)
        final_rows.append(dict(year=year, **dict(zip(['SPAIN_P5_GW', 'SPAIN_P50_GW', 'SPAIN_P95_GW', 'FRANCE_P5_GW', 'FRANCE_P50_GW', 'FRANCE_P95_GW'], vals))))
    pd.DataFrame(final_rows).to_csv(output / 'Table_09_annual_TURNOVER_GW.csv', index=False)
    audit = pd.DataFrame(checks)
    audit.to_csv(run / 'article/printed_numbers_verification.csv', index=False)
    if EXECUTION_MODE == 'REPRODUCE_RELEASE':
        check('printed_paper_numbers', audit.passed.all(), f'{len(audit)} numerical cells')
    for label, block in table_blocks.items():
        number = table_numbers[label]
        (output / f'Table_{number:02d}_{label.split(":")[-1]}.tex').write_text(block + '\n', encoding='utf-8')
    if EXECUTION_MODE != 'REPRODUCE_RELEASE':
        # Keep new fits separate from the reference tables.
        (run / 'article/NEW_EXPERIMENT.txt').write_text('New training run: these results do not automatically replace the manuscript numbers.\n', encoding='utf-8')
    evidence = [
        (1, 'Literature review', 'Manuscript text and bibliography; not a model output'),
        (2, 'Input data', 'data/statistical and archived sources; coverage by variable'),
        (3, 'Hyperparameters', 'data/statistical_configuration.json'),
        (4, 'Chronological protocol', 'statistical_recalculation/*/validation_calendar.csv'),
        (5, 'Project assumptions', 'data/structural/project_capacity_knots.csv'),
        (6, 'Terminal scores', 'Archived model predictions and observations; recalculated metrics'),
        (7, 'Endpoint errors', 'Four audit origins; metrics recalculated by horizon'),
        (8, 'Probability scores', 'Bootstrap recalculated from time-admissible residual banks'),
        (9, 'Annual quantiles', 'Annual maxima of the 5000 simulations under TURNOVER')]
    pd.DataFrame(evidence, columns=['table', 'content', 'computational_basis']).to_csv(output / 'table_evidence_map.csv', index=False)


def write_final_report(run: Path, elapsed: float, retrain: bool) -> None:
    success = all(row['passed'] for row in CHECKS)
    checks = pd.DataFrame(CHECKS)
    checks.to_csv(run / 'verification_checks.csv', index=False)
    environment = {'python': sys.version, 'platform': platform.platform(), 'executable': sys.executable}
    for package in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'statsmodels', 'matplotlib', 'torch', 'lightgbm', 'lightning', 'pytorch-forecasting']:
        try:
            environment[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            environment[package] = 'not_installed'
    jwrite(run / 'environment.json', environment)
    status = dict(status='COMPLETED' if success else 'FAILED', execution_mode=EXECUTION_MODE,
                  interface_language='en', package_version=VERSION,
                  same_verified_release_results=success and not retrain,
                  elapsed_seconds=elapsed, checks=len(CHECKS), all_checks_passed=success,
                  retrained_LSTM_LightGBM_TFT=retrain, weights_recalculated=True,
                  probability_selection_recalculated=True, monte_carlo_generated=True,
                  physical_components_recalculated=True, scenarios_per_country=23,
                  countries=['spain', 'france'], manuscript_figure_count=13,
                  data_figures_regenerated=11, conceptual_illustrations_preserved=2,
                  conditional_scenarios_not_validated_20_year_predictions=True,
                  source_date='2026-09-21', numerical_tolerance_MW=1e-6,
                  complete_utc=datetime.now(timezone.utc).isoformat())
    jwrite(run / 'execution_status.json', status)
    p = pd.read_csv(run / 'annual_peak_demand_P50_MW.csv')
    key_years = p[p.year.isin([2026, 2030, 2035, 2045])]
    text = ('RELEASE REPRODUCTION\n' if not retrain else 'NEW TRAINING RUN\n')
    text += f'Status: {status["status"]}\nChecks: {len(CHECKS)}\n'
    text += f'Results: {run}\n\n' + key_years.to_string(index=False) + '\n\n'
    text += ('Weights, calibration, Monte Carlo, components, quantiles and metrics were recalculated.\n'
             'Neural-model predictions and ETS/SARIMA fits from the archived release\n'
             'are verified intermediate inputs in reproduction mode. They are not new\n'
             'observations or models trained during this run.\n'
             'The bands are conditional on each physical scenario.\n')
    if retrain:
        text = text.replace('Neural-model predictions and ETS/SARIMA fits from the archived release\n'
                            'are verified intermediate inputs in reproduction mode. They are not new\n'
                            'observations or models trained during this run.\n',
                            'LSTM, LightGBM and TFT were retrained. Benchmarks remain\n'
                            'the archived fits, evaluated against the same observations.\n')
    (run / 'REPORT.txt').write_text(text, encoding='utf-8')
    (ROOT / 'LATEST_RUN.txt').write_text(str(run) + '\n', encoding='utf-8')
    shutil.copy2(Path(__file__), run / 'code_used.py.txt')
    shutil.copy2(ROOT / 'matlab/generate_forecasting_figures.m', run / 'generate_forecasting_figures.m')
    jwrite(run / 'output_manifest.json', {str(p.relative_to(run)): sha256(p) for p in sorted(run.rglob('*')) if p.is_file() and p.name != 'output_manifest.json'})
    package_results(run, ROOT / 'RESULTS_TO_SHARE.zip')
    print('\n' + text)
    print('Article figures:', run / 'article/Figures')
    print('Results archive:', ROOT / 'RESULTS_TO_SHARE.zip')


def record_current_inputs() -> dict[str, str]:
    """Record input identities for a new experiment."""
    return {str(p.relative_to(ROOT)): sha256(p)
            for p in sorted((ROOT / 'data').rglob('*')) if p.is_file()}


def preflight_country(country: str, cfg: dict) -> pd.DataFrame:
    """Check target and driver availability before loading the neural libraries."""
    history = stat_load_demand(demand_file(country), country, cfg)
    cache = ROOT / 'data/statistical/exogenous_cache'
    annual = pd.read_csv(cache / 'exogenous_annual_worldbank.csv')
    monthly = pd.read_csv(cache / 'exogenous_monthly_eurostat.csv')
    expected = [(annual, ('country', 'year', *stat_ANNUAL_DRIVERS), 'annual'),
                (monthly, ('country', 'ds', *stat_MONTHLY_DRIVERS), 'monthly')]
    for frame, columns, label in expected:
        missing = set(columns) - set(frame.columns)
        if missing:
            raise ValueError(f'{label} driver cache is missing columns: {sorted(missing)}.')
        if not frame['country'].eq(country.upper()).any():
            raise ValueError(f'{country}: no {label} exogenous data. Add this country to the {label} cache; see README.txt.')
    repo = stat_DriverRepository(annual, monthly, cfg)
    protocol = stat_make_protocol(history, cfg)
    for origin in list(protocol['origin']) + [history['ds'].max()]:
        for driver in stat_ANNUAL_DRIVERS:
            repo.annual_known(country.upper(), driver, origin)
        for driver in stat_MONTHLY_DRIVERS:
            repo.monthly_known(country.upper(), driver, origin)
    return history


def run_statistical_experiment(args, countries: list[str]) -> None:
    """Fit country-specific baselines without inventing physical scenarios."""
    global REF
    stamp = datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')
    run = args.output.resolve() if args.output is not None else ROOT / 'results' / stamp
    if run.exists() and any(run.iterdir()):
        raise ValueError(f'The output folder already contains files: {run}')
    run.mkdir(parents=True, exist_ok=True)
    CHECKS.clear()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(run / 'execution.log', encoding='utf-8')], force=True)
    jwrite(run / 'execution_status.json', {'status': 'RUNNING', 'execution_mode': 'RETRAIN_STATISTICAL'})
    started = time.monotonic()
    try:
        stat_load_probability_stack()
        cfg = jload(ROOT / 'data/statistical_configuration.json')['config']
        for country in countries:
            preflight_country(country, cfg)
            export_observed_inputs(country, run, cfg)
        identities = record_current_inputs()
        jwrite(run / 'input_identities.json', identities)
        if args.check_only:
            jwrite(run / 'execution_status.json', {'status': 'INPUT_CHECKS_COMPLETED',
                   'execution_mode': 'RETRAIN_STATISTICAL', 'countries': countries,
                   'new_training_performed': False})
            print('Target and exogenous input checks passed. No models were fitted.')
            return
        stat_require_real_triad()
        REF = run / 'statistical_recalculation'
        annuals, scores = [], []
        for country in countries:
            key = country.lower()
            regenerate_statistical_baseline(key, REF, retrain=True)
            out = REF / key
            annuals.append(pd.read_csv(out / 'annual_peak_demand_scenarios.csv'))
            d = pd.read_csv(out / 'development_recursive_predictions.csv')
            h = pd.read_csv(out / 'holdout_recursive_predictions.csv')
            for phase, frame in pd.concat([d,h],ignore_index=True).groupby('phase'):
                for model, col in [('LSTM','pred_LSTM'),('LGBM','pred_LGBM'),
                                   ('TFT','pred_TFT'),('Ensemble','pred_ENS')]:
                    result = metrics(frame['y_true'].to_numpy(float), frame[col].to_numpy(float))
                    scores.append({'country':country,'phase':phase,'model':model,**result})
        annual = pd.concat(annuals, ignore_index=True)
        annual.to_csv(run / 'annual_statistical_peak_demand_MW.csv', index=False)
        pd.DataFrame(scores).to_csv(run / 'statistical_model_metrics.csv', index=False)
        pd.DataFrame(CHECKS).to_csv(run / 'verification_checks.csv', index=False)
        jwrite(run / 'resolved_configuration.json', {'statistical':cfg,'countries':countries,
               'physical_scenarios':False})
        jwrite(run / 'environment.json', stat_environment_versions().to_dict('records'))
        jwrite(run / 'execution_status.json', {'status':'COMPLETED',
               'execution_mode':'RETRAIN_STATISTICAL','countries':countries,
               'all_checks_passed':all(row['passed'] for row in CHECKS),
               'same_verified_release_results':False,'retrained_LSTM_LightGBM_TFT':True,
               'physical_components_recalculated':False,'package_version':VERSION,
               'elapsed_seconds':time.monotonic()-started})
        (run / 'REPORT.txt').write_text('NEW STATISTICAL EXPERIMENT\nCountries: '+', '.join(countries)+
            '\nAll three learners were fitted separately for each country.\n'
            'These are statistical-baseline results, not physical scenario totals.\n',encoding='utf-8')
        shutil.copy2(Path(__file__),run / 'code_used.py.txt')
        (ROOT / 'LATEST_RUN.txt').write_text(str(run)+'\n',encoding='utf-8')
        package_results(run, ROOT / 'RESULTS_TO_SHARE.zip')
        print(f'Statistical experiment completed: {run}')
    except (Exception, KeyboardInterrupt) as exc:
        detail = traceback.format_exc()
        jwrite(run / 'execution_status.json', {'status':'FAILED', 'error':str(exc),
               'execution_mode':'RETRAIN_STATISTICAL', 'all_checks_passed':False})
        (run / 'ERROR.txt').write_text(detail,encoding='utf-8')
        package_results(run, ROOT / 'RESULTS_TO_SHARE.zip')
        raise


def main(argv: list[str] | None = None) -> None:
    global REF, EXECUTION_MODE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--retrain', action='store_true', help='Train all three models again; this does not guarantee the archived release values.')
    parser.add_argument('--check-only', action='store_true', help='Check inputs and small accounting tests without running the full experiment.')
    parser.add_argument('--output', type=Path, default=None, help='Output directory for this run.')
    parser.add_argument('--countries', nargs='+', default=None, help='Country identifiers from the input headers.')
    parser.add_argument('--statistical-only', action='store_true', help='Fit only the statistical baseline for the selected countries.')
    args = parser.parse_args(argv)
    retrain = args.retrain or MODE == 'RETRAIN'
    if MODE not in ['REPRODUCE_RELEASE', 'RETRAIN']:
        raise ValueError('MODE must be REPRODUCE_RELEASE or RETRAIN.')
    EXECUTION_MODE = 'RETRAIN' if retrain else 'REPRODUCE_RELEASE'
    countries = [c.upper() for c in (args.countries or COUNTRIES)]
    if not countries or len(set(countries)) != len(countries):
        raise ValueError('Specify at least one country, without duplicates.')
    if any(not re.fullmatch(r'[A-Z][A-Z0-9_]*', c) for c in countries):
        raise ValueError('Country identifiers may contain letters, digits and underscores.')
    statistical_only = args.statistical_only or STATISTICAL_ONLY
    if statistical_only:
        if not retrain:
            raise ValueError('Use --retrain with --statistical-only, or set MODE to RETRAIN.')
        run_statistical_experiment(args, countries)
        return
    if set(countries) != {'SPAIN', 'FRANCE'}:
        raise ValueError('The supplied physical scenario configuration covers Spain and France. '
                         'For other countries, use --retrain --statistical-only with their demand and exogenous inputs.')
    stamp = datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')
    run = (args.output.resolve() if args.output is not None else ROOT / 'results' / stamp)
    if run.exists() and any(run.iterdir()):
        raise ValueError(f'The output folder already contains files: {run}')
    run.mkdir(parents=True, exist_ok=True)
    CHECKS.clear()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(run / 'execution.log', encoding='utf-8')], force=True)
    logging.getLogger('fontTools').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    started = time.monotonic()
    jwrite(run / 'execution_status.json', {'status': 'RUNNING', 'mode': EXECUTION_MODE})
    try:
        identities = verify_inputs() if not retrain else record_current_inputs()
        unit_tests()
        for country in ['spain', 'france']:
            export_observed_inputs(country, run, jload(ROOT / 'data/statistical_configuration.json')['config'])
        if args.check_only:
            pd.DataFrame(CHECKS).to_csv(run / 'verification_checks.csv', index=False)
            jwrite(run / 'execution_status.json', {'status': 'INPUT_CHECKS_COMPLETED', 'n_checks': len(CHECKS)})
            print(f'Input and accounting checks passed: {len(CHECKS)}. No new forecasts were calculated.')
            return
        stat_load_probability_stack()
        if retrain:
            for country in countries:
                preflight_country(country, jload(ROOT / 'data/statistical_configuration.json')['config'])
            stat_require_real_triad()
        REF = run / 'statistical_recalculation'
        for country in ['spain', 'france']:
            logging.info('Statistical baseline and simulation: %s', country)
            regenerate_statistical_baseline(country, REF, retrain)
        config = jload(DATA / 'configuration.json')
        scenarios = jload(DATA / 'scenario_protocol.json')['scenarios']
        stocks = pd.read_csv(DATA / 'passenger_car_stocks.csv')
        projects = pd.read_csv(DATA / 'project_capacity_knots.csv')
        capbases = pd.read_csv(DATA / 'capacity_base_assumptions.csv')
        all_annual, all_metrics = [], []
        for country, cconf in config['countries'].items():
            logging.info('Physical accounting and 23 scenarios: %s', country)
            a, span, met = run_country(country, cconf, config, scenarios, stocks, projects, capbases, run)
            all_annual.append(a)
            all_metrics.append(met)
        annual = pd.concat(all_annual, ignore_index=True)
        write_planning_tables(run, annual, scenarios, config['reference_configuration'])
        pd.concat(all_metrics, ignore_index=True).to_csv(run / 'summary_statistical_benchmarks.csv', index=False)
        if not retrain:
            verify_scenarios(run)
        logging.info('Manuscript tables and figures')
        export_article_tables(run)
        generate_article_figures(run)
        shutil.copytree(DATA, run / 'inputs_used')
        jwrite(run / 'resolved_configuration.json', {'physical': config, 'statistical': jload(ROOT / 'data/statistical_configuration.json'), 'execution_mode': EXECUTION_MODE})
        jwrite(run / 'input_identities.json', identities)
        write_final_report(run, time.monotonic() - started, retrain)
    except (Exception, KeyboardInterrupt) as exc:
        detail = traceback.format_exc()
        jwrite(run / 'execution_status.json', {'status': 'FAILED', 'execution_mode': EXECUTION_MODE,
                                               'error': str(exc), 'traceback': detail, 'all_checks_passed': False})
        (run / 'ERROR.txt').write_text(detail, encoding='utf-8')
        pd.DataFrame(CHECKS).to_csv(run / 'verification_checks.csv', index=False)
        package_results(run, ROOT / 'RESULTS_TO_SHARE.zip')
        print('\nThe run stopped. Diagnostics are in RESULTS_TO_SHARE.zip.')
        raise


if __name__ == '__main__':
    main()
