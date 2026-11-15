"""Transit-delay forecaster (Random Forest with calibrated uncertainty).

Eval 2 used a Random Forest that returned one number. The same model family is
kept, used a little more carefully:

1. Model selection: Linear Regression (baseline), Random Forest and Gradient
   Boosting are compared once with 5-fold cross-validation (MAE, RMSE, R^2).
2. Uncertainty: the spread of the forest's trees gives a per-trip standard
   deviation; a held-out calibration split rescales it (normalised
   split-conformal) so the P10-P90 band covers ~80% of real trips.
3. Decision use: agents receive P(on time) = P(delay <= slack), so a driver
   with a slightly longer mean but a tighter band can win.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.config import HISTORY_CSV, MODEL_CARD_PATH, MODEL_PATH

NUMERIC = ["distance_km", "traffic_speed_factor", "rainfall_mm", "hour_of_day", "weight_kg"]
DERIVED = ["is_peak", "is_two_wheeler", "congestion_km"]
CATEGORICAL = ["vehicle_type", "pickup_zone", "dropoff_zone"]
TARGET = "delay_minutes"
SEED = 42
Z80 = 1.2816  # two-sided 80% normal quantile


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["is_peak"] = out["hour_of_day"].isin([8, 9, 17, 18, 19]).astype(int)
    out["is_two_wheeler"] = out["vehicle_type"].isin(["Bike", "Electric Scooter"]).astype(int)
    out["congestion_km"] = out["distance_km"] * (1 - out["traffic_speed_factor"])
    return out


def _preprocess() -> ColumnTransformer:
    return ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)], remainder="passthrough")


def candidate_models() -> dict[str, Pipeline]:
    return {
        "linear_regression": Pipeline([("prep", _preprocess()), ("model", LinearRegression())]),
        "random_forest": Pipeline([("prep", _preprocess()), ("model", RandomForestRegressor(
            n_estimators=300, min_samples_leaf=4, max_features=0.6, random_state=SEED, n_jobs=-1))]),
        "gradient_boosting": Pipeline([("prep", _preprocess()), ("model", GradientBoostingRegressor(
            n_estimators=300, max_depth=3, learning_rate=0.05, random_state=SEED))]),
    }


def _tree_stats(pipe: Pipeline, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    Xt = pipe.named_steps["prep"].transform(X)
    per_tree = np.stack([t.predict(Xt) for t in pipe.named_steps["model"].estimators_])
    return per_tree.mean(axis=0), per_tree.std(axis=0)


def train(history_csv=HISTORY_CSV) -> dict:
    df = add_features(pd.read_csv(history_csv))
    features = NUMERIC + DERIVED + CATEGORICAL
    X, y = df[features], df[TARGET]

    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    comparison = {}
    for name, pipe in candidate_models().items():
        s = cross_validate(pipe, X, y, cv=cv, scoring=("neg_mean_absolute_error", "neg_root_mean_squared_error", "r2"))
        comparison[name] = {
            "cv_mae": round(-s["test_neg_mean_absolute_error"].mean(), 3),
            "cv_rmse": round(-s["test_neg_root_mean_squared_error"].mean(), 3),
            "cv_r2": round(s["test_r2"].mean(), 3),
            "cv_r2_std": round(s["test_r2"].std(), 3),
        }

    # The forest is kept even though boosting is marginally better on MAE: only
    # the forest exposes per-tree predictions, which the uncertainty band needs.
    X_fit, X_hold, y_fit, y_hold = train_test_split(X, y, test_size=0.25, random_state=SEED)
    X_train, X_cal, y_train, y_cal = train_test_split(X_fit, y_fit, test_size=0.25, random_state=SEED)
    rf = candidate_models()["random_forest"].fit(X_train, y_train)

    mean_cal, std_cal = _tree_stats(rf, X_cal)
    scale = float(np.quantile(np.abs(y_cal.to_numpy() - mean_cal) / (std_cal + 1e-6), 0.80))

    mean_h, std_h = _tree_stats(rf, X_hold)
    half = scale * std_h
    holdout = {
        "mae": round(mean_absolute_error(y_hold, mean_h), 3),
        "rmse": round(math.sqrt(mean_squared_error(y_hold, mean_h)), 3),
        "r2": round(r2_score(y_hold, mean_h), 3),
        "p10_p90_coverage": round(float(np.mean((y_hold >= mean_h - half) & (y_hold <= mean_h + half))), 3),
        "mean_band_width_mins": round(float(np.mean(2 * half)), 2),
    }
    perm = permutation_importance(rf, X_hold, y_hold, n_repeats=8, random_state=SEED, n_jobs=-1)
    importance = sorted(
        ({"feature": f, "importance": round(float(v), 3)} for f, v in zip(features, perm.importances_mean)),
        key=lambda d: d["importance"], reverse=True,
    )
    joblib.dump({"pipeline": rf, "features": features, "conformal_scale": scale}, MODEL_PATH)
    card = {
        "model": "RandomForestRegressor(n_estimators=300, min_samples_leaf=4, max_features=0.6)",
        "rows": int(len(df)),
        "split": {"train": int(len(X_train)), "calibration": int(len(X_cal)), "holdout": int(len(X_hold))},
        "cv_comparison": comparison,
        "holdout": holdout,
        "conformal_scale": round(scale, 3),
        "permutation_importance": importance,
    }
    MODEL_CARD_PATH.write_text(json.dumps(card, indent=2))
    _load.cache_clear()
    return card


@lru_cache(maxsize=1)
def _load() -> dict:
    if not MODEL_PATH.exists():
        train()
    # joblib unpickles; safe here because MODEL_PATH is only ever written by train() above.
    return joblib.load(MODEL_PATH)


@dataclass(frozen=True)
class DelayForecast:
    mean_mins: float
    p10_mins: float
    p90_mins: float
    p_on_time: float | None

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def forecast(distance_km: float, traffic_speed_factor: float, rainfall_mm: float, hour_of_day: int,
             weight_kg: float, vehicle_type: str, pickup_zone: str, dropoff_zone: str,
             slack_mins: float | None = None) -> DelayForecast:
    bundle = _load()
    row = add_features(pd.DataFrame([{
        "distance_km": distance_km, "traffic_speed_factor": traffic_speed_factor,
        "rainfall_mm": rainfall_mm, "hour_of_day": hour_of_day, "weight_kg": weight_kg,
        "vehicle_type": vehicle_type, "pickup_zone": pickup_zone, "dropoff_zone": dropoff_zone,
    }]))[bundle["features"]]
    mean, std = _tree_stats(bundle["pipeline"], row)
    m = max(float(mean[0]), 0.0)
    half = bundle["conformal_scale"] * float(std[0])
    p_on_time = None
    if slack_mins is not None:
        sigma = max(half / Z80, 0.5)
        p_on_time = round(0.5 * (1 + math.erf((slack_mins - m) / (sigma * math.sqrt(2)))), 3)
    return DelayForecast(round(m, 1), round(max(m - half, 0.0), 1), round(m + half, 1), p_on_time)


if __name__ == "__main__":
    print(json.dumps(train(), indent=2))
