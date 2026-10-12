"""ML on (synthetic) telemetry: segment travel time model and ETA to the excavator.

The same code will train on real geofence data from the phone app; only the source changes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit

from .config import Scenario
from .sim import simulate

NUMERIC = ["loaded", "payload_t", "length_m", "grade_pct", "speed_limit_kmh", "rain", "hour_of_shift", "pace"]
CATEGORICAL = ["truck"]


def generate_dataset(sc: Scenario, shifts: int) -> pd.DataFrame:
    """Simulate many shifts (different weather/drivers draw per seed) and stack segment records."""
    frames = []
    for s in range(shifts):
        seg = simulate(sc, seed=sc.seed + s).segments
        seg["shift"] = s
        frames.append(seg)
    return pd.concat(frames, ignore_index=True)


def _features(df: pd.DataFrame, categories: dict[str, list] | None = None) -> pd.DataFrame:
    X = df[NUMERIC].copy()
    for c in CATEGORICAL:
        cats = categories[c] if categories else sorted(df[c].unique())
        X[c] = pd.Categorical(df[c], categories=cats).codes
    return X


@dataclass
class TravelTimeModel:
    model: HistGradientBoostingRegressor
    categories: dict[str, list]

    def predict_s(self, df: pd.DataFrame) -> np.ndarray:
        # model learns pace (s per metre) so it extrapolates to segments of any length
        return self.model.predict(_features(df, self.categories)) * df["length_m"].to_numpy()

    def eta_to_excavator_s(self, sc: Scenario, truck: str, excavator: str, segment_index: int,
                           progress: float, rain: bool, hour_of_shift: float) -> float:
        """ETA of an empty truck driving to `excavator`, currently `progress` (0..1) through
        segment `segment_index` of the reversed route. This is what the operator's phone shows."""
        ex = next(e for e in sc.excavators if e.id == excavator)
        segs = list(reversed(sc.routes[ex.route].segments))
        rows = [{"truck": truck, "loaded": 0, "payload_t": 0.0, "length_m": s.length_m,
                 "grade_pct": -s.grade_pct, "speed_limit_kmh": s.speed_limit_kmh,
                 "rain": int(rain), "hour_of_shift": hour_of_shift, "pace": 1.0} for s in segs[segment_index:]]
        t = self.predict_s(pd.DataFrame(rows))
        t[0] *= 1 - progress
        return float(t.sum())


def train_travel_model(data: pd.DataFrame, seed: int = 0) -> tuple[TravelTimeModel, dict]:
    """Train and evaluate on held-out shifts. Baseline = physics model without learned effects."""
    split = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    train_idx, test_idx = next(split.split(data, groups=data["shift"]))
    train, test = data.iloc[train_idx], data.iloc[test_idx]
    categories = {c: sorted(data[c].unique()) for c in CATEGORICAL}
    cat_mask = [c in CATEGORICAL for c in NUMERIC + CATEGORICAL]

    reg = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08,
                                        categorical_features=cat_mask, random_state=seed)
    reg.fit(_features(train, categories), train["duration_s"] / train["length_m"])
    model = TravelTimeModel(reg, categories)

    pred = model.predict_s(test)
    metrics = {
        "train_rows": len(train), "test_rows": len(test),
        "mae_model_s": round(mean_absolute_error(test["duration_s"], pred), 2),
        "mae_physics_baseline_s": round(mean_absolute_error(test["duration_s"], test["physics_s"]), 2),
        "mean_segment_s": round(float(test["duration_s"].mean()), 1),
    }
    return model, metrics
