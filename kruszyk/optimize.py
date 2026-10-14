"""Scenario experiments: fleet sizing, fragmentation trade-off, scenario comparison."""
from __future__ import annotations

import copy

import pandas as pd

from .config import Scenario
from .sim import simulate


def kpi_row(k: dict) -> dict:
    """Flat, numeric subset of the KPIs for tables and charts."""
    ex = k["excavators"]
    return {
        "throughput_tph": k["throughput_tph"],
        "crushed_tph": k["crushed_tph"],
        "hauled_t": k["hauled_t"],
        "cost_per_t": k["cost_per_t"],
        "cost_total": k["cost_total"],
        "goal_t": k.get("goal_t") or 0.0,
        "objective": k.get("objective", k["cost_per_t"]),
        "goal_pct": k.get("goal_pct"),
        "excavator_utilization": sum(e["utilization"] for e in ex) / len(ex),
        "crusher_utilization": k["crusher_utilization"],
        "avg_queue_per_cycle_min": k["avg_queue_per_cycle_min"],
        "truck_queue_crusher_h": k["truck_queue_crusher_h"],
        "truck_queue_excavator_h": k["truck_queue_excavator_h"],
        "fuel_l_per_t": k["fuel_l_per_t"],
        "stockpile_end_t": k["stockpile_end_t"],
        "feed_dev_pp": k.get("feed_dev_pp", 0.0),
        "parked_excavator_h": k.get("parked_excavator_h", 0.0),
        "parked_truck_h": k.get("parked_truck_h", 0.0),
        "topup_skips": k.get("topup_skips", 0),
        "bay_wait_h": k.get("bay_wait_h", 0.0),
        "spot_exit_h": k.get("spot_exit_h", 0.0),
        **{f"koszt_{name}": v for name, v in k["cost_breakdown"].items()},
    }


def replicate(sc: Scenario, replications: int) -> dict:
    """Mean KPIs over several shifts (different weather, breakdowns, noise)."""
    rows = [kpi_row(simulate(sc, seed=sc.seed + 1000 * r).kpi) for r in range(replications)]
    return pd.DataFrame(rows).mean(numeric_only=True).round(3).to_dict()


def sweep_trucks(sc: Scenario, n_min: int, n_max: int, replications: int = 5,
                 dispatch: str | None = None) -> pd.DataFrame:
    rows = []
    for n in range(n_min, n_max + 1):
        variant = sc.with_truck_count(n)
        if dispatch:
            variant.dispatch = dispatch
        rows.append({"trucks": n, **replicate(variant, replications)})
    return pd.DataFrame(rows)


def sweep_fragmentation(sc: Scenario, values: list[float], replications: int = 3) -> pd.DataFrame:
    """Same fleet, all blasts with a given fragmentation: blasting cost vs digging/crushing cost."""
    rows = []
    for f in values:
        variant = copy.deepcopy(sc)
        for b in variant.blasts:
            b.fragmentation = f
        rows.append({"fragmentation": f, **replicate(variant, replications)})
    return pd.DataFrame(rows)


def recommend(summary: pd.DataFrame, min_tph: float | None = None) -> pd.Series:
    """Cheapest fleet per tonne; optionally the cheapest one meeting a throughput target."""
    df = summary.dropna(subset=["cost_per_t"])
    if min_tph is not None:
        ok = df[df["throughput_tph"] >= min_tph]
        if len(ok):
            df = ok
    return df.loc[df["cost_per_t"].idxmin()]
