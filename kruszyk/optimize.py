"""Fleet sizing by simulation: sweep the number of trucks, replicate, compare."""
from __future__ import annotations

import pandas as pd

from .config import Scenario
from .sim import simulate


def sweep_trucks(sc: Scenario, n_min: int, n_max: int, replications: int = 5,
                 dispatch: str | None = None) -> pd.DataFrame:
    rows = []
    for n in range(n_min, n_max + 1):
        variant = sc.with_truck_count(n)
        if dispatch:
            variant.dispatch = dispatch
        for r in range(replications):
            k = simulate(variant, seed=sc.seed + 1000 * r).kpi
            ex_util = sum(e["utilization"] for e in k["excavators"]) / len(k["excavators"])
            rows.append({
                "trucks": n, "replication": r, "throughput_tph": k["throughput_tph"],
                "hauled_t": k["hauled_t"], "cost_per_t": k["cost_per_t"],
                "excavator_utilization": ex_util, "crusher_utilization": k["crusher_utilization"],
                "avg_queue_per_cycle_min": k["avg_queue_per_cycle_min"],
            })
    df = pd.DataFrame(rows)
    return df.groupby("trucks").mean(numeric_only=True).drop(columns="replication").round(3).reset_index()


def recommend(summary: pd.DataFrame, min_tph: float | None = None) -> pd.Series:
    """Cheapest fleet per tonne; optionally the cheapest one meeting a throughput target."""
    df = summary.dropna(subset=["cost_per_t"])
    if min_tph is not None:
        ok = df[df["throughput_tph"] >= min_tph]
        if len(ok):
            df = ok
    return df.loc[df["cost_per_t"].idxmin()]
