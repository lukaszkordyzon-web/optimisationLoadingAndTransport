"""Reinforcement learning of the dispatch policy.

The agent decides, every time a truck has dumped, which excavator it should go to. A policy scores
each allowed excavator with a linear function of what the system knows at that moment
(Simulation.features) and picks the lowest score. The weights are learned by policy search with the
cross-entropy method (CEM): sample policies, run them on simulated shifts (episodes), keep the best
ones, move the sampling distribution towards them, repeat. Reward = -cost per tonne of the shift.

Candidates in one iteration are evaluated on the same random shifts (common random numbers), so they
are compared fairly; every iteration uses new shifts, so the policy does not overfit to one day.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from .config import Scenario
from .optimize import kpi_row
from .sim import simulate

FEATURE_NAMES = [
    "przewidywane czekanie pod koparką [min]",
    "dojazd pusty [min]",
    "odwóz pełny do kruszarki [min]",
    "podstawienie + załadunek [min]",
    "pozostały urobek (udział)",
    "wozidła już wysłane do koparki",
    "„własna” koparka kierowcy",
]
POLICIES = {"natural": "naturalny (obecna praktyka)", "fixed": "stały do koparki",
            "dynamic": "heurystyka systemu", "rl": "polityka RL"}


def _episode_scenario(sc: Scenario, policy: str, weights=None, shift_hours: float | None = None) -> Scenario:
    v = copy.deepcopy(sc)
    v.dispatch = policy
    v.rl_weights = list(map(float, weights)) if weights is not None else None
    if shift_hours:
        v.shift_hours = shift_hours
    return v


def evaluate(sc: Scenario, policy: str, seeds: list[int], weights=None, shift_hours: float | None = None) -> dict:
    """Mean KPIs of a policy over the given simulated shifts."""
    v = _episode_scenario(sc, policy, weights, shift_hours)
    rows = [kpi_row(simulate(v, seed=s).kpi) for s in seeds]
    df = pd.DataFrame(rows)
    out = df.mean(numeric_only=True).to_dict()
    out["cost_per_t"] = float(df["cost_per_t"].fillna(1e6).mean())  # a policy that hauls nothing is the worst
    return out


@dataclass
class TrainResult:
    weights: list[float]
    history: pd.DataFrame
    comparison: pd.DataFrame
    baseline_by_iteration: dict = field(default_factory=dict)


def train(sc: Scenario, iterations: int = 10, population: int = 12, elite_frac: float = 0.25,
          episodes: int = 2, shift_hours: float | None = 4.0, eval_episodes: int = 4, seed: int = 0,
          callback: Callable[[int, int, dict], None] | None = None) -> TrainResult:
    rng = np.random.default_rng(seed)
    dim = len(FEATURE_NAMES)
    mu, sigma = np.zeros(dim), np.full(dim, 1.5)  # start: an untrained (arbitrary) policy
    n_elite = max(2, int(round(population * elite_frac)))
    history = []

    def record(it: int, seeds: list[int], pop_costs: list[float] | None):
        policy = evaluate(sc, "rl", seeds, mu, shift_hours)
        natural = evaluate(sc, "natural", seeds, None, shift_hours)
        row = {
            "iteracja": it,
            "koszt_populacja_sredni": float(np.mean([min(c, 1e3) for c in pop_costs])) if pop_costs else None,
            "koszt_najlepszy": float(pop_costs[0]) if pop_costs else None,
            "koszt_polityka": policy["cost_per_t"],
            "koszt_naturalny": natural["cost_per_t"],
            # same simulated shifts for both -> a fair, low-noise measure of what the agent gained
            "oszczednosc_pct": 100 * (1 - policy["cost_per_t"] / natural["cost_per_t"]),
            "throughput_tph": policy["throughput_tph"],
            "fuel_l_per_t": policy["fuel_l_per_t"],
            "avg_queue_per_cycle_min": policy["avg_queue_per_cycle_min"],
            "excavator_utilization": policy["excavator_utilization"],
            "naturalny_throughput_tph": natural["throughput_tph"],
            "naturalny_fuel_l_per_t": natural["fuel_l_per_t"],
            "naturalny_avg_queue_per_cycle_min": natural["avg_queue_per_cycle_min"],
            "naturalny_excavator_utilization": natural["excavator_utilization"],
            **{f"w{i}": float(x) for i, x in enumerate(mu)},
        }
        history.append(row)
        if callback:
            callback(it, iterations, row)

    record(0, [sc.seed + 10_000 + e for e in range(episodes)], None)  # the untrained starting point
    for it in range(iterations):
        seeds = [sc.seed + 10_000 + it * 100 + e for e in range(episodes)]
        cand = mu + sigma * rng.standard_normal((population, dim))
        cand[0] = mu  # keep the current mean in the race
        scores = []
        for k, w in enumerate(cand):
            r = evaluate(sc, "rl", seeds, w, shift_hours)
            scores.append((r["cost_per_t"], k, r))
        scores.sort(key=lambda x: x[0])
        elite = np.array([cand[k] for _, k, _ in scores[:n_elite]])
        mu = elite.mean(axis=0)
        sigma = elite.std(axis=0) + max(0.05, 0.5 * (1 - it / max(iterations - 1, 1)))  # exploration noise decays
        record(it + 1, seeds, [c for c, _, _ in scores])

    # final, fair comparison on fresh full shifts
    eval_seeds = [sc.seed + 90_000 + e for e in range(eval_episodes)]
    comp = []
    for policy in ("natural", "fixed", "dynamic", "rl"):
        r = evaluate(sc, policy, eval_seeds, mu if policy == "rl" else None)
        comp.append({"polityka": POLICIES[policy], **r})
    return TrainResult(weights=[float(x) for x in mu], history=pd.DataFrame(history), comparison=pd.DataFrame(comp))


def explain(weights: list[float]) -> pd.DataFrame:
    """Learned weights in words: positive = the agent avoids excavators with more of this."""
    rows = []
    for name, w in zip(FEATURE_NAMES, weights):
        effect = "unika" if w > 0.05 else ("preferuje" if w < -0.05 else "obojętne")
        rows.append({"cecha": name, "waga": round(w, 3), "wpływ": effect})
    return pd.DataFrame(rows)
