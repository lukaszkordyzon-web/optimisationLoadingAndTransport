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


ADVISOR_PARAMS = ["zapas przyjazdu pod koparkę [s]", "maks. zwolnienie [%]", "zwalnianie przed kruszarką [×]"]


def decode_advisor(x) -> dict:
    """Unbounded learned numbers -> sensible advisor settings. x = 0 gives the hand-made rules
    (no margin, at most 30 % slower, crusher pacing 1×) – learning starts from them."""
    x = np.asarray(x, dtype=float)
    sig = lambda v: 1 / (1 + np.exp(-v))  # noqa: E731
    return {"arrival_margin_s": float(120 * np.tanh(x[0])),
            "max_slowdown": float(0.6 * sig(x[1])),
            "crusher_pacing": float(2 * sig(x[2]))}


def _episode_scenario(sc: Scenario, policy: str, theta=None, shift_hours: float | None = None,
                      advisor: bool = False) -> Scenario:
    """theta = dispatch weights (+ 3 advisor numbers when they are learned too)."""
    v = copy.deepcopy(sc)
    v.dispatch = policy
    n = len(FEATURE_NAMES)
    v.rl_weights = [float(w) for w in theta[:n]] if theta is not None else None
    # today's practice and the fixed heuristic run without real-time advice; the agent may use it
    v.advisor.enabled = bool(advisor) and policy == "rl"
    if v.advisor.enabled and theta is not None and len(theta) > n:
        for key, val in decode_advisor(theta[n:]).items():
            setattr(v.advisor, key, val)
    if shift_hours:
        v.shift_hours = shift_hours
    return v


def evaluate(sc: Scenario, policy: str, seeds: list[int], theta=None, shift_hours: float | None = None,
             advisor: bool = False) -> dict:
    """Mean KPIs of a policy over the given simulated shifts."""
    v = _episode_scenario(sc, policy, theta, shift_hours, advisor)
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
    advisor: dict | None = None   # learned advisor settings (None when not learned)
    baseline_by_iteration: dict = field(default_factory=dict)


def train(sc: Scenario, iterations: int = 10, population: int = 12, elite_frac: float = 0.25,
          episodes: int = 2, shift_hours: float | None = 4.0, eval_episodes: int = 4, seed: int = 0,
          callback: Callable[[int, int, dict], None] | None = None, advisor: bool = False,
          learn_advisor: bool = True) -> TrainResult:
    """Learn the dispatch weights; with `advisor` the trucks/operators get real-time advice and, with
    `learn_advisor`, its three settings are learned together with the dispatch (starting from the rules)."""
    rng = np.random.default_rng(seed)
    n = len(FEATURE_NAMES)
    learn_adv = advisor and learn_advisor
    dim = n + (len(ADVISOR_PARAMS) if learn_adv else 0)
    mu = np.zeros(dim)  # start: an untrained dispatch policy + the hand-made advice rules
    sigma = np.concatenate([np.full(n, 1.5), np.full(dim - n, 1.0)])
    n_elite = max(2, int(round(population * elite_frac)))
    history = []

    def record(it: int, seeds: list[int], pop_costs: list[float] | None):
        policy = evaluate(sc, "rl", seeds, mu, shift_hours, advisor)
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
            **{f"w{i}": float(x) for i, x in enumerate(mu[:n])},
            **({k: v for k, v in decode_advisor(mu[n:]).items()} if learn_adv else {}),
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
            r = evaluate(sc, "rl", seeds, w, shift_hours, advisor)
            scores.append((r["cost_per_t"], k, r))
        scores.sort(key=lambda x: x[0])
        elite = np.array([cand[k] for _, k, _ in scores[:n_elite]])
        mu = elite.mean(axis=0)
        sigma = elite.std(axis=0) + max(0.05, 0.5 * (1 - it / max(iterations - 1, 1)))  # exploration noise decays
        record(it + 1, seeds, [c for c, _, _ in scores])

    # final, fair comparison on fresh full shifts
    eval_seeds = [sc.seed + 90_000 + e for e in range(eval_episodes)]
    comp = []
    for policy in ("natural", "fixed", "dynamic"):
        comp.append({"polityka": POLICIES[policy], **evaluate(sc, policy, eval_seeds)})
    if learn_adv:
        rules = np.concatenate([mu[:n], np.zeros(dim - n)])  # same dispatch, advice by the hand-made rules
        comp.append({"polityka": "polityka RL + zalecenia wg reguł", **evaluate(sc, "rl", eval_seeds, rules, None, True)})
        comp.append({"polityka": "polityka RL + zalecenia wyuczone", **evaluate(sc, "rl", eval_seeds, mu, None, True)})
    else:
        comp.append({"polityka": POLICIES["rl"] + (" + zalecenia" if advisor else ""),
                     **evaluate(sc, "rl", eval_seeds, mu, None, advisor)})
    return TrainResult(weights=[float(x) for x in mu[:n]], history=pd.DataFrame(history),
                       comparison=pd.DataFrame(comp), advisor=decode_advisor(mu[n:]) if learn_adv else None)


def explain_advisor(adv: dict) -> list[str]:
    """Learned advice settings in plain words."""
    m, sd, cp = adv["arrival_margin_s"], adv["max_slowdown"], adv["crusher_pacing"]
    when = (f"ok. {abs(m):.0f} s **przed** zwolnieniem koparki (zapas na losowość jazdy)" if m > 5 else
            f"ok. {abs(m):.0f} s **po** zwolnieniu koparki" if m < -5 else "dokładnie, gdy koparka się zwolni")
    return [f"🚚 Kierowca ma dojeżdżać pod koparkę {when}.",
            f"🐢 Zalecenie może spowolnić jazdę najwyżej o **{100 * sd:.0f} %** (nigdy ponad limit prędkości).",
            ("🏭 Przed kruszarką: " + ("nie zwalniać – kosz nadąża." if cp < 0.15 else
                                       f"zwalniać, gdy grozi kolejka (siła {cp:.1f}× względem reguły)."))]


def explain(weights: list[float]) -> pd.DataFrame:
    """Learned weights in words: positive = the agent avoids excavators with more of this."""
    rows = []
    for name, w in zip(FEATURE_NAMES, weights):
        effect = "unika" if w > 0.05 else ("preferuje" if w < -0.05 else "obojętne")
        rows.append({"cecha": name, "waga": round(w, 3), "wpływ": effect})
    return pd.DataFrame(rows)
