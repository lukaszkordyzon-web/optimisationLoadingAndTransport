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
    "niedobór usypu względem udziału w nadawie [kursy]",
]
EXCAVATOR_PARAMS = ["min. wozideł na koparkę", "ostatnia łyżka: min. wypełnienie", "ostatnia łyżka: min. kolejka"]
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


def decode_excavators(x) -> dict:
    """Unbounded learned numbers -> excavator management. x = 0 is today's practice:
    all excavators work, every truck is filled to the payload."""
    x = np.asarray(x, dtype=float)
    return {"min_trucks_per_excavator": float(3 * max(0.0, np.tanh(x[0]))),
            "topup_min_fill": float(1 - 0.25 * max(0.0, np.tanh(x[1]))),
            "topup_min_queue": float(3 / (1 + np.exp(-x[2])))}


def split_theta(theta, learn_excavators: bool, learn_advisor: bool):
    """theta = [dispatch weights | excavator management (3) | advice settings (3)]."""
    n = len(FEATURE_NAMES)
    theta = np.asarray(theta, dtype=float)
    ex = theta[n:n + 3] if learn_excavators else None
    adv_start = n + (3 if learn_excavators else 0)
    adv = theta[adv_start:adv_start + 3] if learn_advisor else None
    return theta[:n], ex, adv


def _episode_scenario(sc: Scenario, policy: str, theta=None, shift_hours: float | None = None,
                      advisor: bool = False, learn_excavators: bool = False, learn_advisor: bool = False) -> Scenario:
    v = copy.deepcopy(sc)
    v.dispatch = policy
    # today's practice and the fixed heuristic: no real-time advice, all excavators, full trucks
    v.advisor.enabled = bool(advisor) and policy == "rl"
    if theta is not None:
        w, ex, adv = split_theta(theta, learn_excavators, learn_advisor and v.advisor.enabled)
        v.rl_weights = [float(x) for x in w]
        if ex is not None and policy == "rl":
            for key, val in decode_excavators(ex).items():
                setattr(v.excavator_policy, key, val)
        if adv is not None:
            for key, val in decode_advisor(adv).items():
                setattr(v.advisor, key, val)
    else:
        v.rl_weights = None
    if shift_hours:
        v.shift_hours = shift_hours
    return v


def evaluate(sc: Scenario, policy: str, seeds: list[int], theta=None, shift_hours: float | None = None,
             advisor: bool = False, learn_excavators: bool = False, learn_advisor: bool = False) -> dict:
    """Mean KPIs of a policy over the given simulated shifts."""
    v = _episode_scenario(sc, policy, theta, shift_hours, advisor, learn_excavators, learn_advisor)
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
    excavators: dict | None = None  # learned excavator management (None when not learned)
    baseline_by_iteration: dict = field(default_factory=dict)


def train(sc: Scenario, iterations: int = 10, population: int = 12, elite_frac: float = 0.25,
          episodes: int = 2, shift_hours: float | None = 4.0, eval_episodes: int = 4, seed: int = 0,
          callback: Callable[[int, int, dict], None] | None = None, advisor: bool = False,
          learn_advisor: bool = True, learn_excavators: bool = True) -> TrainResult:
    """Learn the dispatch weights; with `advisor` the trucks/operators get real-time advice and, with
    `learn_advisor`, its three settings are learned together with the dispatch (starting from the rules)."""
    rng = np.random.default_rng(seed)
    n = len(FEATURE_NAMES)
    learn_adv = advisor and learn_advisor
    dim = n + (3 if learn_excavators else 0) + (len(ADVISOR_PARAMS) if learn_adv else 0)
    kw = dict(advisor=advisor, learn_excavators=learn_excavators, learn_advisor=learn_adv)
    mu = np.zeros(dim)  # start: an untrained dispatch policy + the hand-made advice rules
    sigma = np.concatenate([np.full(n, 1.5), np.full(dim - n, 1.0)])
    n_elite = max(2, int(round(population * elite_frac)))
    history = []

    def record(it: int, seeds: list[int], pop_costs: list[float] | None):
        policy = evaluate(sc, "rl", seeds, mu, shift_hours, **kw)
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
            **(decode_excavators(split_theta(mu, learn_excavators, learn_adv)[1]) if learn_excavators else {}),
            **(decode_advisor(split_theta(mu, learn_excavators, learn_adv)[2]) if learn_adv else {}),
            "feed_dev_pp": policy.get("feed_dev_pp", 0.0),
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
            r = evaluate(sc, "rl", seeds, w, shift_hours, **kw)
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
    if learn_excavators or learn_adv:
        # the same learned dispatch, but excavators and advice as today / by the rules
        only_dispatch = np.concatenate([mu[:n], np.zeros(dim - n)])
        comp.append({"polityka": "polityka RL – sam przydział",
                     **evaluate(sc, "rl", eval_seeds, only_dispatch, **kw)})
    comp.append({"polityka": "polityka RL – pełna" if (learn_excavators or learn_adv) else
                 POLICIES["rl"] + (" + zalecenia" if advisor else ""), **evaluate(sc, "rl", eval_seeds, mu, **kw)})
    _, ex_x, adv_x = split_theta(mu, learn_excavators, learn_adv)
    return TrainResult(weights=[float(x) for x in mu[:n]], history=pd.DataFrame(history),
                       comparison=pd.DataFrame(comp),
                       advisor=decode_advisor(adv_x) if learn_adv else None,
                       excavators=decode_excavators(ex_x) if learn_excavators else None)


def explain_excavators(p: dict) -> list[str]:
    """Learned excavator management in plain words."""
    out = []
    if p["min_trucks_per_excavator"] < 0.3:
        out.append("⛏️ Wszystkie koparki pracują – odstawianie się nie opłaca.")
    else:
        out.append(f"⛏️ Na każdą pracującą koparkę ma przypadać co najmniej **{p['min_trucks_per_excavator']:.1f}** "
                   "wolnych wozideł; gdy jest ich mniej, koparka z najdłuższym cyklem wozideł zostaje odstawiona "
                   "(sprawdzane co 30 min).")
    if p["topup_min_fill"] > 0.98:
        out.append("🪣 Wozidła są zawsze ładowane do pełna.")
    else:
        out.append(f"🪣 Ostatnią łyżkę można pominąć, gdy wozidło jest pełne w **{100 * p['topup_min_fill']:.0f} %** "
                   f"lub więcej, a w kolejce czeka co najmniej **{p['topup_min_queue']:.1f}** wozidło.")
    return out


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
