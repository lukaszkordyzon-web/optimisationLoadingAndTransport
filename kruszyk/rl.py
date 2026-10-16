"""Reinforcement learning of the dispatch policy.

The agent decides, every time a truck has dumped, which excavator it should go to. A policy scores
each allowed excavator with a linear function of what the system knows at that moment
(Simulation.features) and picks the lowest score. The weights are learned by policy search with the
cross-entropy method (CEM): sample policies, run them on simulated shifts (episodes), keep the best
ones, move the sampling distribution towards them, repeat. Reward = -cost per tonne of the shift.

Candidates in one iteration are evaluated on the same random shifts (common random numbers), so they
are compared fairly; every iteration uses new shifts, so the policy does not overfit to one day.

With `chain` = N every episode is N shifts one after another (as in the schedule): what one shift takes out of
a pile is missing on the next one, so a policy that empties the nearest pile pays for it later.
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
    "przewidywane czekanie na mijankach [min]",
]
EXCAVATOR_PARAMS = ["min. wozideł na koparkę", "ostatnia łyżka: min. wypełnienie", "ostatnia łyżka: min. kolejka",
                    "parking wozidła: próg kolejki [min]", "parking wozidła: min. udział floty w pracy"]
N_EX = len(EXCAVATOR_PARAMS)
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
    x = np.pad(np.asarray(x, dtype=float), (0, max(0, N_EX - len(x))))
    return {"min_trucks_per_excavator": float(3 * max(0.0, np.tanh(x[0]))),
            "topup_min_fill": float(1 - 0.25 * max(0.0, np.tanh(x[1]))),
            "topup_min_queue": float(3 / (1 + np.exp(-x[2]))),
            # trucks park instead of queueing at the crusher only when x > 0 (x = 0: today, nobody parks)
            "truck_park_queue_min": float(30 * (1 - np.tanh(x[3]))) if x[3] > 0 else -1.0,
            "truck_min_working_share": float(0.25 + 0.75 / (1 + np.exp(-x[4])))}


def split_theta(theta, learn_excavators: bool, learn_advisor: bool):
    """theta = [dispatch weights | excavator and truck management (N_EX) | advice settings (3)]."""
    n = len(FEATURE_NAMES)
    theta = np.asarray(theta, dtype=float)
    ex = theta[n:n + N_EX] if learn_excavators else None
    adv_start = n + (N_EX if learn_excavators else 0)
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
        if v.goal_t:  # training on shorter shifts: the goal shrinks with them
            v.shift_goal = {**v.shift_goal, "feed_t": v.goal_t * shift_hours / sc.shift_hours}
        v.shift_hours = shift_hours
    return v


def chain_kpis(sc: Scenario, seeds: list[int], n: int) -> list[dict]:
    """KPIs of `n` shifts one after another for every seed, the scenario as it is (dispatch, policy): the piles
    carry over from shift to shift, excavators move and areas are blasted as in the schedule."""
    import datetime as dt

    from .period import PeriodPlan, run_period

    plan = PeriodPlan(start=dt.date(2026, 1, 5), days=n, shifts_per_day=1, saturdays=True, sundays=True)
    out: list[dict] = []
    for s in seeds:
        run_period(sc, plan, seed=s, kpis=out)
    return out


def evaluate(sc: Scenario, policy: str, seeds: list[int], theta=None, shift_hours: float | None = None,
             advisor: bool = False, learn_excavators: bool = False, learn_advisor: bool = False,
             chain: int = 1) -> dict:
    """Mean KPIs of a policy over the given simulated shifts (`chain` > 1: series of shifts in a row)."""
    v = _episode_scenario(sc, policy, theta, None if chain > 1 else shift_hours, advisor, learn_excavators,
                          learn_advisor)
    rows = [kpi_row(k) for k in chain_kpis(v, seeds, chain)] if chain > 1 else \
        [kpi_row(simulate(v, seed=s).kpi) for s in seeds]
    df = pd.DataFrame(rows)
    out = df.mean(numeric_only=True).to_dict()
    out.update(aggregate(rows))
    return out


def aggregate(rows: list[dict]) -> dict:
    """Cost per tonne over several shifts = all costs / all tonnes (one shift with a long crusher breakdown and
    a few hundred tonnes must not outweigh the rest, as a mean of per-shift zł/t would); the objective is that,
    made worse by the share of the shift goal not reached."""
    t = sum(r["hauled_t"] for r in rows)
    cost = sum(r["cost_total"] for r in rows)
    if t <= 0:
        return {"cost_per_t": 1e6, "objective": 1e6}  # a policy that hauls nothing is the worst
    goal = sum(r.get("goal_t") or 0.0 for r in rows)
    short = max(0.0, goal - t) / goal if goal else 0.0
    return {"cost_per_t": cost / t, "objective": cost / t * (1 + 3 * short)}



@dataclass
class TrainResult:
    weights: list[float]
    history: pd.DataFrame
    comparison: pd.DataFrame
    advisor: dict | None = None   # learned advisor settings (None when not learned)
    excavators: dict | None = None  # learned excavator management (None when not learned)
    baseline_by_iteration: dict = field(default_factory=dict)
    setup: dict = field(default_factory=dict)  # what the policy learned on (for the description)


def train(sc: Scenario, iterations: int = 10, population: int = 12, elite_frac: float = 0.25,
          episodes: int = 2, shift_hours: float | None = 4.0, eval_episodes: int = 4, seed: int = 0,
          callback: Callable[[int, int, dict], None] | None = None, advisor: bool = False,
          learn_advisor: bool = True, learn_excavators: bool = True, chain: int = 1) -> TrainResult:
    """Learn the dispatch weights; with `advisor` the trucks/operators get real-time advice and, with
    `learn_advisor`, its three settings are learned together with the dispatch (starting from the rules).
    `chain` > 1: every episode is that many full shifts in a row (piles carry over), not one separate shift."""
    chain = max(1, int(chain))
    if chain > 1:
        shift_hours = None  # depletion of the piles needs whole shifts
    rng = np.random.default_rng(seed)
    n = len(FEATURE_NAMES)
    learn_adv = advisor and learn_advisor
    dim = n + (N_EX if learn_excavators else 0) + (len(ADVISOR_PARAMS) if learn_adv else 0)
    kw = dict(advisor=advisor, learn_excavators=learn_excavators, learn_advisor=learn_adv, chain=chain)
    mu = np.zeros(dim)  # start: an untrained dispatch policy + the hand-made advice rules
    sigma = np.concatenate([np.full(n, 1.5), np.full(dim - n, 1.0)])
    n_elite = max(2, int(round(population * elite_frac)))
    history = []

    def record(it: int, seeds: list[int], pop_costs: list[float] | None):
        policy = evaluate(sc, "rl", seeds, mu, shift_hours, **kw)
        natural = evaluate(sc, "natural", seeds, None, shift_hours, chain=chain)
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
            scores.append((r["objective"], k, r))
        scores.sort(key=lambda x: x[0])
        elite = np.array([cand[k] for _, k, _ in scores[:n_elite]])
        mu = elite.mean(axis=0)
        sigma = elite.std(axis=0) + max(0.05, 0.5 * (1 - it / max(iterations - 1, 1)))  # exploration noise decays
        record(it + 1, seeds, [c for c, _, _ in scores])

    # final, fair comparison on fresh full shifts
    eval_seeds = [sc.seed + 90_000 + e for e in range(eval_episodes)]
    comp = []
    for policy in ("natural", "fixed", "dynamic"):
        comp.append({"polityka": POLICIES[policy], **evaluate(sc, policy, eval_seeds, chain=chain)})
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
                       excavators=decode_excavators(ex_x) if learn_excavators else None,
                       setup={"iterations": iterations, "population": population, "episodes": episodes,
                              "shift_hours": shift_hours, "eval_episodes": eval_episodes, "chain": chain,
                              "shifts": (iterations * (population + 2) + 2) * episodes * chain
                              + (4 + (learn_excavators or learn_adv)) * eval_episodes * chain})


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
    if p.get("truck_park_queue_min", -1.0) < 0:
        out.append("🚚 Wozidła nie są odstawiane – każde jeździ całą zmianę.")
    else:
        out.append(f"🅿️ Wozidło po wyładunku **parkuje z wyłączonym silnikiem**, gdy po następnym kursie czekałoby "
                   f"pod kruszarką dłużej niż **{p['truck_park_queue_min']:.0f} min**; co kilka minut sprawdza "
                   f"kolejkę. W pracy zostaje zawsze co najmniej **{100 * p['truck_min_working_share']:.0f} %** "
                   "wozideł.")
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


# ---------------------------------------------------------------------------------------------- driving signals
# The trucks go where the foreman sends them (as today); the system only tells the driver HOW and WHEN to drive:
# the speed (arrive when the excavator is free, slow down instead of queueing at the crusher) and when to wait
# with the engine off. The policy learns these few settings; a month with them is compared with a month without.
SIGNAL_PARAMS = ADVISOR_PARAMS + ["postój: kolejka pod kruszarką, od której czekać [min]",
                                  "postój: min. udział wozideł w pracy"]
SIGNAL_KEYS = ["arrival_margin_s", "max_slowdown", "crusher_pacing", "truck_park_queue_min", "truck_min_working_share"]
KEEP_T = 5.0  # every 1 % of tonnes below today's costs 5 % on the objective: cheaper must not mean less


def decode_signals(x) -> dict:
    """Unbounded learned numbers -> the five signal settings. x = 0: the hand-made speed rules, no waiting."""
    x = np.asarray(x, dtype=float)
    ex = decode_excavators(np.concatenate([[0.0, 0.0, 0.0], x[3:5]]))
    return decode_advisor(x[:3]) | {k: ex[k] for k in ("truck_park_queue_min", "truck_min_working_share")}


def with_signals(sc: Scenario, params: dict | None = None) -> Scenario:
    """The scenario driven with the signals (destinations as today)."""
    v = copy.deepcopy(sc)
    v.dispatch, v.advisor.enabled = "signals", True
    for k, val in (params or decode_signals(np.zeros(len(SIGNAL_KEYS)))).items():
        setattr(v.advisor if hasattr(v.advisor, k) else v.excavator_policy, k, val)
    return v


def today(sc: Scenario) -> Scenario:
    v = copy.deepcopy(sc)
    v.dispatch = "natural"
    return v


def _run(cases: list[tuple[Scenario, int]], params: dict | None) -> list[dict]:
    return [kpi_row(simulate(with_signals(s, params) if params is not None else today(s), seed=seed).kpi)
            for s, seed in cases]


def signals_objective(rows: list[dict], base: list[dict], keep: float = KEEP_T, gain: float = 0.0) -> float:
    """Cost per tonne over the shifts, made worse by tonnes hauled below today's on the same shifts (`keep` per
    share of tonnes lost) and better by tonnes above today's (`gain`)."""
    t, t0 = sum(r["hauled_t"] for r in rows), sum(r["hauled_t"] for r in base)
    if t <= 0:
        return 1e6
    short = max(0.0, 1 - t / t0) if t0 else 0.0
    more = max(0.0, t / t0 - 1) if t0 else 0.0
    return sum(r["cost_total"] for r in rows) / t * max(0.05, 1 + keep * short - gain * more)


def goal_weights(w: float) -> tuple[float, float]:
    """The planner's slider, 0 = the cheapest tonne … 1 = the most tonnes, as (keep, gain) of the objective:
    0 – only the cost per tonne counts; 0.5 – as before (losing tonnes is costly); 1 – every tonne counts."""
    w = float(np.clip(w, 0.0, 1.0))
    keep = 20.0 * w * w
    return keep, keep * max(0.0, 2 * w - 1)


@dataclass
class SignalsResult:
    params: dict
    history: pd.DataFrame
    setup: dict


def train_signals(states: list[Scenario], iterations: int = 8, population: int = 10, episodes: int = 3,
                  seed: int = 0, callback: Callable[[int, int, dict], None] | None = None,
                  weight: float = 0.5, check: int = 8) -> SignalsResult:
    """Learn the signal settings (CEM) on shifts starting from `states` – the situations of a month worked as
    today (full piles, last tonnes of a pile, an excavator after a move …). Every iteration practises on other
    shifts; all candidates of an iteration on the same ones, compared with today's driving on them."""
    rng = np.random.default_rng(seed)
    dim = len(SIGNAL_KEYS)
    mu, sigma = np.zeros(dim), np.ones(dim)
    n_elite = max(2, population // 4)
    keep, gain = goal_weights(weight)
    history, means = [], []
    for it in range(iterations):
        cases = [(states[int(rng.integers(len(states)))], 50_000 + 1000 * it + e) for e in range(episodes)]
        base = _run(cases, None)
        cand = mu + sigma * rng.standard_normal((population, dim))
        cand[0] = mu
        scored = sorted(((signals_objective(_run(cases, decode_signals(x)), base, keep, gain), k)
                         for k, x in enumerate(cand)))
        elite = np.array([cand[k] for _, k in scored[:n_elite]])
        mu = elite.mean(axis=0)
        means.append(mu.copy())
        sigma = elite.std(axis=0) + max(0.05, 0.5 * (1 - it / max(iterations - 1, 1)))
        a = aggregate(base)
        row = {"runda": it + 1, "dzis_zl_t": a["cost_per_t"], "sygnaly_zl_t": scored[0][0]}
        history.append(row)
        if callback:
            callback(it + 1, iterations, row)
    # a round practises on a few random shifts, so its winner may just have been lucky: the last settings are
    # checked on the same spread of the month's shifts and the best one is kept
    if check and len(means) > 1:
        idx = np.linspace(0, len(states) - 1, min(check, len(states))).round().astype(int)
        cases = [(states[i], 90_000 + e) for e, i in enumerate(idx)]
        base = _run(cases, None)
        mu = min(means[-3:], key=lambda m: signals_objective(_run(cases, decode_signals(m)), base, keep, gain))
    return SignalsResult(params=decode_signals(mu), history=pd.DataFrame(history),
                         setup={"iterations": iterations, "population": population, "episodes": episodes,
                                "states": len(states), "shifts": iterations * (population + 1) * episodes,
                                "weight": weight})


def explain_signals(p: dict) -> list[str]:
    """The learned signals in plain words."""
    out = explain_advisor(p)
    if p.get("truck_park_queue_min", -1.0) < 0:
        out.append("⏸️ Wozidła nie czekają na postoju – każde jeździ bez przerwy.")
    else:
        out.append(f"⏸️ Gdy po kursie wozidło stałoby pod kruszarką dłużej niż **{p['truck_park_queue_min']:.0f} min**, "
                   f"dostaje sygnał „czekaj, silnik wyłączony”. Zawsze jeździ co najmniej "
                   f"**{100 * p['truck_min_working_share']:.0f} %** wozideł.")
    return out
