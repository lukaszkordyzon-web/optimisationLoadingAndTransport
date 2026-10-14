"""Does the learned policy (RL) give sensible answers? Three situations on the test mine, each policy on the same
30 shifts (same weather, drivers, breakdowns):

  A. crusher is the bottleneck (too many trucks)   -> RL should park trucks, tonnes stay, cost falls
  B. haulage is the bottleneck (big crusher, long roads, few trucks) -> little to gain: every truck is needed
  C. balanced, two levels at different distances   -> dispatch to the nearer level / where the queue is shorter

    PYTHONPATH=. python tools/walidacja/polityki.py   -> tools/walidacja/wyniki/polityki.csv
"""
from __future__ import annotations

import copy
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from czulosc import _params  # noqa: E402
from kopalnia_testowa import make  # noqa: E402

from kruszyk.rl import explain_excavators, train  # noqa: E402
from kruszyk.sim import simulate  # noqa: E402

OUT = Path(__file__).parent / "wyniki"
SITUATIONS = {
    "A kruszarka wąskim gardłem": {"trucks": 6},
    "B transport wąskim gardłem": {"trucks": 3, "crusher_tph": 800, "dist": 2.0},
    "C zrównoważona": {"trucks": 4, "crusher_tph": 600, "dist": 1.5},
    "D jak C + skład nadawy 50/50": {"trucks": 4, "crusher_tph": 600, "dist": 1.5, "share_b1": 50.0},
}
EVAL = range(500, 530)


def learn(item):
    name, p = item
    sc = make(**_params(p))
    res = train(sc, iterations=10, population=12, episodes=2, shift_hours=4.0, eval_episodes=3, seed=1,
                advisor=False, learn_excavators=True)
    pol = copy.deepcopy(sc)
    pol.dispatch, pol.rl_weights = "rl", res.weights
    for k, v in (res.excavators or {}).items():
        setattr(pol.excavator_policy, k, v)
    rows = []
    for disp in ("natural", "fixed", "dynamic", "rl"):
        v = pol if disp == "rl" else copy.deepcopy(sc)
        v.dispatch = disp
        for s in EVAL:
            k = simulate(v, seed=s).kpi
            rows.append({"sytuacja": name, "polityka": disp, "seed": s, "t": k["hauled_t"], "zl_t": k["cost_per_t"],
                         "paliwo_l_t": k["fuel_l_per_t"], "kolejka_kr_h": k["truck_queue_crusher_h"],
                         "kolejka_kop_h": k["truck_queue_excavator_h"], "parking_h": k["parked_truck_h"],
                         "koparki_odst_h": k["parked_excavator_h"],
                         "z_B1_pct": k["feed_share"].get("B1", 0.0)})
    return rows, name, explain_excavators(res.excavators) if res.excavators else []


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    with ProcessPoolExecutor(4) as ex:
        out = list(ex.map(learn, SITUATIONS.items()))
    rows = [r for rs, _, _ in out for r in rs]
    pd.DataFrame(rows).to_csv(OUT / "polityki.csv", index=False)
    for _, name, lessons in out:
        print(name, "\n  " + "\n  ".join(lessons))
