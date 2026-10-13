"""Production goal planned backwards from the crushers."""
from pathlib import Path

import yaml

from kruszyk.config import scenario_from_dict
from kruszyk.goal import apply_plan, default_yields, plan_backwards, products_made

MINE = Path(__file__).parent.parent / "scenarios" / "kruszyk_kopalnia.yaml"
MIX = {"grysy (2–16)": 75.0, "kliniec 16–31,5": 15.0, "mieszanki / podbudowa": 10.0}


def mine():
    return scenario_from_dict(yaml.safe_load(MINE.read_text(encoding="utf-8")))


def yields(sc):
    return {c.id: default_yields(c.product) for c in sc.crushers}


def test_goal_is_met_within_capacity_and_supply():
    sc = mine()
    gp = plan_backwards(sc, 200_000, MIX, yields(sc), 44)
    assert gp.feasible, gp.limits
    for p in gp.products:
        assert p["plan_t"] >= p["cel_t"] - 1
    assert abs(sum(c["nadawa_t"] for c in gp.crushers) - gp.feed_t) <= 2
    for s in gp.sources:  # never more than there is
        assert s["plan_t"] <= s["dostepne_t"] + 1
    for c, cfg in zip(gp.crushers, sc.crushers):  # C-1 takes only from O3
        assert c["nadawa_t"] <= cfg.capacity_tph * 44 * sc.shift_hours
    assert {f["skad"] for f in gp.flows if f["kruszarka"] == "C-1"} <= {"O3"}
    assert gp.drilling["metry"] > 0 and gp.haulage["wozidel_potrzeba"] > 0




def test_too_big_goal_names_the_limit():
    sc = mine()
    gp = plan_backwards(sc, 5_000_000, MIX, yields(sc), 10)
    assert not gp.feasible
    assert any("brakuje" in line for line in gp.limits)
    assert all(p["brak_t"] > 0 for p in gp.products)


def test_product_nobody_makes_is_reported():
    sc = mine()
    gp = plan_backwards(sc, 50_000, {"kamień łamany / narzut 63–250": 100.0}, yields(sc), 44)
    assert not gp.feasible and "żadna kruszarka" in gp.limits[0]


def test_apply_sets_targets_and_shares_and_stays_valid():
    sc = mine()
    gp = plan_backwards(sc, 200_000, MIX, yields(sc), 44)
    apply_plan(sc, gp)
    sc.validate()
    assert all(c.target_t > 0 for c in sc.crushers if any(f["kruszarka"] == c.id for f in gp.flows))
    shares = [b.target_share for b in sc.blasts] + [a.get("target_share") for a in sc.areas]
    assert abs(sum(x or 0 for x in shares) - 100) < 1.0
    made = products_made({c["kruszarka"]: c["nadawa_t"] for c in gp.crushers}, gp.yields, list(MIX))
    assert made["grysy (2–16)"] >= 150_000 - 5


def test_haulage_calibrated_on_today():
    sc = mine()
    plain = plan_backwards(sc, 200_000, MIX, yields(sc), 44)
    slow = plan_backwards(sc, 200_000, MIX, yields(sc), 44, today={"hauled_t": 2000.0})
    assert slow.haulage["wozidel_potrzeba"] > plain.haulage["wozidel_potrzeba"]
