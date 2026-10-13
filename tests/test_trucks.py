"""Every truck on its own: own id, own parameters, own breakdowns."""
from pathlib import Path

import yaml

from kruszyk.config import expand_trucks, scenario_from_dict, truck_units
from kruszyk.sim import simulate

THREE = Path(__file__).parent.parent / "scenarios" / "kruszyk_3_koparki.yaml"


def fleet():
    raw = yaml.safe_load(THREE.read_text(encoding="utf-8"))
    raw["trucks"] = expand_trucks(raw["trucks"])
    return raw


def test_groups_split_into_named_trucks():
    raw = fleet()
    assert [t["count"] for t in raw["trucks"]] == [1] * 13
    assert [t["id"] for t in raw["trucks"]] == [f"WZ-{i:02d}" for i in range(1, 14)]
    sc = scenario_from_dict(raw)
    assert [tid for tid, _ in truck_units(sc.trucks)] == [f"WZ-{i:02d}" for i in range(1, 14)]


def test_one_truck_with_its_own_parameters():
    raw = fleet()
    raw["trucks"][2]["payload_t"] = 20.0  # WZ-03 is a small one
    res = simulate(scenario_from_dict(raw), seed=3)
    loads = res.loads.groupby("truck")["tonnes"].max()
    assert loads["WZ-03"] <= 20.0 + 1e-6 and loads.drop("WZ-03").max() > 30


def test_truck_out_of_work_and_fleet_size_changes():
    raw = fleet()
    raw["trucks"][0]["count"] = 0  # WZ-01 in repair
    sc = scenario_from_dict(raw)
    assert sc.truck_count == 12 and "WZ-01" not in {t for t, _ in truck_units(sc.trucks)}
    more = sc.with_truck_count(14)
    ids = [t for t, _ in truck_units(more.trucks)]
    assert len(ids) == 14 and "WZ-01" in ids and len(set(ids)) == 14  # the idle one comes back first, then a copy
    less = sc.with_truck_count(5)
    assert [t for t, _ in truck_units(less.trucks)] == ["WZ-02", "WZ-03", "WZ-04", "WZ-05", "WZ-06"]


def test_breakdowns_per_truck():
    raw = fleet()
    raw["reliability"] = {"machines": {"WZ-05": {"count": 400, "hours": 4000},  # always broken
                                       **{f"WZ-{i:02d}": {"count": 0, "hours": 0} for i in range(1, 14) if i != 5}}}
    res = simulate(scenario_from_dict(raw), seed=2)
    broken = res.states[res.states["state"] == "awaria"]
    assert set(broken["truck"]) == {"WZ-05"}
