import copy
from pathlib import Path

import pytest
import yaml

from kruszyk.config import scenario_from_dict
from kruszyk.physics import crusher_rate_tph, load_time_s
from kruszyk.psd import PRESETS, clean, fragmentation_index, passing_at, rosin_rammler, size_at, summary
from kruszyk.site import RoadGraph, site_to_scenario

DEMO = Path(__file__).parent.parent / "scenarios" / "kruszyk_mapa_demo.yaml"


def test_rosin_rammler_hits_p80():
    c = rosin_rammler(400, 1.2)
    assert size_at(c, 80) == pytest.approx(400, rel=0.03)
    assert passing_at(c, 400) == pytest.approx(80, abs=1.5)


def test_coarser_curve_means_slower_loading_and_crushing():
    fr = [fragmentation_index(c) for c in PRESETS.values()]
    assert fr == sorted(fr) and fr[0] < 0.1 and fr[-1] > 0.9
    s = summary(PRESETS["grube"])
    assert s["nadziarno [%]"] > 0 and s["załadunek dłuższy o [%]"] > 0


def test_clean_makes_curve_monotonic():
    assert clean([[100, 50], [10, 20], [50, 10], [200, 120]]) == [[10, 20], [50, 20], [100, 50], [200, 100]]


def test_curve_sets_pile_fragmentation_and_affects_physics():
    raw = yaml.safe_load(DEMO.read_text(encoding="utf-8"))
    fine, coarse = copy.deepcopy(raw), copy.deepcopy(raw)
    for b in fine["blasts"]:
        b["psd"] = PRESETS["drobne (dobry odstrzał)"]
    for b in coarse["blasts"]:
        b["psd"] = PRESETS["bardzo grube (dużo brył)"]
    f, c = scenario_from_dict(fine), scenario_from_dict(coarse)
    ex, truck = f.excavators[0], f.trucks[0]
    assert load_time_s(ex, truck, c.blasts[0], 1.75) > load_time_s(ex, truck, f.blasts[0], 1.75)
    assert crusher_rate_tph(c.crusher, c.blasts[0].fragmentation) < crusher_rate_tph(f.crusher, f.blasts[0].fragmentation)


def test_planned_road_is_ignored_until_switched_on():
    raw = yaml.safe_load(DEMO.read_text(encoding="utf-8"))
    roads = raw["site"]["roads"]
    planned = next(r for r in roads if r.get("active") is False)
    assert len(RoadGraph(roads).roads) == len(roads) - 1
    today = scenario_from_dict(site_to_scenario(copy.deepcopy(raw["site"]), copy.deepcopy(raw)))
    planned["active"] = True
    variant = scenario_from_dict(site_to_scenario(copy.deepcopy(raw["site"]), copy.deepcopy(raw)))
    names = lambda sc: {s.name for s in sc.routes["T-K-3"].segments}  # noqa: E731
    assert planned["name"] not in names(today) and planned["name"] in names(variant)
