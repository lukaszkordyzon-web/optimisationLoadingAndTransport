import copy
from pathlib import Path

import pytest
import yaml

from kruszyk.config import load_scenario
from kruszyk.sim import Simulation, simulate
from kruszyk.site import (RoadGraph, analyze_site, crew_route, excavator_move, normalize_roads, polyline_route,
                          round_route)

DEMO = Path(__file__).parent.parent / "scenarios" / "kruszyk_mapa_demo.yaml"
C = [50.0, 20.0]
E = [50.0, 20.014]   # ~1 km east


def _site(roads):
    return {"crusher": {"lat": C[0], "lon": C[1], "elev_m": 0.0}, "roads": roads,
            "excavators": [{"id": "K-1", "lat": E[0], "lon": E[1]}]}


def test_drawing_direction_decides_up_or_down():
    down = _site([{"name": "r", "grade_draw_pct": -5.0, "coords": [C, E]}])        # drawn from crusher, going down
    up = _site([{"name": "r", "grade_draw_pct": 5.0, "coords": [E, C]}])           # drawn towards crusher, climbing
    hill = _site([{"name": "r", "grade_draw_pct": 5.0, "coords": [C, E]}])         # away from crusher but UP
    lv = lambda s: analyze_site(s)["levels"]["K-1"]["elev_m"]  # noqa: E731
    assert lv(down) == pytest.approx(-50, abs=1.5)
    assert lv(up) == pytest.approx(-50, abs=1.5)
    assert lv(hill) == pytest.approx(50, abs=1.5)   # not everything goes down from the crusher


def test_legacy_grades_are_converted_keeping_the_numbers():
    raw = yaml.safe_load(DEMO.read_text(encoding="utf-8"))
    site = raw["site"]
    for r in site["roads"]:  # rebuild an old-style map: grade towards the crusher
        g = r.pop("grade_draw_pct")
        r["grade_pct"] = abs(g) if r["name"] != "planowany wyjazd wschodni" else 3.3
    before = copy.deepcopy(site)
    normalize_roads(site)
    assert {r["name"]: r["grade_draw_pct"] for r in site["roads"]}["rampa główna"] == -9.0
    assert analyze_site(site)["levels"]["K-2"]["elev_m"] == pytest.approx(-66.2, abs=0.2)
    assert all("grade_pct" not in r for r in site["roads"]) and before != site


def test_dedicated_route_shares_roads_and_has_stops():
    sc = load_scenario(DEMO)
    site = copy.deepcopy(sc.site)
    r = round_route(site, "sztygar", [(49.9968, 20.0085)])
    names = [s["name"] for s in r["segments"]]
    assert "droga główna" in names and "rampa główna" in names
    ramp = next(s for s in r["segments"] if s["name"] == "rampa główna")
    assert ramp["grade_pct"] < 0  # driven down the ramp
    assert r["stop_after"] and r["stop_after"][0] == len(r["segments"]) - 1


def test_crew_route_starts_at_blast_point():
    sc = load_scenario(DEMO)
    r = crew_route(copy.deepcopy(sc.site))
    assert r["path"][0] == pytest.approx([sc.site["blast"]["lat"], sc.site["blast"]["lon"]], abs=1e-6)


def test_excavator_goes_to_parking_or_backs_off():
    sc = load_scenario(DEMO)
    site = copy.deepcopy(sc.site)
    k1 = next(e for e in site["excavators"] if e["id"] == "K-1")
    k2 = next(e for e in site["excavators"] if e["id"] == "K-2")
    back = excavator_move(site, k1, sc.routes["T-K-1"].path, retreat_m=200)
    assert back["parking"] is False and back["segments"][0]["length_m"] == pytest.approx(200, abs=1)
    park = excavator_move(site, k2, sc.routes["T-K-2"].path)
    assert park["parking"] is True and park["path"][-1] == pytest.approx([49.9953, 20.0195], abs=1e-4)


def test_blasting_downtime_includes_driving_to_parking():
    sc = load_scenario(DEMO)
    sim = Simulation(sc)
    a, b = sim.closure
    for ex_id, ex in sim.excavators.items():
        win = next(w for w in ex.down if w[2] == "strzelanie")
        move_t = sim.excavator_moves[ex_id][1]
        assert win[0] == pytest.approx(a - move_t) and win[1] == pytest.approx(b + move_t)
    res = simulate(sc)
    assert set(res.excavator_tracks) == {"K-1", "K-2", "K-3"}
    assert any(v["name"] == "sztygar" for v in res.light_vehicles)


def test_planned_roads_are_not_used_by_graph():
    sc = load_scenario(DEMO)
    assert len(RoadGraph(sc.site["roads"]).roads) == len(sc.site["roads"]) - 1
    assert len(RoadGraph(sc.site["roads"], include_planned=True).roads) == len(sc.site["roads"])
