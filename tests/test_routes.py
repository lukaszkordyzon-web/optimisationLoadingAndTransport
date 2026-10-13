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


def test_single_lane_ramp_sections_and_uphill_priority():
    """A single-lane ramp is split at passing bays; trucks wait at the bays and fewer bays cost more time."""
    import copy

    import yaml

    from kruszyk.config import scenario_from_dict
    from kruszyk.optimize import replicate
    from kruszyk.site import point_along, site_to_scenario
    from kruszyk.sim import simulate

    raw = yaml.safe_load(open(Path(__file__).parent.parent / "scenarios" / "kruszyk_mapa_demo.yaml"))

    def variant(fracs, single=True):
        r = copy.deepcopy(raw)
        ramp = next(x for x in r["site"]["roads"] if x["id"] == "R2")
        ramp["single_lane"] = single
        from kruszyk.site import polyline_length_m
        L = polyline_length_m(ramp["coords"])
        r["site"]["passing_bays"] = [dict(zip(("lat", "lon"), point_along(ramp["coords"], f * L))) for f in fracs]
        return scenario_from_dict(site_to_scenario(r["site"], r))

    sc = variant([0.25, 0.5, 0.75])
    secs = [s.section for s in sc.routes["T-K-3"].segments if s.name == "rampa główna"]
    assert len(secs) == 4 and len(set(secs)) == 4
    two_lane = replicate(variant([], single=False), 3)
    none = replicate(variant([]), 3)
    many = replicate(variant([0.25, 0.5, 0.75]), 3)
    assert two_lane["bay_wait_h"] == 0
    assert none["bay_wait_h"] > many["bay_wait_h"] > 0
    assert two_lane["throughput_tph"] > none["throughput_tph"]
    # a stretch never carries both directions at once
    violations = []

    class Checked(Simulation):
        def _enter_section(self, truck, seg, forward, route, k, load_t):
            yield from super()._enter_section(truck, seg, forward, route, k, load_t)
            d, _ = self._dirs(seg, forward)
            if self.sections[seg.section]["inside"][-d]:
                violations.append((seg.section, self.env.now))

    sim = Checked(variant([0.5]), seed=3)
    res = sim.run()
    assert not violations
    assert sim.bay_stops > 0 and (res.states["state"] == "mijanka").any()


def test_spotting_modes_two_sided_is_faster():
    import copy

    from kruszyk.config import load_scenario
    from kruszyk.optimize import replicate

    sc = load_scenario(Path(__file__).parent.parent / "scenarios" / "kruszyk_mapa_demo.yaml")
    out = {}
    for mode in ("tyl_1", "tyl_2", "bok_cofanie"):
        v = copy.deepcopy(sc)
        for e in v.excavators:
            e.spotting = {mode: 100}
        out[mode] = replicate(v, 3)
    assert out["tyl_2"]["throughput_tph"] > out["tyl_1"]["throughput_tph"]
    assert out["tyl_2"]["excavator_utilization"] < out["tyl_1"]["excavator_utilization"]  # spotting hidden


def test_excavator_advances_along_digging_line():
    from kruszyk.config import load_scenario

    sc = load_scenario(Path(__file__).parent.parent / "scenarios" / "kruszyk_mapa_demo.yaml")
    k3 = next(e for e in sc.excavators if e.id == "K-3")
    assert len(k3.dig_routes) > 2 and k3.route == k3.dig_routes[0]
    fronts = [sc.routes[r].segments[0].length_m for r in k3.dig_routes]
    assert fronts[-1] > fronts[0] + 30  # the end of the line is further from the road
    sim = Simulation(sc, seed=2)
    res = sim.run()
    ex = sim.excavators["K-3"]
    assert ex.relocations >= 1 and ex.route_id == k3.dig_routes[ex.dig_step]
    assert res.kpi["excavator_down_h"]["przestawienie"] > 0
    used = set(res.segments.loc[res.segments["excavator"] == "K-3", "route"])
    assert len(used & set(k3.dig_routes)) >= 2  # trucks drove to more than one place on the face
    track = res.excavator_tracks["K-3"]
    assert len({(round(a, 6), round(b, 6)) for _, a, b in track}) >= 2  # the excavator moves in the animation
