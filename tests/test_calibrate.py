"""Calibration: the model is tuned to what a real shift measured."""
from pathlib import Path

import pytest
import yaml

from kruszyk.calibrate import apply_to_raw, calibrate, model_shift
from kruszyk.config import dump_scenario, scenario_from_dict

MINE = Path(__file__).parent.parent / "scenarios" / "kruszyk_kopalnia.yaml"


def raw():
    return yaml.safe_load(MINE.read_text(encoding="utf-8"))


def test_lower_real_production_is_matched():
    sc = scenario_from_dict(raw())
    fitted, before, after = calibrate(sc, {"tonnes_t": 3500, "trips": 95, "load_min": 3.5, "fuel_l": 900, "kwh": 4200})
    assert before["tonnes_t"] > 4500
    assert after["tonnes_t"] == pytest.approx(3500, rel=0.04)
    assert after["trips"] == pytest.approx(95, rel=0.05)
    assert after["load_min"] == pytest.approx(3.5, rel=0.08)
    assert after["fuel_l"] == pytest.approx(900, rel=0.05)
    assert after["kwh"] == pytest.approx(4200, rel=0.05)
    assert fitted.calibration.delay_min_per_trip > 0


def test_higher_real_production_than_the_crusher_allows_is_explained():
    sc = scenario_from_dict(raw())  # here the 800 t/h crusher C-2 takes all the feed of a single shift
    base = model_shift(sc)["tonnes_t"]
    fitted, _, after = calibrate(sc, {"tonnes_t": base * 1.3})
    assert fitted.calibration.delay_min_per_trip == 0
    assert "C-2" in after["note"]


def test_higher_real_production_speeds_a_fleet_limited_model_up():
    r = raw()
    for c in [r["crusher"]] + r["crushers_extra"]:
        c["capacity_tph"] = 3000.0
    sc = scenario_from_dict(r).with_truck_count(4)
    sc.dispatch = "natural"
    base = model_shift(sc)["tonnes_t"]
    fitted, _, after = calibrate(sc, {"tonnes_t": base * 1.15, "trucks": 4})
    assert fitted.calibration.travel_factor < 1
    assert after["tonnes_t"] == pytest.approx(base * 1.15, rel=0.05)


def test_calibration_is_saved_in_the_mine():
    r = raw()
    fitted, _, after = calibrate(scenario_from_dict(r), {"tonnes_t": 4000, "trips": 110})
    apply_to_raw(r, fitted)
    again = scenario_from_dict(r)
    again.dispatch = "natural"  # calibrated on today's way of dispatching…
    again.reliability.enabled = False  # …on a shift without breakdowns
    assert again.calibration.measured["tonnes_t"] == 4000
    assert "calibration" in dump_scenario(again)
    assert model_shift(again)["tonnes_t"] == pytest.approx(after["tonnes_t"], rel=0.01)
