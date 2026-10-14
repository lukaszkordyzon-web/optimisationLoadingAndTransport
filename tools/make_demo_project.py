"""Demo project on the former Zakrzówek quarry (Kraków): a realistic, fictional limestone operation.

Design basis (typical values for a mid-size Polish limestone quarry):
  * 1.5 Mt of run-of-quarry rock a year; 2 shifts × 8 h, Mon–Fri + some Saturdays ≈ 500 shifts → ~3 000 t a shift,
    ~430 t/h of hauling in the ~7 productive hours of a shift;
  * blasted limestone: P50 240 mm, P80 450 mm (Rosin–Rammler n ≈ 1.34), loose 1.70 t/m³, in situ 2.65 t/m³;
    15 m benches, Ø 102–115 mm holes on a 3.5 × 4.0 m pattern → ~33 t per metre drilled; at most 3 000 kg of
    explosives a blast (usual Polish permit) at ~0.145 kg/t → blasts of ≤ 20 kt, ~75 a year;
  * plant with two lines fed by dump trucks: line I (jaw + cone + VSI) for chippings, line II (impact) for
    16–31.5, mixes and road base; product split from typical limestone yields;
  * fleet a few years old: two 70 t crawler excavators with 4.5 m³ rock buckets, a 45 t wheel loader as reserve,
    six 38–39 t dump trucks (4 rigid + 2 articulated), two drill rigs.
Roads, piles and extraction areas are drawn on the map by the user; machines and plant are placed there and take
their parameters from the catalogue below.

    python tools/make_demo_project.py
"""
from __future__ import annotations

from pathlib import Path

import yaml

from kruszyk.project import DEMO_CENTER, DEMO_NAME, DEMO_ZOOM, new_project

ROOT = Path(__file__).parent.parent
template = yaml.safe_load((ROOT / "scenarios" / "kruszyk_kopalnia.yaml").read_text(encoding="utf-8"))
raw = new_project(template, DEMO_NAME, DEMO_CENTER, DEMO_ZOOM)

RIGID = {"model": "wozidło sztywnoramowe 38 t", "payload_t": 38.0, "empty_mass_t": 33.5, "power_kw": 380,
         "max_speed_kmh": 60, "drive_efficiency": 0.75, "fuel_lph_loaded": 48, "fuel_lph_empty": 26,
         "fuel_lph_idle": 8, "tank_l": 550.0}
ARTIC = {"model": "wozidło przegubowe 39 t", "payload_t": 39.0, "empty_mass_t": 31.0, "power_kw": 350,
         "max_speed_kmh": 55, "drive_efficiency": 0.75, "fuel_lph_loaded": 42, "fuel_lph_empty": 24,
         "fuel_lph_idle": 7, "tank_l": 400.0}
raw["trucks"] = [{**(RIGID if i <= 4 else ARTIC), "id": f"WZ-{i:02d}", "count": 1, "excavators": []}
                 for i in range(1, 7)]
raw["drills"] = [{"id": "W-1", "in_work": True, "m_per_shift": 280.0, "speed_kmh": 2.5},
                 {"id": "W-2", "in_work": True, "m_per_shift": 220.0, "speed_kmh": 2.0}]
raw["crusher"] = {"capacity_tph": 400.0, "hopper_t": 90.0, "dump_time_s": 60.0, "max_feed_mm": 700.0,
                  "kwh_per_t": 2.0, "fragmentation_penalty": 0.4, "kwh_fragmentation_extra": 0.6}
raw["costs"] = {"truck_per_h": 260.0, "excavator_per_h": 520.0, "crusher_per_h": 1100.0, "fuel_per_l": 6.3,
                "energy_per_kwh": 0.85, "drill_blast_per_t": 3.6, "drill_blast_fine_extra_per_t": 2.0,
                "excavator_idle_share": 0.35, "blend_penalty_per_t_pp": 0.05}
raw["shift_plan"] = {"start_time": "06:00", "startup_min": 15.0, "end_early_min": 10.0,
                     "breaks": [{"start": "10:00", "min": 20.0}, {"start": "18:00", "min": 20.0}]}
raw["blasting"] = {"enabled": False, "crew_arrive": "11:00", "fire_from": "13:00", "fire_to": "13:20",
                   "evacuation_min": 15.0, "crew_leave": "13:40", "retreat_m": 200.0, "place": "",
                   "max_explosive_kg": 3000.0, "powder_factor_kg_t": 0.145}
raw["refueling"] = {**raw.get("refueling", {}), "truck_reserve_pct": 15.0, "truck_refuel_min": 10.0}
raw["variability"] = {**raw.get("variability", {}), "rain_probability": 0.25}
raw["loose_density_t_m3"] = 1.70

# breakdowns per machine: failures a month and their total hours (fleet a few years old)
raw["reliability"] = {"enabled": True, "ref_shifts_per_day": 2.0, "ref_days_per_month": 22.0, "machines": {
    **{f"WZ-{i:02d}": {"count": 3.0, "hours": 8.0} for i in range(1, 5)},
    **{f"WZ-{i:02d}": {"count": 3.0, "hours": 7.0} for i in range(5, 7)},
    "K-1": {"count": 2.0, "hours": 10.0}, "K-2": {"count": 2.0, "hours": 10.0}, "Ł-1": {"count": 2.0, "hours": 8.0},
    "C-1": {"count": 4.0, "hours": 14.0}, "C-2": {"count": 3.0, "hours": 9.0},
    "W-1": {"count": 3.0, "hours": 10.0}, "W-2": {"count": 4.0, "hours": 14.0}}}

# production: aimed at chippings; yields of the two lines [% of feed] – the rest is undersize / dust
YIELDS = {
    "C-1": {"piasek łamany 0–2": 18.0, "grys 2–8": 24.0, "grys 8–16": 27.0, "kliniec 16–31,5": 16.0,
            "mieszanka 0–31,5": 8.0},
    "C-2": {"kamień łamany / narzut 63–250": 5.0, "tłuczeń 31,5–63": 18.0, "kliniec 16–31,5": 22.0,
            "kruszywo 0–63 (podbudowa)": 20.0, "mieszanka 0–31,5": 25.0},
}
raw["production"] = {"total_t": 110_000.0, "yields": YIELDS,
                     "mix": {"grysy (2–16)": 45.0, "kliniec 16–31,5": 18.0, "mieszanki / podbudowa": 22.0,
                             "piasek łamany 0–2": 10.0, "tłuczeń 31,5–63": 5.0}}

# catalogue: what the drawing forms fill in when machines / piles / areas are placed on the map
raw["defaults"] = {
    "material": "wapień", "psd_preset": "średnio-grube (P50 240, P80 450 mm)", "loose_density_t_m3": 1.70,
    "solid_density_t_m3": 2.65, "bench_m": 15.0, "t_per_m": 33.0, "blast_max_t": 20_000.0,
    "reblast_below_t": 8_000.0, "min_gap_days": 2,
    "excavators": [
        {"id": "K-1", "type": "koparka gąsienicowa 70 t, łyżka skalna 4,5 m³", "bucket_m3": 4.5, "swing_cycle_s": 27.0,
         "fill_factor": 0.85},
        {"id": "K-2", "type": "koparka gąsienicowa 70 t, łyżka skalna 4,5 m³", "bucket_m3": 4.5, "swing_cycle_s": 27.0,
         "fill_factor": 0.85},
        {"id": "Ł-1", "type": "ładowarka kołowa 45 t, łyżka 6,0 m³ (rezerwa)", "bucket_m3": 6.0, "swing_cycle_s": 36.0,
         "fill_factor": 0.8}],
    "crushers": [
        {"id": "C-1", "type": "linia I – szczękowa + stożkowe + VSI, grysy, 400 t/h", "product": "grys 8–16",
         "capacity_tph": 400.0, "hopper_t": 90.0, "kwh_per_t": 2.0},
        {"id": "C-2", "type": "linia II – udarowa, kliniec / mieszanki / podbudowa, 300 t/h",
         "product": "mieszanka 0–31,5", "capacity_tph": 300.0, "hopper_t": 70.0, "kwh_per_t": 1.2}],
}
out = ROOT / "scenarios" / "kruszyk_demo_pusty.yaml"  # the drawn demo (kruszyk_demo_zakrzowek.yaml) is kept
out.write_text("# Projekt demonstracyjny na terenie dawnego kamieniołomu Zakrzówek (Kraków; nieczynny, bez operatora).\n"
               "# Wszystkie obiekty i dane są fikcyjne – typowe wartości dla średniej kopalni wapienia (1,5 mln t/rok).\n"
               "# Generuje: python tools/make_demo_project.py\n"
               + yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8")
print("→", out)
