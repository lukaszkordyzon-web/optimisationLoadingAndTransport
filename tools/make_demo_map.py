"""Builds scenarios/kruszyk_mapa_demo.yaml – a synthetic pit drawn as on the map screen.

Geometry is illustrative (not a real site): main road, a ramp, a branch leaving the ramp half-way
down to an intermediate level, a bottom level with two piles, three excavators, mixed fleet.
"""
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kruszyk.config import yaml_dump  # noqa: E402
from kruszyk.psd import PRESETS  # noqa: E402
from kruszyk.site import analyze_site, site_to_scenario  # noqa: E402

C = [50.0000, 20.0000]          # crusher
J1 = [50.0000, 20.0125]         # end of the main road, top of the ramp (~900 m east)
B = [49.9955, 20.0180]          # bottom of the ramp (~640 m)
MID = [(J1[0] + B[0]) / 2, (J1[1] + B[1]) / 2]  # half-way down the ramp – branch starts mid-segment

site = {
    "center": [49.9975, 20.0100], "zoom": 15,
    "crusher": {"lat": C[0], "lon": C[1], "elev_m": 0.0},
    "roads": [
        {"id": "R1", "name": "droga główna", "kind": "droga", "grade_draw_pct": -1.0, "speed_limit_kmh": 40,
         "coords": [C, [50.0003, 20.0060], J1]},
        {"id": "R2", "name": "rampa główna", "kind": "rampa", "grade_draw_pct": -9.0, "speed_limit_kmh": 25,
         "single_lane": True, "coords": [J1, B]},
        {"id": "R3", "name": "zjazd na poziom pośredni", "kind": "droga", "grade_draw_pct": 0.0, "speed_limit_kmh": 30,
         "coords": [MID, [49.9972, 20.0110], [49.9968, 20.0085]]},
        {"id": "R4", "name": "poziom dolny", "kind": "droga", "grade_draw_pct": 0.0, "speed_limit_kmh": 30,
         "coords": [B, [49.9950, 20.0225], [49.9945, 20.0270]]},
        # planned: a gentler exit from the east end of the bottom level straight to the crusher (off today)
        {"id": "R5", "name": "planowany wyjazd wschodni", "kind": "rampa", "grade_draw_pct": 3.3, "speed_limit_kmh": 35,
         "active": False, "coords": [[49.9945, 20.0270], [49.9975, 20.0200], C]},
    ],
    "piles": [
        {"id": "U-01", "target_share": 40, "material": "wapień", "tonnage_t": 9000, "psd": PRESETS["średnie"], "loose_density_t_m3": 1.75,
         "polygon": [[49.9962, 20.0078], [49.9962, 20.0092], [49.9957, 20.0092], [49.9957, 20.0078]]},
        {"id": "U-02", "target_share": 30, "material": "dolomit", "tonnage_t": 9000, "psd": PRESETS["grube"], "loose_density_t_m3": 1.85,
         "polygon": [[49.9944, 20.0218], [49.9944, 20.0232], [49.9939, 20.0232], [49.9939, 20.0218]]},
        {"id": "U-03", "target_share": 30, "material": "wapień", "tonnage_t": 12000, "psd": PRESETS["drobne (dobry odstrzał)"], "loose_density_t_m3": 1.75,
         "polygon": [[49.9940, 20.0265], [49.9940, 20.0280], [49.9935, 20.0280], [49.9935, 20.0265]]},
    ],
    "excavators": [
        {"id": "K-1", "lat": 49.99665, "lon": 20.0085, "pile": "U-01", "bucket_m3": 6.0, "swing_cycle_s": 30,
         "spotting": {"tyl_1": 70, "bok_rondo": 30}},
        {"id": "K-2", "lat": 49.99475, "lon": 20.0225, "pile": "U-02", "bucket_m3": 4.5, "swing_cycle_s": 28,
         "spotting": {"tyl_1": 100}},
        {"id": "K-3", "lat": 49.99435, "lon": 20.0270, "pile": "U-03", "bucket_m3": 7.5, "swing_cycle_s": 34,
         "spotting": {"tyl_1": 50, "bok_cofanie": 50}},
    ],
}

raw = yaml.safe_load((ROOT / "scenarios" / "kruszyk_bazowy.yaml").read_text(encoding="utf-8"))
raw["name"] = "Kopalnia Kruszyk – mapa demo (sieć dróg, 3 usypy, mieszanka)"
raw["dispatch"] = "natural"
raw["trucks"] = [
    {"model": "wozidło 55 t", "count": 7, "payload_t": 55, "empty_mass_t": 40, "power_kw": 520,
     "max_speed_kmh": 45, "fuel_lph_loaded": 60, "fuel_lph_empty": 32, "fuel_lph_idle": 10},
    {"model": "wozidło 40 t – mieszanka K-1 + K-2", "count": 2, "payload_t": 40, "empty_mass_t": 30,
     "power_kw": 400, "max_speed_kmh": 48, "fuel_lph_loaded": 45, "fuel_lph_empty": 24, "fuel_lph_idle": 8,
     "blend": {"K-1": 0.5, "K-2": 0.5}},
]
# the main ramp is single-lane: trucks pass only at the junction half-way down and at the lower bay
site["passing_bays"] = [{"id": "M1", "lat": MID[0], "lon": MID[1]},
                        {"id": "M2", "lat": MID[0] + 0.5 * (B[0] - MID[0]), "lon": MID[1] + 0.5 * (B[1] - MID[1])}]
site["blast"] = {"lat": 49.99405, "lon": 20.0262}  # next blast: on the bottom level, east of U-03
# where excavators go for blasting / refuelling (K-1 has no marker: it backs off along its route)
site["parking"] = [{"excavator": "K-2", "lat": 49.9953, "lon": 20.0195},
                   {"excavator": "K-3", "lat": 49.99515, "lon": 20.0200}]
# foreman's own route: main road, ramp, intermediate level and back
site["dedicated"] = [{"for": "sztygar", "coords": [C, [50.0003, 20.0060], J1, MID, [49.9972, 20.0110], [49.9968, 20.0085]]}]
raw["blasting"] = {"enabled": True, "crew_arrive": "09:00", "fire_from": "11:00", "fire_to": "11:20",
                   "evacuation_min": 10, "crew_leave": "11:40"}
raw["inspection"] = {"enabled": True, "start": "09:30", "stops": ["K-1", "K-2", "K-3"], "stop_min": 10}
raw["refueling"] = {"tanker_times": ["13:00"], "excavator_refuel_min": 20}
raw["shift_plan"] = {"start_time": "06:00", "startup_min": 15, "end_early_min": 10,
                     "breaks": [{"start": "10:00", "min": 20}]}
raw["crusher"]["capacity_tph"] = 2600
raw["crusher"]["hopper_t"] = 250
new = site_to_scenario(site, raw)
info = analyze_site(site)
print("poziomy koparek:", info["levels"])
print("węzły:", len(info["junctions"]), "ostrzeżenia:", info["warnings"])
out = ROOT / "scenarios" / "kruszyk_mapa_demo.yaml"
out.write_text("# Wygenerowane przez tools/make_demo_map.py – geometria przykładowa (nie rzeczywisty teren)\n"
               + yaml_dump(new), encoding="utf-8")
print("zapisano", out)
