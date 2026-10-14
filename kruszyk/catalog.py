"""Catalogue of machines common in Polish aggregate quarries: pick a model, its parameters come with it.

Values are typical (manufacturers' data sheets, rounded; rock buckets, limestone / granite in quarry conditions) –
good enough to start a model, to be corrected by the calibration on a real shift. Excavator and loader cycles are
for loading dump trucks from a blasted pile; crusher capacities are for typical settings on blasted limestone.
"""
from __future__ import annotations

# bucket [m³], one bucket cycle [s], bucket fill in blasted rock, operating mass [t] (information)
EXCAVATORS: dict[str, dict] = {
    "Komatsu PC490LC-11 (koparka 49 t)": {"bucket_m3": 2.9, "swing_cycle_s": 23.0, "fill_factor": 0.85, "mass_t": 49},
    "CAT 352 (koparka 52 t)": {"bucket_m3": 3.1, "swing_cycle_s": 23.0, "fill_factor": 0.85, "mass_t": 52},
    "Volvo EC550E (koparka 55 t)": {"bucket_m3": 3.4, "swing_cycle_s": 24.0, "fill_factor": 0.85, "mass_t": 55},
    "Hitachi ZX690LCH-7 (koparka 70 t)": {"bucket_m3": 4.0, "swing_cycle_s": 26.0, "fill_factor": 0.85, "mass_t": 70},
    "CAT 374 (koparka 72 t)": {"bucket_m3": 4.5, "swing_cycle_s": 27.0, "fill_factor": 0.85, "mass_t": 72},
    "Volvo EC750E (koparka 75 t)": {"bucket_m3": 4.6, "swing_cycle_s": 27.0, "fill_factor": 0.85, "mass_t": 75},
    "Komatsu PC800SE-8 (koparka 78 t)": {"bucket_m3": 5.0, "swing_cycle_s": 28.0, "fill_factor": 0.85, "mass_t": 78},
    "Liebherr R 956 (koparka 85 t)": {"bucket_m3": 5.5, "swing_cycle_s": 29.0, "fill_factor": 0.85, "mass_t": 85},
    "CAT 390F (koparka 90 t)": {"bucket_m3": 5.8, "swing_cycle_s": 29.0, "fill_factor": 0.85, "mass_t": 90},
    # wheel loaders: longer cycle (they drive between the pile and the truck)
    "Volvo L180H (ładowarka 30 t)": {"bucket_m3": 4.2, "swing_cycle_s": 33.0, "fill_factor": 0.8, "mass_t": 30},
    "CAT 980 (ładowarka 31 t)": {"bucket_m3": 4.6, "swing_cycle_s": 34.0, "fill_factor": 0.8, "mass_t": 31},
    "Volvo L220H (ładowarka 33 t)": {"bucket_m3": 5.2, "swing_cycle_s": 34.0, "fill_factor": 0.8, "mass_t": 33},
    "Komatsu WA600-8 (ładowarka 53 t)": {"bucket_m3": 6.4, "swing_cycle_s": 36.0, "fill_factor": 0.8, "mass_t": 53},
    "Liebherr L 586 XPower (ładowarka 33 t)": {"bucket_m3": 5.0, "swing_cycle_s": 34.0, "fill_factor": 0.8, "mass_t": 33},
    "CAT 988K (ładowarka 51 t)": {"bucket_m3": 6.3, "swing_cycle_s": 37.0, "fill_factor": 0.8, "mass_t": 51},
}

_T = ("payload_t", "empty_mass_t", "power_kw", "max_speed_kmh", "fuel_lph_loaded", "fuel_lph_empty", "fuel_lph_idle",
      "tank_l")


def _truck(*v) -> dict:
    return {**dict(zip(_T, v)), "drive_efficiency": 0.75}


# payload [t], empty mass [t], power [kW], top speed on quarry roads [km/h], fuel loaded / empty / idle [l/h], tank [l]
TRUCKS: dict[str, dict] = {
    "CAT 770G (sztywnoramowe 36 t)": _truck(36.0, 33.6, 382, 60, 46, 25, 8, 568),
    "Komatsu HD325-8 (sztywnoramowe 36 t)": _truck(36.5, 32.6, 386, 60, 45, 25, 8, 568),
    "BELAZ 7547 (sztywnoramowe 45 t)": _truck(45.0, 32.0, 380, 50, 56, 30, 10, 650),
    "CAT 772G (sztywnoramowe 47 t)": _truck(47.0, 37.6, 446, 60, 53, 28, 9, 700),
    "Komatsu HD405-8 (sztywnoramowe 40 t)": _truck(40.0, 33.0, 397, 60, 48, 26, 8, 568),
    "CAT 773G (sztywnoramowe 55 t)": _truck(55.0, 43.6, 552, 60, 62, 32, 10, 878),
    "Komatsu HD465-8 (sztywnoramowe 55 t)": _truck(55.0, 43.0, 551, 60, 61, 31, 10, 845),
    "Volvo A30G (przegubowe 28 t)": _truck(28.5, 23.6, 265, 50, 32, 18, 6, 400),
    "Volvo A40G (przegubowe 39 t)": _truck(39.0, 31.0, 350, 55, 41, 22, 7, 480),
    "Volvo A45G (przegubowe 41 t)": _truck(41.0, 33.0, 375, 55, 43, 24, 7, 480),
    "CAT 745 (przegubowe 41 t)": _truck(41.0, 33.0, 379, 55, 43, 24, 7, 556),
    "Bell B45E (przegubowe 41 t)": _truck(41.0, 34.0, 380, 55, 44, 24, 7, 550),
}

# capacity on blasted limestone [t/h], hopper [t], largest feed [mm], tipping time [s]; mobile = tracked unit with a
# small hopper (a truck tips in portions, it takes longer). Energy per tonne comes from the product (line).
CRUSHERS: dict[str, dict] = {
    "Metso Nordberg C120 – szczękowa stacjonarna": {"capacity_tph": 300.0, "hopper_t": 80.0, "max_feed_mm": 700.0,
                                                   "dump_time_s": 60.0, "mobile": False},
    "Metso Nordberg C130 – szczękowa stacjonarna": {"capacity_tph": 400.0, "hopper_t": 100.0, "max_feed_mm": 800.0,
                                                   "dump_time_s": 60.0, "mobile": False},
    "Metso Nordberg C150 – szczękowa stacjonarna": {"capacity_tph": 600.0, "hopper_t": 150.0, "max_feed_mm": 950.0,
                                                   "dump_time_s": 60.0, "mobile": False},
    "Sandvik CJ615 – szczękowa stacjonarna": {"capacity_tph": 450.0, "hopper_t": 120.0, "max_feed_mm": 900.0,
                                             "dump_time_s": 60.0, "mobile": False},
    "Hazemag APS 1313 – udarowa stacjonarna (wapień)": {"capacity_tph": 400.0, "hopper_t": 100.0, "max_feed_mm": 800.0,
                                                       "dump_time_s": 60.0, "mobile": False},
    "Metso Nordberg NP1315 – udarowa stacjonarna (wapień)": {"capacity_tph": 450.0, "hopper_t": 120.0,
                                                            "max_feed_mm": 900.0, "dump_time_s": 60.0, "mobile": False},
    "Metso Lokotrack LT106 – szczękowa mobilna": {"capacity_tph": 300.0, "hopper_t": 15.0, "max_feed_mm": 600.0,
                                                 "dump_time_s": 120.0, "mobile": True},
    "Metso Lokotrack LT120 – szczękowa mobilna": {"capacity_tph": 450.0, "hopper_t": 20.0, "max_feed_mm": 750.0,
                                                 "dump_time_s": 120.0, "mobile": True},
    "Kleemann MC 110 Z EVO – szczękowa mobilna": {"capacity_tph": 350.0, "hopper_t": 15.0, "max_feed_mm": 650.0,
                                                 "dump_time_s": 120.0, "mobile": True},
    "Kleemann MC 120 Z PRO – szczękowa mobilna": {"capacity_tph": 450.0, "hopper_t": 20.0, "max_feed_mm": 750.0,
                                                 "dump_time_s": 120.0, "mobile": True},
    "Kleemann MR 130 Z EVO2 – udarowa mobilna (wapień)": {"capacity_tph": 400.0, "hopper_t": 15.0,
                                                         "max_feed_mm": 700.0, "dump_time_s": 120.0, "mobile": True},
    "Sandvik QJ341 – szczękowa mobilna": {"capacity_tph": 400.0, "hopper_t": 14.0, "max_feed_mm": 650.0,
                                         "dump_time_s": 120.0, "mobile": True},
}

KINDS = {"excavator": EXCAVATORS, "truck": TRUCKS, "crusher": CRUSHERS}
# what a catalogue model sets on the object
FIELDS = {"excavator": ("bucket_m3", "swing_cycle_s", "fill_factor"),
          "truck": (*_T, "drive_efficiency"),
          "crusher": ("capacity_tph", "hopper_t", "max_feed_mm", "dump_time_s")}


def names(kind: str) -> list[str]:
    return list(KINDS[kind])


def params(kind: str, model: str) -> dict:
    """Parameters a model sets (empty for a model outside the catalogue)."""
    m = KINDS[kind].get(model) or {}
    return {k: m[k] for k in FIELDS[kind] if k in m}


def apply(kind: str, obj: dict, model: str) -> dict:
    """Give `obj` the model's parameters (and its name); returns obj."""
    p = params(kind, model)
    if p:
        obj.update(p)
        obj["model"] = model
    return obj
