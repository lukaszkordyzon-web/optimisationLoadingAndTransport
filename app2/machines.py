"""Editors of the machines (trucks, excavators, crushers) with the catalogue of models – used in ① Kopalnia → Maszyny
and in the variants of ② Harmonogram (a variant may have other machines than the mine)."""
from __future__ import annotations

import copy

import pandas as pd
import streamlit as st

from kruszyk import catalog

OWN = "(własny)"


def _num(v, default=0.0) -> float:
    try:
        return float(v) if v is not None and not pd.isna(v) else default
    except (TypeError, ValueError):
        return default


def _models(kind: str, current: list[str]) -> list[str]:
    return [OWN] + catalog.names(kind) + sorted({m for m in current if m and m not in catalog.KINDS[kind]})


CHANGED = "_model_changed"  # set on a machine whose model was just picked: the table must show its parameters


def _apply_model(kind: str, obj: dict, old_model: str, new_model: str) -> dict:
    """A model picked from the list brings its parameters; the planner's own values stay otherwise."""
    if new_model and new_model != old_model and new_model in catalog.KINDS[kind]:
        catalog.apply(kind, obj, new_model)
        obj[CHANGED] = True
    elif new_model and new_model != OWN:
        obj["model"] = new_model
    return obj


TRUCK_COLS = {"id": "ID", "model": "model", "ladownosc_t": "ładowność [t]", "masa_t": "masa pusta [t]",
              "moc_kw": "moc [kW]", "vmax_kmh": "v max [km/h]", "pal_pelny": "paliwo pełny [l/h]",
              "pal_pusty": "paliwo pusty [l/h]", "pal_jalowy": "paliwo jałowy [l/h]", "bak_l": "bak [l]"}
TRUCK_KEYS = {"ladownosc_t": ("payload_t", 36), "masa_t": ("empty_mass_t", 30), "moc_kw": ("power_kw", 350),
              "vmax_kmh": ("max_speed_kmh", 45), "pal_pelny": ("fuel_lph_loaded", 45),
              "pal_pusty": ("fuel_lph_empty", 25), "pal_jalowy": ("fuel_lph_idle", 8), "bak_l": ("tank_l", 600)}


def truck_editor(trucks: list[dict], key: str) -> list[dict]:
    """Every truck in a row: pick a model from the catalogue (its parameters come with it) or type your own values.
    Rows can be added and removed."""
    df = pd.DataFrame([{"id": t.get("id", ""), "model": t.get("model") or OWN,
                        **{c: t.get(k) for c, (k, _) in TRUCK_KEYS.items()}} for t in trucks],
                      columns=list(TRUCK_COLS))
    cfg = {c: st.column_config.NumberColumn(label) for c, label in TRUCK_COLS.items() if c not in ("id", "model")}
    cfg["id"] = st.column_config.TextColumn("ID")
    cfg["model"] = st.column_config.SelectboxColumn("model", options=_models("truck", [t.get("model", "") for t in trucks]),
                                                   width="large")
    ed = st.data_editor(df, key=key, num_rows="dynamic", hide_index=True, width="stretch", column_config=cfg)
    out, used = [], set()
    for i, (_, r) in enumerate(ed.iterrows()):
        old = trucks[i] if i < len(trucks) else {}
        t = copy.deepcopy(old)
        t.update({k: _num(r[c], d) for c, (k, d) in TRUCK_KEYS.items()})
        t = _apply_model("truck", t, old.get("model", ""), str(r["model"] or ""))
        tid = str(r["id"] or "").strip() or old.get("id") or ""
        n = len(out) + 1
        while not tid or tid in used:
            tid, n = f"WZ-{n:02d}", n + 1
        used.add(tid)
        t.update({"id": tid, "count": 1})
        t.setdefault("excavators", [])
        t.setdefault("drive_efficiency", 0.75)
        out.append(t)
    return out


EX_COLS = {"id": "ID", "model": "model", "lyzka_m3": "łyżka [m³]", "cykl_s": "cykl łyżki [s]",
           "wypelnienie": "wypełnienie łyżki", "podstawienie_s": "podstawienie wozidła [s]",
           "obszary": "może pracować w (puste = wszędzie)"}
EX_KEYS = {"lyzka_m3": ("bucket_m3", 4.0), "cykl_s": ("swing_cycle_s", 30.0), "wypelnienie": ("fill_factor", 0.9),
           "podstawienie_s": ("spot_time_s", 40.0)}


def excavator_editor(exs: list[dict], key: str, area_ids: list[str], models: dict[str, str]) -> list[dict]:
    """Excavators / loaders (added on the map): model from the catalogue, bucket, cycle, fill, spotting time and the
    areas each may work in. `models` = the model of every excavator (kept on the map)."""
    df = pd.DataFrame([{"id": e["id"], "model": e.get("model") or models.get(e["id"]) or OWN,
                        **{c: e.get(k, d) for c, (k, d) in EX_KEYS.items()},
                        "obszary": ", ".join(a for a in (e.get("areas") or []) if a in area_ids)} for e in exs],
                      columns=list(EX_COLS))
    cfg = {c: st.column_config.NumberColumn(label) for c, label in EX_COLS.items()
           if c not in ("id", "model", "obszary")}
    cfg["id"] = st.column_config.TextColumn("ID")
    cfg["model"] = st.column_config.SelectboxColumn(
        "model", options=_models("excavator", [str(x) for x in df["model"]]), width="large")
    cfg["obszary"] = st.column_config.TextColumn(EX_COLS["obszary"], help="Np. O1, O2 – obszary z mapy.")
    ed = st.data_editor(df, key=key, hide_index=True, width="stretch", disabled=["id"], column_config=cfg)
    out = []
    for e, (_, r) in zip(exs, ed.iterrows()):
        x = copy.deepcopy(e)
        x.update({k: _num(r[c], d) for c, (k, d) in EX_KEYS.items()})
        x = _apply_model("excavator", x, e.get("model") or models.get(e["id"], ""), str(r["model"] or ""))
        x["areas"] = [a.strip() for a in str(r["obszary"] or "").replace(";", ",").split(",") if a.strip() in area_ids]
        out.append(x)
    return out


def crusher_editor(crushers: list[dict], key: str) -> list[dict]:
    df = pd.DataFrame([{"id": c.get("id"), "model": c.get("model") or OWN, "wydajnosc": c.get("capacity_tph"),
                        "kosz": c.get("hopper_t"), "nadawa_mm": c.get("max_feed_mm", 900.0),
                        "wyladunek_s": c.get("dump_time_s", 60.0), "kwh_t": c.get("kwh_per_t", 1.1)} for c in crushers])
    cfg = {"id": "ID", "wydajnosc": "wydajność [t/h]", "kosz": "kosz [t]", "nadawa_mm": "max nadawa [mm]",
           "wyladunek_s": "wyładunek [s]", "kwh_t": "energia [kWh/t]",
           "model": st.column_config.SelectboxColumn("model", options=_models("crusher", [str(x) for x in df["model"]]),
                                                     width="large")}
    ed = st.data_editor(df, key=key, hide_index=True, width="stretch", disabled=["id"], column_config=cfg)
    out = []
    for c, (_, r) in zip(crushers, ed.iterrows()):
        x = copy.deepcopy(c)
        x.update({"capacity_tph": _num(r["wydajnosc"], 400), "hopper_t": _num(r["kosz"], 100),
                  "max_feed_mm": _num(r["nadawa_mm"], 900), "dump_time_s": _num(r["wyladunek_s"], 60),
                  "kwh_per_t": _num(r["kwh_t"], 1.1)})
        out.append(_apply_model("crusher", x, c.get("model", ""), str(r["model"] or "")))
    return out


def model_changed(items: list[dict]) -> bool:
    """A model was just picked in one of the rows (the flags are removed)."""
    hit = False
    for x in items:
        hit = bool(x.pop(CHANGED, False)) or hit
    return hit


def map_models(raw: dict) -> dict[str, str]:
    return {e["id"]: e.get("model", "") for e in (raw.get("site") or {}).get("excavators", [])}


def sync_site(raw: dict) -> None:
    """The map keeps the excavators' data too (it rebuilds the scenario from it)."""
    for se in (raw.get("site") or {}).get("excavators", []):
        x = next((e for e in raw.get("excavators", []) if e["id"] == se.get("id")), None)
        if x:
            se.update({k: x[k] for k in ("bucket_m3", "swing_cycle_s", "fill_factor", "spot_time_s", "areas", "model")
                       if k in x})
    site = raw.get("site") or {}
    if site.get("crusher") and raw.get("crusher"):
        site["crusher"]["model"] = raw["crusher"].get("model", site["crusher"].get("model", ""))


def variant_raw(raw: dict, work: dict) -> dict:
    """The mine as a variant has it: its own trucks / excavator parameters when it changed them."""
    m = (work or {}).get("machines") or {}
    if not m:
        return raw
    new = copy.deepcopy(raw)
    if m.get("trucks") is not None:
        new["trucks"] = copy.deepcopy(m["trucks"])
    for e in m.get("excavators") or []:
        for tgt in new.get("excavators", []) + (new.get("site") or {}).get("excavators", []):
            if tgt.get("id") == e["id"]:
                tgt.update({k: e[k] for k in ("bucket_m3", "swing_cycle_s", "fill_factor", "spot_time_s", "model")
                            if k in e})
    return new
