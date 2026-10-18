"""① Kopalnia → Mapa: draw on the orthophoto, points stick to what is already drawn; the clicked object on the right."""
from __future__ import annotations

import copy

import folium
import streamlit as st
from streamlit_folium import st_folium

from kruszyk import catalog
from kruszyk.psd import DEFAULT_PRESET, PRESETS

from . import state, theme

ADD = {  # what the planner adds -> (mode of the map module (data, kept in Polish), drawing tool)
    "Road / ramp": ("Droga / rampa", "line"),
    "Passing bay": ("Mijanka", "marker on a road"),
    "Mining area": ("Obszar wydobycia (strzały automatycznie)", "polygon"),
    "Excavator": ("Koparka", "marker"),
    "Crusher": ("Kruszarka", "marker"),
}
DIRS = {"⬇ downhill": -1, "⬆ uphill": 1, "➡ flat": 0}
KIND_LABELS = {"droga": "road", "rampa": "ramp"}  # road kinds are data


def _attrs(kind: str, site: dict) -> dict:
    """The few settings of the object being added (the rest: defaults, editable after a click on it)."""
    from app_map import MATERIALS

    if kind == "Road / ramp":
        r = st.selectbox("Kind", ["droga", "rampa"], key="k2_add_kind", format_func=KIND_LABELS.get)
        d = st.radio("Drawing…", list(DIRS), index=0 if r == "rampa" else 2, horizontal=True, key="k2seg_dir")
        g = 0.0 if DIRS[d] == 0 else st.number_input("Grade [%]", 0.5, 25.0, 9.0 if r == "rampa" else 2.0, 0.5,
                                                       key="k2_add_grade")
        one = st.checkbox("Single lane (passing bays)", r == "rampa", key="k2_add_one")
        return {"kind": r, "name": f"{KIND_LABELS[r]} {len(site['roads']) + 1}", "grade_draw_pct": DIRS[d] * g + 0.0,
                "speed_limit_kmh": 25.0 if r == "rampa" else 40.0, "single_lane": one}
    if kind == "Mining area":
        size = st.number_input("Blast size [t]", 1000.0, 100_000.0, 12_000.0, 1000.0, key="k2_add_bt")
        tpm = st.number_input("t/m", 5.0, 200.0, 30.0, 1.0, key="k2_add_tpm")
        mat = st.selectbox("Material", MATERIALS, key="k2_add_amat")
        return {"material": mat, "blast_max_t": size, "t_per_m": tpm, "bench_m": 15.0, "solid_density_t_m3": 2.65,
                "first_drilled": False, "reblast_below_t": 0.0, "min_gap_days": 1, "max_per_day": 1,
                "psd": PRESETS[DEFAULT_PRESET], "loose_density_t_m3": 1.7}
    if kind == "Excavator":
        model = st.selectbox("Model", catalog.names("excavator"), key="k2_add_ex")
        return {"pile": "(najbliższy)", "bucket_m3": 5.0, "swing_cycle_s": 30.0, **catalog.params("excavator", model),
                "model": model}
    return {}


def map_screen() -> None:
    from app_map import (add_drawing, analyze_site, apply_edit_drawing, delete_obj, draw_network, draw_site,
                         editing_map, find_obj, get_site, object_editor, object_properties, road_points_editor,
                         sel_from_click)
    from kruszyk.site import site_to_scenario

    raw = state.raw()
    site = get_site(raw)
    st.session_state.setdefault("map_key", 0)
    left, mid, right = st.columns([1.05, 3.4, 1.75])
    with left, theme.card("map_add"):
        theme.h4("Add")
        kind = st.radio("What you add", list(ADD), key="k2_add", label_visibility="collapsed")
        attrs = _attrs(kind, site)
        theme.note(f"On the map: the <b>{ADD[kind][1]}</b> tool. Points snap to roads and objects.")
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        theme.h4("Machines")
        n_tr = sum(int(t.get("count", 1)) for t in raw.get("trucks", []))
        st.markdown(f"<div class='k2-note'>Trucks <b>{n_tr}</b> · excavators <b>{len(site.get('excavators', []))}</b> · "
                    f"crushers <b>{1 + len(site.get('crushers', []))}</b></div>", unsafe_allow_html=True)

    with mid:
        objects = folium.FeatureGroup(name="Objects")
        sel = st.session_state.get("info_sel")
        if sel and find_obj(site, sel) is not None:
            from app_map import highlight

            highlight(objects, site, sel)
        draw_site(objects, site, plans=True)
        info = analyze_site(site)
        draw_network(objects, info)
        out = st_folium(editing_map(site), height=640, use_container_width=True, key=f"map_{st.session_state.map_key}",
                        feature_group_to_add=objects,
                        returned_objects=["all_drawings", "last_object_clicked", "last_object_clicked_tooltip"]) or {}
        c = st.columns([3, 1.3])
        for w in info.get("warnings", [])[:3]:
            c[0].warning(w)
        if c[1].button("💾 Save mine", type="primary", key="k2_map_save", width="stretch"):
            try:
                new = site_to_scenario(copy.deepcopy(site), copy.deepcopy(raw))
                state.set_raw(new)
                st.session_state.pop("approved", None)
                st.session_state.k2_map_msg = "✅ Mine saved – routes recalculated."
                st.rerun()
            except ValueError as e:
                c[0].error(str(e))

    # a click on an object: show it on the right
    click = (out.get("last_object_clicked_tooltip"), repr(out.get("last_object_clicked")))
    if out.get("last_object_clicked") and click != st.session_state.get("last_click"):
        st.session_state.last_click = click
        hit = sel_from_click(site, click[0], out.get("last_object_clicked"))
        if hit:
            st.session_state.info_sel = hit
            st.rerun()
    # a new drawing: add it (or redraw the armed object)
    seen = st.session_state.setdefault("seen_drawings", set())
    fresh = [f for f in (out.get("all_drawings") or []) if f.get("geometry") and repr(f["geometry"]) not in seen]
    for f in fresh:
        seen.add(repr(f["geometry"]))
    if fresh:
        g = fresh[-1]["geometry"]
        arm = st.session_state.pop("k2_arm", None)
        msg = apply_edit_drawing(site, arm, g) if arm else add_drawing(site, g, ADD[kind][0], attrs)
        st.session_state.k2_map_msg = msg + (" Save the mine to include it in the calculations." if msg.startswith("✅") else "")
        st.session_state.obj_ver = st.session_state.get("obj_ver", 0) + 1
        st.rerun()

    with right, theme.card("map_obj"):
        msg = st.session_state.pop("k2_map_msg", None)
        if msg:
            (st.success if msg.startswith("✅") else st.warning)(msg)
        sel = st.session_state.get("info_sel")
        o = find_obj(site, sel) if sel else None
        if o is None:
            theme.h4("Selected object")
            theme.note("Click a road, muck pile, excavator or crusher on the map.")
            return
        props = object_properties(site, sel, raw)
        theme.h4(props[0][1])
        c = st.columns(2)
        if c[0].button("✏️ Redraw", key="k2_arm_btn", width="stretch",
                       help="draw the object again or place a marker in a new spot – it replaces the old one"):
            st.session_state.k2_arm = sel
            st.session_state.k2_map_msg = "Draw the object again on the map."
            st.rerun()
        if c[1].button("🗑️ Delete", key="k2_del_btn", width="stretch"):
            st.session_state.k2_map_msg = delete_obj(site, sel)
            st.session_state.pop("info_sel", None)
            st.session_state.obj_ver = st.session_state.get("obj_ver", 0) + 1
            st.rerun()
        object_editor(site, sel, raw)
        if sel.startswith("road:"):
            road_points_editor(o, analyze_site(site))
