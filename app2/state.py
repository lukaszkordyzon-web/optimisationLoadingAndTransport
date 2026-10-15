"""The mine being worked on (session state), shared by all screens of Kruszyk 2."""
from __future__ import annotations

import copy
from pathlib import Path

import streamlit as st
import yaml

from kruszyk.config import scenario_from_dict, yaml_dump

SCEN_DIR = Path(__file__).parent.parent / "scenarios"
DEFAULT = "kruszyk_demo_zakrzowek"  # the mine opened at start
# results that belong to one mine / one schedule – dropped when the mine changes
DERIVED = ("work", "variants", "k2seg_variant", "k2_cmp", "approved", "month", "split", "site", "site_ver", "info_sel")


def presets() -> dict[str, Path]:
    return {yaml.safe_load(p.read_text(encoding="utf-8")).get("name", p.stem): p for p in sorted(SCEN_DIR.glob("*.yaml"))}


def init() -> None:
    if "raw" not in st.session_state:
        st.session_state.raw = yaml.safe_load((SCEN_DIR / f"{DEFAULT}.yaml").read_text(encoding="utf-8"))
        st.session_state.ver = 0


def set_raw(raw: dict, new_mine: bool = False) -> None:
    """A new version of the mine (the map, the machines): widgets reload, a new mine drops all results."""
    st.session_state.raw = raw
    st.session_state.ver = st.session_state.get("ver", 0) + 1
    if new_mine:
        for k in DERIVED:
            st.session_state.pop(k, None)


def raw() -> dict:
    return st.session_state.raw


def raw_yaml() -> str:
    return yaml_dump(st.session_state.raw)


def scenario():
    return scenario_from_dict(copy.deepcopy(st.session_state.raw))


def export() -> str:
    """The mine with the planner's schedule (to save and open later)."""
    data = copy.deepcopy(st.session_state.raw)
    if st.session_state.get("work") is not None:
        data["k2_schedule"] = {"work": st.session_state.work, "plan": st.session_state.get("k2_plan", {}),
                               "variants": st.session_state.get("variants") or {},
                               "variant": st.session_state.get("k2seg_variant") or "A"}
    return yaml_dump(data)


def load(text: str) -> None:
    data = yaml.safe_load(text) or {}
    sched = data.pop("k2_schedule", None)
    data.pop("_plik", None)
    set_raw(data, new_mine=True)
    if sched:
        if sched.get("variants"):
            st.session_state.variants = sched["variants"]
            st.session_state.k2seg_variant = sched.get("variant") or next(iter(sched["variants"]))
        elif sched.get("work"):
            st.session_state.work = sched["work"]
        st.session_state.k2_plan = sched.get("plan", {})
