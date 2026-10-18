"""Kruszyk 2 – the same month driven today and with the RL driving signals.  Run: streamlit run kruszyk2.py"""
from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Kruszyk 2 – a month with and without RL signals", page_icon="⛏️", layout="wide",
                   initial_sidebar_state="collapsed")

from app2 import state, theme  # noqa: E402

STEPS = ["① Mine", "② Schedule", "③ Simulation", "④ Result"]

theme.style()
state.init()

if "nav_next" in st.session_state:  # set by a button (a widget's value can't be changed after it is drawn)
    st.session_state.k2_nav = st.session_state.pop("nav_next")

with st.container(key="k2_top"):
    c = st.columns([1.0, 5.2, 2.6, 0.9], vertical_alignment="center")
    c[0].markdown('<div class="k2-brand">Kruszyk</div>', unsafe_allow_html=True)
    with c[1]:
        step = st.radio("Step", STEPS, horizontal=True, key="k2_nav", label_visibility="collapsed")
    c[2].markdown(f'<div class="k2-mine">Mine: {state.raw().get("name", "unnamed")}</div>', unsafe_allow_html=True)
    with c[3], st.container(key="k2_file"), st.popover("File", width="stretch"):
        names = list(state.presets())
        pick = st.selectbox("Mine from the list", names, index=next((i for i, n in enumerate(names)
                                                                  if state.presets()[n].stem == state.DEFAULT), 0),
                            key="k2_preset")
        if st.button("Open", key="k2_open", width="stretch"):
            import yaml

            state.set_raw(yaml.safe_load(state.presets()[pick].read_text(encoding="utf-8")), new_mine=True)
            st.rerun()
        up = st.file_uploader("Load a saved mine (.yaml)", type=["yaml", "yml"], key="k2_upload")
        if up is not None and st.session_state.get("k2_uploaded") != up.file_id:
            st.session_state.k2_uploaded = up.file_id
            state.load(up.getvalue().decode("utf-8"))
            st.rerun()
        st.download_button("Save the mine and schedule", state.export().encode("utf-8"), "kopalnia.kruszyk2.yaml",
                           "text/yaml", key="k2_save", width="stretch")

if step == STEPS[0]:
    from app2.mine import step_mine

    step_mine()
elif step == STEPS[1]:
    from app2.plan import step_plan

    step_plan()
elif step == STEPS[2]:
    from app2.compare import step_simulation

    step_simulation()
else:
    from app2.compare import step_result

    step_result()
