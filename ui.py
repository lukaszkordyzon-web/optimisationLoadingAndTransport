"""Look of the app: a grey page with white cards, clear step tabs and tables with a dark header row."""
from __future__ import annotations

import streamlit as st

PAGE = "#e1e6ed"
CSS = f"""
<style>
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {{ background: {PAGE} !important; }}
[data-testid="stHeader"] {{ background: {PAGE} !important; }}
[data-testid="stMainBlockContainer"] {{ padding-top: 3.6rem; }}
/* cards: st.container(border=True) made with card() */
div[class*="st-key-card"] {{
  background: #ffffff; border: 1px solid #d3dae3 !important; border-radius: 0.75rem;
  box-shadow: 0 1px 3px rgba(16, 24, 40, 0.08); padding: 1rem 1.2rem 1.1rem 1.2rem;
}}
/* step tabs: large pills */
[data-testid="stTabs"] [role="tablist"] {{ gap: 0.4rem; flex-wrap: wrap; border-bottom: none; }}
[data-testid="stTab"] {{
  background: #ffffff; border: 1px solid #d3dae3; border-radius: 999px; padding: 0.35rem 1rem; height: auto;
}}
[data-testid="stTab"] p {{ font-size: 0.98rem; font-weight: 600; }}
[data-testid="stTab"][aria-selected="true"] {{ background: #1f2d3d; border-color: #1f2d3d; }}
[data-testid="stTab"][aria-selected="true"] p {{ color: #ffffff; }}
[data-testid="stTabs"] [data-baseweb="tab-highlight"], [data-testid="stTabs"] [data-baseweb="tab-border"] {{ display: none; }}
/* module switch */
.st-key-work_kind [data-testid="stRadioGroup"] {{ gap: 0.5rem; }}
.st-key-work_kind [data-testid="stRadioGroup"] > * {{
  background: #ffffff; border: 1px solid #d3dae3; border-radius: 0.6rem; padding: 0.45rem 0.9rem; margin: 0;
}}
.st-key-work_kind [data-testid="stRadioOption"] p {{ font-size: 1.02rem; font-weight: 650; }}
/* step titles */
[data-testid="stTabPanel"] h3 {{ border-left: 6px solid #c8102e; padding-left: 0.7rem; margin-top: 0.4rem; }}
/* KPI tiles */
[data-testid="stMetric"] {{
  background: #ffffff; border: 1px solid #d3dae3; border-radius: 0.75rem; padding: 0.7rem 0.9rem;
  box-shadow: 0 1px 2px rgba(16, 24, 40, 0.06);
}}
[data-testid="stMetricValue"] {{ font-size: 1.7rem; }}
/* tables, expanders and charts stand out from the page */
[data-testid="stDataFrame"] {{ box-shadow: 0 1px 3px rgba(16, 24, 40, 0.10); border-radius: 0.5rem; }}
[data-testid="stMain"] [data-testid="stExpander"] details {{ background: #ffffff; }}
[data-testid="stMain"] [data-testid="stMarkdownContainer"] table {{ background: #ffffff; border-radius: 0.5rem; }}
[data-testid="stMain"] [data-testid="stMarkdownContainer"] th {{ background: #2f4156; color: #ffffff; }}
[data-testid="stPlotlyChart"] {{ background: #ffffff; border-radius: 0.75rem; padding: 0.3rem; }}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {{ color: #b8c4d1; }}
[data-testid="stSidebar"] [data-testid="stExpander"] details {{ background: #2c3e52; border-color: #3d5269; }}
</style>
"""


def style() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def card(name: str):
    """A white card on the grey page (use as `with card("rules_goal"):`)."""
    return st.container(border=True, key=f"card_{name}")
