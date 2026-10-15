"""The look of Kruszyk 2 – the same as the design: navy top bar with step pills, IBM Plex type, light grey page,
white cards, red primary buttons, mono numbers."""
from __future__ import annotations

import html

import streamlit as st

INK, MUTED, LINE, SOFT, RED, NAVY = "#1B2733", "#4A5562", "#D9DDE3", "#F7F8FA", "#B3122A", "#1F2D3D"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');
html, body, .stApp, .stApp p, .stApp label, .stApp input, .stApp select, .stApp textarea, .stApp button,
.stApp [data-testid="stMarkdownContainer"] {{ font-family: 'IBM Plex Sans', sans-serif; }}
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {{ background: #F3F4F6 !important; color: {INK}; }}
body {{ overflow-x: hidden; }}
[data-testid="stHeader"], [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"],
[data-testid="stToolbar"], [data-testid="stDecoration"] {{ display: none !important; }}
[data-testid="stMainBlockContainer"] {{ max-width: 1376px; padding: 0 32px 48px 32px; }}
/* top bar: full width navy */
.st-key-k2_top {{ background: {NAVY}; width: 100vw !important; max-width: none !important; position: relative; left: 50%;
  margin-left: -50vw;
  padding: 10px max(32px, calc((100vw - 1376px) / 2 + 32px)); margin-bottom: 8px; }}
.st-key-k2_top p, .st-key-k2_top span {{ color: #FFFFFF; }}
.k2-brand {{ font-weight: 700; font-size: 18px; letter-spacing: .02em; color: #fff; padding-top: 6px; }}
.k2-mine {{ font-size: 13px; color: #C9D2DC !important; text-align: right; padding-top: 10px; }}
.st-key-k2_nav [data-testid="stRadioGroup"], .st-key-k2_nav [role="radiogroup"] {{ gap: 6px; flex-wrap: wrap; }}
.st-key-k2_nav [data-testid="stRadioOption"] {{ margin: 0 !important; padding: 7px 14px; border-radius: 999px; cursor: pointer; }}
.st-key-k2_nav [data-testid="stRadioOption"] > div > div:not([data-testid]) {{ display: none; }}
.st-key-k2_nav [data-testid="stRadioOption"] p {{ color: #C9D2DC !important; font-size: 14px; }}
.st-key-k2_nav [data-testid="stRadioOption"][data-selected="true"] {{ background: #FFFFFF; }}
.st-key-k2_nav [data-testid="stRadioOption"][data-selected="true"] p {{ color: {INK} !important; font-weight: 600; }}
.st-key-k2_file button {{ background: transparent; color: #C9D2DC; border: 1px solid #3D5269; }}
/* page header */
.k2-h1 {{ margin: 18px 0 0; font-size: 28px; font-weight: 700; color: {INK}; line-height: 1.25; }}
.k2-sub {{ margin: 6px 0 4px; color: {MUTED}; font-size: 15px; }}
.k2-h4 {{ font-weight: 600; font-size: 15px; color: {INK}; margin: 0 0 6px; }}
.k2-h2 {{ font-weight: 700; font-size: 18px; color: {INK}; margin: 0 0 6px; }}
.k2-note {{ font-size: 13px; color: {MUTED}; }}
/* cards */
div[class*="st-key-card_k2"] {{ background: #FFFFFF; border: 1px solid {LINE} !important; border-radius: 12px;
  box-shadow: none; padding: 18px 22px; }}
/* buttons */
.stApp [data-testid="stBaseButton-primary"] {{ background: {RED}; border-color: {RED}; color: #fff; font-weight: 600;
  border-radius: 8px; }}
.stApp [data-testid="stBaseButton-primary"]:hover {{ background: #8C0E20; border-color: #8C0E20; }}
.stApp [data-testid="stBaseButton-secondary"] {{ border-radius: 8px; border-color: #C3C9D1; color: {INK}; }}
/* inputs */
.stApp [data-baseweb="input"] input, .stApp [data-baseweb="select"] > div {{ font-family: 'IBM Plex Mono', monospace; }}
.stApp [data-testid="stWidgetLabel"] p {{ font-size: 13px; color: {MUTED}; }}
/* segmented radios (Dzień / Zmiana, Ortofoto / Plan) */
div[class*="st-key-k2seg"] [role="radiogroup"] {{ gap: 0; border: 1px solid #C3C9D1; border-radius: 8px; overflow: hidden;
  width: fit-content; }}
div[class*="st-key-k2seg"] [data-testid="stRadioOption"] {{ margin: 0 !important; padding: 7px 16px; background: #fff; cursor: pointer; }}
div[class*="st-key-k2seg"] [data-testid="stRadioOption"] > div > div:not([data-testid]) {{ display: none; }}
div[class*="st-key-k2seg"] [data-testid="stRadioOption"] p {{ font-weight: 600; font-size: 14px; color: {INK}; }}
div[class*="st-key-k2seg"] [data-testid="stRadioOption"][data-selected="true"] {{ background: {INK}; }}
div[class*="st-key-k2seg"] [data-testid="stRadioOption"][data-selected="true"] p {{ color: #fff !important; }}
/* underline tabs (Mapa / Maszyny) */
.st-key-k2_mine_tab [role="radiogroup"] {{ gap: 4px; border-bottom: 2px solid {LINE}; width: fit-content; margin-left: auto; }}
.st-key-k2_mine_tab [data-testid="stRadioOption"] {{ margin: 0 !important; padding: 8px 18px; cursor: pointer; }}
.st-key-k2_mine_tab [data-testid="stRadioOption"] > div > div:not([data-testid]) {{ display: none; }}
.st-key-k2_mine_tab [data-testid="stRadioOption"] p {{ font-size: 15px; color: {MUTED}; }}
.st-key-k2_mine_tab [data-testid="stRadioOption"][data-selected="true"] {{ border-bottom: 3px solid {RED}; margin-bottom: -2px !important; }}
.st-key-k2_mine_tab [data-testid="stRadioOption"][data-selected="true"] p {{ color: {INK} !important; font-weight: 700; }}
/* "Dodaj" list on the map: plain rows, the chosen one dark */
.st-key-k2_add [data-testid="stRadioOption"] {{ margin: 0 0 6px 0 !important; padding: 9px 12px; border-radius: 8px;
  border: 1px solid #C3C9D1; width: 100%; cursor: pointer; }}
.st-key-k2_add [data-testid="stRadioOption"] > div > div:not([data-testid]) {{ display: none; }}
.st-key-k2_add [data-testid="stRadioOption"] p {{ font-size: 14px; font-weight: 500; color: {INK}; }}
.st-key-k2_add [data-testid="stRadioOption"][data-selected="true"] {{ background: {INK}; border-color: {INK}; }}
.st-key-k2_add [data-testid="stRadioOption"][data-selected="true"] p {{ color: #fff !important; }}
.st-key-k2_add [data-testid="stRadioGroup"] {{ width: 100%; }}
/* KPI tiles */
.stApp [data-testid="stMetric"] {{ background: #FFFFFF; border: 1px solid {LINE}; border-radius: 12px; padding: 16px 20px; box-shadow: none; }}
.stApp [data-testid="stMetricLabel"] p {{ font-size: 13px; color: {MUTED}; }}
.stApp [data-testid="stMetricValue"] {{ font-family: 'IBM Plex Mono', monospace; font-weight: 600; font-size: 28px; color: {INK}; }}
/* html tables */
table.k2t {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
table.k2t th {{ text-align: left; color: {MUTED}; font-size: 12px; text-transform: uppercase; letter-spacing: .04em;
  padding: 6px 8px; border-bottom: 1px solid {LINE}; font-weight: 600; }}
table.k2t td {{ padding: 8px; border-bottom: 1px solid #EEF0F3; font-family: 'IBM Plex Mono', monospace; color: {INK}; }}
table.k2t td.n, table.k2t th.n {{ text-align: right; }}
table.k2t td.s {{ font-family: 'IBM Plex Sans', sans-serif; font-weight: 600; }}
.k2-chip {{ display: inline-block; border-radius: 999px; padding: 5px 12px; font-size: 13px; font-weight: 600; }}
.k2-chip.ok {{ background: #E3F1E6; color: #1E5A2C; }}
.k2-chip.warn {{ background: #FDE2E2; color: #8C0E20; }}
.k2-box {{ background: {SOFT}; border-radius: 8px; padding: 10px 12px; font-size: 14px; color: #26323F; margin-bottom: 8px; }}
.k2-issue {{ background: #FDE2E2; border: 1px solid #E8A0A8; border-radius: 8px; padding: 10px 12px; font-size: 14px;
  color: #5E0A16; margin-bottom: 6px; }}
.k2-issue.w {{ background: #FFF7E0; border-color: #E8CF8A; color: #5A4300; }}
.k2-darkbar {{ background: {NAVY}; color: #fff; border-radius: 12px; padding: 14px 20px; display: flex; flex-wrap: wrap;
  gap: 10px 28px; align-items: center; font-weight: 600; }}
[data-testid="stPlotlyChart"] {{ background: #fff; border-radius: 8px; }}
</style>
"""


def style() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def header(title: str, sub: str = "") -> None:
    st.markdown(f'<div class="k2-h1">{html.escape(title)}</div>' + (f'<div class="k2-sub">{sub}</div>' if sub else ""),
                unsafe_allow_html=True)


def h4(text: str, note: str = "") -> None:
    st.markdown(f'<div class="k2-h4">{text}' + (f' <span class="k2-note" style="font-weight:400">· {note}</span>'
                                                 if note else "") + "</div>", unsafe_allow_html=True)


def h2(text: str) -> None:
    st.markdown(f'<div class="k2-h2">{text}</div>', unsafe_allow_html=True)


def note(text: str) -> None:
    st.markdown(f'<div class="k2-note">{text}</div>', unsafe_allow_html=True)


def chip(text: str, ok: bool = True) -> str:
    return f'<span class="k2-chip {"ok" if ok else "warn"}">{html.escape(text)}</span>'


def table(cols: list[tuple[str, str]], rows: list[list], numeric: set[int] = frozenset(), sans: set[int] = frozenset()) -> str:
    """A table as in the design: cols = [(title, align)], rows of values."""
    head = "".join(f'<th class="{"n" if i in numeric else ""}">{html.escape(c)}</th>' for i, (c, _) in enumerate(cols))
    body = "".join("<tr>" + "".join(
        f'<td class="{"n" if i in numeric else ""}{" s" if i in sans else ""}">{html.escape(str(v))}</td>'
        for i, v in enumerate(r)) + "</tr>" for r in rows)
    return f'<div style="overflow-x:auto"><table class="k2t"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def card(name: str):
    return st.container(border=True, key=f"card_k2{name}")
