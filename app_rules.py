"""Step "Rules of the shift": what the shift has to deliver and how the mine works today – one screen, in plain words.

These rules describe today's practice (the reference). The learned policy (RL) keeps the rules marked as binding
and may change the others when that is cheaper – the result shows what it would change.
"""
from __future__ import annotations

import copy

import streamlit as st

from ui import card

from kruszyk.goal import GOAL_ITEMS, default_yields, members

NO_GOAL, FEED, PRODUCT = "Ile się da (bez celu)", "Tony przez zakład", "Tony produktu"


def goal_feed_t(raw: dict, item: str, tonnes: float) -> tuple[float, float]:
    """Feed the plant needs for `tonnes` of a product: (feed t, yield %). Lines weighted by their capacity."""
    stored = (raw.get("production") or {}).get("yields") or {}
    site = raw.get("site") or {}
    main = raw.get("crusher") or {}
    lines = [(main.get("id") or "C-1", float(main.get("capacity_tph") or 400), (site.get("crusher") or {}).get("product")
              or main.get("product", ""))]
    lines += [(c["id"], float(c.get("capacity_tph") or 400), c.get("product", "")) for c in site.get("crushers", [])]
    num = den = 0.0
    for cid, cap, product in lines:
        y = stored.get(cid) or default_yields(product)
        num += cap * sum(float(y.get(p, 0.0)) for p in members(item))
        den += cap
    share = num / den if den else 0.0
    return (tonnes / (share / 100) if share > 0 else 0.0), share


def step_rules(raw: dict, set_raw) -> None:
    from app_map import get_site
    from kruszyk.config import expand_trucks
    from kruszyk.site import site_to_scenario

    st.markdown("### ② Plan zmiany (reguły)")
    st.markdown("Jak dziś pracuje zmiana: **ile ma dać**, **która koparka gdzie ładuje** i **które wozidła dokąd "
                "jeżdżą**. To punkt odniesienia dla „Jak było dziś”. Optymalizacja RL trzyma się reguł oznaczonych "
                "🔒, a pozostałe może zmienić, jeśli tak jest taniej – i pokaże, co zmieniła.")
    site = get_site(raw)
    excavators = site.get("excavators", [])
    piles = [p["id"] for p in site.get("piles", [])]
    if not excavators or not piles:
        st.info("Najpierw narysuj kopalnię w kroku **① Stan wyjściowy kopalni** (usypy, koparki, drogi) i zapisz ją.")
        return
    ver = st.session_state.get("ver", 0)
    k = lambda name: f"rules_{name}_{ver}"  # noqa: E731

    # ---------------------------------------------------------------- goal
    with card("rules_goal"):
        st.markdown("#### 🎯 Cel zmiany")
        g = raw.get("shift_goal") or {}
        kinds = [NO_GOAL, FEED, PRODUCT]
        kind = st.radio("Cel", kinds, index={"feed": 1, "product": 2}.get(g.get("kind"), 0), horizontal=True,
                        key=k("goal_kind"), label_visibility="collapsed")
        goal = {}
        if kind == FEED:
            t = st.number_input("Ile ton nadawy ma przejść przez zakład [t]", 100.0, 50_000.0,
                                float(g.get("t") or 2000.0) if g.get("kind") == "feed" else 2000.0, 100.0, key=k("goal_t"))
            goal = {"kind": "feed", "t": t, "feed_t": t}
        elif kind == PRODUCT:
            c = st.columns([2, 1])
            item = c[0].selectbox("Produkt", GOAL_ITEMS, index=GOAL_ITEMS.index(g["item"]) if g.get("item") in GOAL_ITEMS
                                  else 0, key=k("goal_item"))
            t = c[1].number_input("Ile ton produktu [t]", 10.0, 50_000.0,
                                  float(g.get("t") or 800.0) if g.get("kind") == "product" else 800.0, 50.0, key=k("goal_pt"))
            feed, share = goal_feed_t(raw, item, t)
            if feed:
                st.caption(f"Z każdej tony nadawy linie dają średnio **{share:.0f} %** produktu „{item}” → potrzeba "
                           f"ok. **{feed:,.0f} t nadawy**. Uzyski linii ustawisz w Planowaniu produkcji → 🎯 Cel "
                           "produkcyjny.".replace(",", " "))
                goal = {"kind": "product", "item": item, "t": t, "feed_t": round(feed)}
            else:
                st.warning(f"Żadna kruszarka nie produkuje „{item}” – ustaw produkty / uzyski linii.")
        if goal:
            goal["stop"] = st.checkbox("Po osiągnięciu celu wozidła kończą wożenie (parkują)", bool(g.get("stop", True)),
                                       key=k("goal_stop"), help="Bez zaznaczenia wożą do końca zmiany – cel to minimum.")
        else:
            st.caption("Bez celu: zmiana wozi, ile zdąży; optymalizacja szuka najniższego kosztu na tonę.")

    # ---------------------------------------------------------------- excavators
    with card("rules_ex"):
        st.markdown("#### ⛏️ Koparki – gdzie ładują i dokąd przejeżdżają")
        st.caption("Gdy usyp się kończy (albo zostanie na nim mniej niż próg), koparka **w trakcie zmiany** "
                   "przejeżdża na wskazany usyp; resztę ładuje koparka, która została na starym.")
        h = st.columns([1, 2, 2, 2])
        for c_, t_ in zip(h, ("**koparka**", "**ładuje z usypu**", "**potem przejeżdża na**", "**gdy zostanie mniej niż [t]**")):
            c_.markdown(t_)
        ex_rules = {}
        for e in excavators:
            c = st.columns([1, 2, 2, 2])
            c[0].markdown(f"**{e['id']}**  \n<small>{e.get('model', '')}</small>", unsafe_allow_html=True)
            pile = c[1].selectbox("usyp", piles, index=piles.index(e["pile"]) if e.get("pile") in piles else 0,
                                  key=k(f"pile_{e['id']}"), label_visibility="collapsed")
            nxt_opts = ["— zostaje —"] + [p for p in piles if p != pile]
            nxt = c[2].selectbox("potem", nxt_opts, index=nxt_opts.index(e["then_pile"]) if e.get("then_pile") in nxt_opts
                                 else 0, key=k(f"then_{e['id']}"), label_visibility="collapsed")
            below = c[3].number_input("próg", 0.0, 100_000.0, float(e.get("then_below_t") or 0.0), 100.0,
                                      key=k(f"below_{e['id']}"), label_visibility="collapsed",
                                      disabled=nxt == nxt_opts[0])
            ex_rules[e["id"]] = (pile, "" if nxt == nxt_opts[0] else nxt, below)

    # ---------------------------------------------------------------- trucks
    trucks = expand_trucks(raw.get("trucks", []))
    tids = [t["id"] for t in trucks]
    with card("rules_trucks"):
        st.markdown("#### 🚚 Wozidła – kto dokąd jeździ")
        st.caption("Przypisz wozidła do koparek tak, jak robi to sztygar. Nieprzypisane sztygar dzieli po równo "
                   "między koparki. "
                   "🔒 = przydział sztywny (np. decyzja sztygara, osobny materiał); bez 🔒 RL może przesunąć wozidło, "
                   "gdy pod jedną koparką robi się kolejka, a druga czeka.")
        assign, locks = {}, {}
        for e in excavators:
            c = st.columns([5, 1])
            now = [t["id"] for t in trucks if e["id"] in (t.get("excavators") or [])]
            pile = ex_rules[e["id"]][0]
            assign[e["id"]] = c[0].multiselect(f"Do **{e['id']}** (usyp {pile}) jeżdżą", tids, now, key=k(f"tr_{e['id']}"))
            locked = any(t.get("assign_locked", True) for t in trucks if t["id"] in now) if now else False
            locks[e["id"]] = c[1].toggle("🔒", locked, key=k(f"lock_{e['id']}"), help="przydział sztywny")
        free = [t for t in tids if not any(t in v for v in assign.values())]
        st.caption(("Bez przypisania (podział po równo): **" + ", ".join(free) + "**") if free
                   else "Wszystkie wozidła są przypisane.")
        switch = st.slider("Czy kierowcy sami (bez polecenia sztygara) jeżdżą do innej koparki? [% kursów]", 0, 100,
                           int(round(100 * float(raw.get("natural_switch_prob", 0.0)))), 5, key=k("switch"),
                           help="zwykle 0: kierowca jeździ tam, gdzie przypisał go sztygar; zmienia miejsce przy awarii "
                                "koparki albo gdy usyp się skończy (polecenie przez radio)")

    # ---------------------------------------------------------------- feed mix
    with card("rules_feed"):
        st.markdown("#### 🧪 Skład nadawy – udział usypów")
        st.caption("Ile procent nadawy ma iść z danego usypu (puste / 0 = bez wymogu). Liczą się proporcje.")
        c = st.columns(max(len(piles), 1))
        shares = {}
        for i, p in enumerate(site.get("piles", [])):
            shares[p["id"]] = c[i % len(c)].number_input(
                f"{p['id']} ({float(p.get('tonnage_t') or 0):,.0f} t)".replace(",", " "), 0.0, 100.0,
                float(p.get("target_share") or 0.0), 5.0, key=k(f"share_{p['id']}"))

    if st.button("💾 Zastosuj reguły", type="primary", key="rules_apply"):
        site = copy.deepcopy(site)
        for e in site["excavators"]:
            pile, nxt, below = ex_rules[e["id"]]
            e["pile"] = pile
            if nxt:
                e["then_pile"], e["then_below_t"] = nxt, float(below)
            else:
                e.pop("then_pile", None)
                e.pop("then_below_t", None)
        for p in site["piles"]:
            if shares.get(p["id"]):
                p["target_share"] = float(shares[p["id"]])
            else:
                p.pop("target_share", None)
        new_trucks = []
        for t in trucks:
            exs = [x for x, ids in assign.items() if t["id"] in ids]
            locked = any(locks[x] for x in exs)
            new_trucks.append({**t, "excavators": exs, "assign_locked": locked if exs else True})
        new = copy.deepcopy(raw)
        new["trucks"] = new_trucks
        new["natural_switch_prob"] = switch / 100
        if goal:
            new["shift_goal"] = goal
        else:
            new.pop("shift_goal", None)
        try:
            new = site_to_scenario(site, new)
        except ValueError as e:
            st.error(str(e))
            return
        set_raw(new)
        st.session_state.rules_msg = True
        st.rerun()
    if st.session_state.pop("rules_msg", None):
        st.success("✅ Reguły zapisane. Zobacz efekt w **⑤ Jak było dziś**, a co zmieniłby system – w **⑥ Optymalizacja RL**.")
    else:
        st.caption("Zmiany wchodzą do obliczeń po kliknięciu **Zastosuj reguły**.")
