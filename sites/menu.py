import streamlit as st

###########################################
# Funkcja do ustawienia aktywnej zakładki
def set_active_tab(tab_name):
    st.session_state.active_tab = tab_name
###########################################

def show_menu():
    c1,c2,c3 = st.columns([3,1,3])
    with c2:
        st.image("logo.png", caption="Run4You", use_container_width=True)

    # Zapewnienie, że `active_tab` jest zainicjalizowane
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "t1"

    # Definicja zakładek
    tabs = [
        ("t1", "Start"),
        ("t2", "O Tobie"),
        ("t3", "Finish"),
        ("t4", "Ustawienia"),
    ]

    cols = st.columns(len(tabs))
    for i, (tab_key, tab_label) in enumerate(tabs):
        with cols[i]:
            is_active = st.session_state.active_tab == tab_key
            button_type = "primary" if is_active else "secondary"
            is_disabled = tab_key == "t3" and st.session_state.get("button_disabled", False)
            st.button(tab_label, on_click=set_active_tab, args=(tab_key,), use_container_width=True, type=button_type, disabled=is_disabled)

    st.write("---")
