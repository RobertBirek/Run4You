import streamlit as st

def show_page():
    c1,c2 = st.columns([8,3])
    with c1:
        st.title("Sprawdź jak poradzisz sobie w maratonie")
        
        
    with c2:
        st.image("sites/puchar.png")

    st.header("Podaj swoje parametry, aby przewidzieć swoje miejsce na mecie. 🏁")
    st.text("Na podstawie wyników biegu z półmaratonu Wrocławskiego z lat 2023 - 2024")
    if st.button("Zaczynamy", use_container_width=True, type="primary"):
        if "active_tab" not in st.session_state:
            st.session_state.active_tab = "t2"
        elif st.session_state.active_tab != "t2":
            st.session_state.active_tab = "t2"
            st.rerun()