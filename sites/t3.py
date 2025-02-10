import streamlit as st
from datetime import datetime


####################################
def show_page():
    gender_options = ["Kobieta", "Mężczyzna"]
    current_year = datetime.now().year
    years = list(range(1923, current_year - 15))
    if "predicted" in st.session_state:
        predicted_place_5k = st.session_state["predicted"]["user_place_5k"]
        predicted_tempo5k = st.session_state["predicted"]["tempo5k"]
        predicted_place = st.session_state["predicted"]["place"]
        predict_motivation = st.session_state["predicted"]["motivation"]
    st.title("Oto Twój wynik")
    # st.header("Dla danych:")
    st.subheader("Jeżeli przystąpisz do biegu, posiadając parametry:")
    c1, c2, c3, c4  = st.columns(4)
    with c1:
        st.metric(label="Twoje imię:", value=st.session_state.name)
    with c2:
        st.metric(label="Twoja płeć", value=gender_options[st.session_state.gender])
    with c3:
        st.metric(label="Twój rok urodzenia", value=st.session_state.yborn)
    with c4:
        st.metric(label="Twój czas/5km", value=st.session_state.time5km)

    st.subheader("Możesz ukończyć go na miejscu:")

    cc1, cc2  = st.columns([6,3])
    with cc1:
        st.markdown(
            f"""
            <h1 style="font-size: 125px; text-align: center; color: gold;">{predicted_place_5k}</h1>
            """,
            unsafe_allow_html=True,
        )
    with cc2:
        st.image("puchar.png")

    cg1,cg2,cg3 = st.columns([3,6,3])
    with cg2:
        st.title("GRATULUJEMY!")
    txt = f"Twoje tempo na 5km to : ~{predicted_tempo5k}, jeżeli utrzymasz je przez cały bieg, to ukończysz go na miejscu {predicted_place}."
    st.write(txt)
    # Wyświetlenie odpowiedzi AI
    if predict_motivation:
        st.subheader("Twoja indywidualna motywacja:")
        st.markdown(predict_motivation, unsafe_allow_html=True)