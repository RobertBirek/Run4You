import os
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime
# from openai import OpenAI
# #from langfuse.decorators import observe
# #from langfuse.openai import OpenAI

# load_dotenv()

# # OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
# openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


gender_options = ["Kobieta", "Mężczyzna"]
current_year = datetime.now().year
years = list(range(1923, current_year - 15))

if "predicted" in st.session_state:
        predicted_name = st.session_state["predicted"].get("name", None)
        predicted_gender = st.session_state["predicted"].get("gender", None)
        predicted_yborn = st.session_state["predicted"].get("yborn", None)
        predicted_time5k = st.session_state["predicted"].get("time5k", None)
        predicted_place_5k = st.session_state["predicted"].get("user_place_5k", None)
        predicted_place_5k_min = st.session_state["predicted"].get("user_place_5k_min", None)
        predicted_tempo5k = st.session_state["predicted"].get("tempo5k", None)
        predicted_place = st.session_state["predicted"].get("place", None)
        predict_motivation = st.session_state["predicted"].get("motivation", None)


####################################
def show_page():
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
    st.subheader("Twoja indywidualna motywacja:")
    st.markdown(st.session_state["predicted"]["motivation"], unsafe_allow_html=True)