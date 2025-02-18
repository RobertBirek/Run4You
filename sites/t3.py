import streamlit as st
from datetime import datetime
import os
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Foldery lokalne
LOCAL_DATA_FOLDER = Path("data/")
LOCAL_RAW_FOLDER = LOCAL_DATA_FOLDER / "raw/"
LOCAL_CURRENT_FOLDER = LOCAL_DATA_FOLDER / "current/"
LOCAL_BACKUP_FOLDER = LOCAL_DATA_FOLDER / "backup/"

MARGED_CLEAN_FILE="all_cleaned.csv"


####################################
def show_page():
    gender_options = ["Kobieta", "Mężczyzna"]
    current_year = datetime.now().year
    years = list(range(1923, current_year - 15))
    
    if "predicted" in st.session_state:
            predicted = st.session_state["predicted"]
            

    st.title("Oto Twój wynik")
    # st.header("Dla danych:")
    st.subheader("Startujesz w Półmaratonie Wrocławskim")
    st.subheader("Bieg ma długość: 21,0975 km")
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
            <h1 style="font-size: 125px; text-align: center; color: gold;">{predicted["place"]}</h1>
            """,
            unsafe_allow_html=True,
        )
    with cc2:
        st.image("sites/puchar.png")

    cg1,cg2,cg3 = st.columns([3,6,3])
    with cg2:
        st.title("GRATULUJEMY!")

    st.subheader("Twoje wyniki:")
    
    st.markdown(
        f"""<table width="100%">
        <tr><th>Nazwa</th><th>Twoje dane</th><th>Najlepsze w Twojej grupie</th></tr>
        <tr><td>Czas na 5 km:</td><td>{predicted['time_5k']}</td><td>{predicted['time_5k_min']}</td></tr>
        <tr><td>Tempo na 5 km:</td><td>{predicted['tempo_5k']}</td><td>{predicted['tempo_5k_min']}</td></tr>
        <tr><td>Miejcse na 5 km:</td><td>{predicted['place_5k']}</td><td>{predicted['place_5k_min']}</td></tr>
        <tr><td>Czas na mecie:</td><td>{predicted['time']}</td><td>{predicted['time_min']}</td></tr>
        <tr><td>Miejsce na mecie:</td><td>{predicted['place']}</td><td>{predicted['place_min']}</td></tr>
        <tr><td>Miejsce w grupie:</td><td>{predicted['place_group']}</td><td>{predicted['place_group_min']}</td></tr>
        </table>
        """,
        unsafe_allow_html=True,
    )

    file = os.path.join(LOCAL_CURRENT_FOLDER, MARGED_CLEAN_FILE)
    df = pd.read_csv(file, sep=";")

    # Dane użytkownika
    user_czas = predicted['time_sek']  # Czas użytkownika w sekundach
    user_rocznik = predicted['yborn'] # Rok urodzenia użytkownika

    # Tworzenie DataFrame dla użytkownika
    df_user = pd.DataFrame({"czas": [user_czas], "rocznik": [user_rocznik]})
    
    fig, ax = plt.subplots(figsize=(10, 5))
    # Cały maraton
    sns.scatterplot(data=df, x="czas", y="rocznik", ax=ax, color="red", linewidth=0, alpha=0.5, label="Cały maraton")
    # Twoja pozycja
    sns.scatterplot(data=df_user, x="czas", y="rocznik", ax=ax, color="blue", linewidth=0, alpha=1, label="Ty")
    # Dodanie tytułu i legendy
    ax.set_title("Twoje miejsce w maratonie")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)

    # Wyświetlenie odpowiedzi AI
    # st.write(st.session_state["predicted"])

    if predicted["motivation"]:
        st.subheader("Twoja indywidualna motywacja:")
        st.markdown(predicted["motivation"], unsafe_allow_html=True)