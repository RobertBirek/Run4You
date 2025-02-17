import streamlit as st
import pandas as pd
from datetime import datetime
import re
import os
from pathlib import Path
from dotenv import load_dotenv
from pycaret.regression import load_model, predict_model
# from openai import OpenAI 
from langfuse.decorators import observe
from langfuse.openai import OpenAI

load_dotenv()

# Połączenie z OpenAI
# openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Inicjalizacja `session_state` dla `predicted`
# if "predicted" not in st.session_state:
#     st.session_state["predicted"] = {}

# Opcje płci
gender_options = ["Kobieta", "Mężczyzna"]

# Generowanie listy lat urodzenia
current_year = datetime.now().year
years = list(range(1923, current_year - 15))

# Folder na modele
LOCAL_DATA_FOLDER = Path("data/")
LOCAL_RAW_FOLDER = LOCAL_DATA_FOLDER / "raw/"
LOCAL_CURRENT_FOLDER = LOCAL_DATA_FOLDER / "current/"
LOCAL_BACKUP_FOLDER = LOCAL_DATA_FOLDER / "backup/"

# Ścieżki do modeli
PATH_MODEL_PLACE_TIME5K = os.path.join(LOCAL_CURRENT_FOLDER, "model_place_time5k")
PATH_MODEL_PLACE5K_TIME5K = os.path.join(LOCAL_CURRENT_FOLDER, "model_place5k_time5k")
PATH_MODEL_TEMPO5K_TIME5K = os.path.join(LOCAL_CURRENT_FOLDER, "model_tempo5k_time5k")
PATH_MODEL_PLACE_TEMPO5K = os.path.join(LOCAL_CURRENT_FOLDER, "model_place_tempo5k")
PATH_MODEL_TIME_TIME5K = os.path.join(LOCAL_CURRENT_FOLDER, "model_time_time5k")
PATH_MODEL_PGROUP_TIME5K = os.path.join(LOCAL_CURRENT_FOLDER, "model_pgroup_time5k")

PATH_MODEL_TIME5K_TEMPO5K = os.path.join(LOCAL_CURRENT_FOLDER, "model_time5k_tempo5k")


####################################
def update_name():
    st.session_state.name = st.session_state.input_name
####################################
def update_gender():
    st.session_state.gender = gender_options.index(st.session_state.selected_gender)
####################################
def update_yborn():
    st.session_state.yborn = st.session_state.selected_yborn
####################################
def update_time():
    st.session_state.time5km = st.session_state.input_time5km
####################################
# Funkcja walidująca format czasu
def validate_time_format(input_text):
    if not input_text:  # Jeśli pole jest puste, zwróć False
        return False
    if input_text == "00:00:00":  # Jeśli wartość to "00:00:00", zwróć False
        return False
    # Sprawdzenie, czy format jest zgodny z hh:mm:ss
    # pattern = r"^\d{2}:\d{2}:\d{2}$"
    pattern = r"^(?:[01]\d|2[0-3]):[0-5]\d:[0-5]\d$"
    if re.match(pattern, input_text):
        return True
    return False
####################################
def validate_params():
    # Sprawdzenie, czy imię zostało podane
    if not st.session_state.get("name"):
        st.error("Musisz podać swoje imię.")
        return False
    
    # Sprawdzenie, czy płeć została wybrana (powinna być automatycznie poprawna z selectbox)
    if "gender" not in st.session_state or st.session_state.gender not in [0, 1]:
        st.error("Musisz wybrać swoją płeć.")
        return False

    # Sprawdzenie, czy rok urodzenia został wybrany
    if "yborn" not in st.session_state or not isinstance(st.session_state.yborn, int):
        st.error("Musisz wybrać swój rok urodzenia.")
        return False

    # Sprawdzenie, czy tempo jest w prawidłowym formacie
    if not st.session_state.get("time5km"):
        st.error("Musisz podać swój czas na 5km w formacie hh:mm:ss.")
        return False
    if not validate_time_format(st.session_state.time5km):
        st.error("Nieprawidłowy format czasu. Wprowadź czas w formacie hh:mm:ss.")
        return False

    # Wszystkie warunki spełnione
    return True
####################################
def time_to_seconds(time):
    try:
        h, m, s = map(int, time.split(':'))
        return h * 3600 + m * 60 + s
    except (ValueError, AttributeError):
        return None  # Zwraca None dla niepoprawnych wartości   
###########################################
def seconds_to_time(seconds):
    if pd.isnull(seconds):
        return None
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02}:{m:02}:{s:02}"
####################################
def predict_results(df,model_path,):
    # Sprawdzenie, czy model istnieje
    if not os.path.exists(model_path + ".pkl"):
        st.error("Model nie istnieje! Najpierw wygeneruj model.")
        return None
    
    try:
        # Wczytanie modelu
        model = load_model(model_path)
        
        # Wykonanie predykcji
        preds = predict_model(model, data=df)
        # print(preds.columns)
        
        # Sprawdzenie dostępnych kolumn w wyniku predykcji
        # st.write("📊 Kolumny predykcji:", preds.columns.tolist())

        # Pobranie przewidywanej wartości z `prediction_label`, jeśli istnieje
        if "prediction_label" in preds.columns:
            return preds["prediction_label"].iloc[0]
        else:
            st.error("Brak kolumny `prediction_label` w wyniku predykcji.")
            return None

    except Exception as e:
        st.error(f"Błąd podczas predykcji: {e}")
        return None
####################################
@observe()
def generate_ai_motivation(model: str = "gpt-4o-mini") -> str:
    
    try:
        if "predicted" in st.session_state:
            predicted = st.session_state["predicted"]


        persona = """
        Jesteś trenerem personalnym, który od 20 lat trenuje przyszłych marytończyków i maratonczyki,
        pomagając im osiągnąć ich cele.   
        """.strip()
        question = f"""
        Jednym z Twoich podopiecznych jest {predicted["name"]}, jest {predicted["gender"]}, jest z rocznika {predicted["yborn"]} który przygotowuje się do półmaratonu, który wynosi około 21 km.
        jego czas na 5 km wynosi {predicted["time_5k"]}, a najszybszy zawodnik w jego kategorii biegnie 5 km w czasie {predicted["time_5k_min"]}.
        Według Twoich obliczeń, {predicted["name"]} powinien ukończyć bieg na miejscu {predicted["place"]}, a w swojej kategorii na miejscu {predicted["place_group"]}.
        Jeżeli {predicted["name"]} utrzyma tempo {predicted["name"]} przez cały bieg, to powinien ukończyć bieg na miejscu {predicted["name"]}.
        Jakie rady mógłbyś dać {predicted["name"]} przed biegiem?
        Zmotywuj go do osiągnięcia jak najlepszego wyniku.
        odpowiedz w formacie markdown.
        """.strip()


        # Tworzenie zapytania
        messages = [
            {"role": "system", "content": persona},  # Persona określająca kontekst
            {"role": "user", "content": question}    # Zapytanie użytkownika
        ]

        # try:
        # Wysłanie zapytania do OpenAI
        response = openai_client.chat.completions.create(
            model=model,
            messages=messages,
            name="motivation"        )

        # Pobranie odpowiedzi
        answer = response.choices[0].message.content
        return answer

    except openai_client.OpenAIError as e:
        return f"Błąd OpenAI: {str(e)}"

####################################
def show_page():
    st.title("Podaj Swoje parametry")

    name = st.text_input(
        "Jak masz na imię?",
        value=st.session_state.name,
        key="input_name",
        on_change=update_name
    )
 
    gender = st.selectbox(
        "Podaj swoją płeć",
        gender_options,
        index=st.session_state.gender,
        key="selected_gender",
        on_change=update_gender,
    )

    yborn = st.selectbox(
        "Podaj Swój rok urodzenia",
        years,
        index=years.index(st.session_state.yborn),
        key="selected_yborn",
        on_change=update_yborn,
    )

    time5 = st.text_input(
        "Jaki jest Twój czas na 5km? (hh:mm:ss)",
        value=st.session_state.time5km,
        key="input_time5km",  # Przechowuje bieżącą wartość
        on_change=update_time,  # Funkcja wywoływana po zmianie
    )

    st.session_state.button_disabled = not validate_params()

    

    if st.button("Oblicz", use_container_width=True, type="primary", disabled=st.session_state.button_disabled):
        del st.session_state["predicted"]
        if "predicted" not in st.session_state:
            st.session_state["predicted"] = {}

        # Sprawdzenie, czy model istnieje
        if not os.path.exists(PATH_MODEL_PLACE_TIME5K + ".pkl"):
            st.error("Model nie istnieje! Najpierw wygeneruj model.")
            return None
        else:
            with st.spinner("Generowanie wyniku..."):
                # Konwersja płci (Kobieta = 0, Mężczyzna = 1)
                gender_numeric = 0 if gender == "Kobieta" else 1
                # Konwersja czasu na sekundy
                time5_seconds = time_to_seconds(time5) # Czas w hh:mm:ss
                year = datetime.now().year
                df_user = pd.DataFrame([{
                    'rok': year,
                    'płeć': gender_numeric,
                    'rocznik': yborn,
                    '5_km_czas': time5_seconds  
                }])
                
                df_user_min = pd.DataFrame([{
                    'rok': year,
                    'płeć': gender_numeric,
                    'rocznik': yborn,
                    '5_km_czas': 600  
                }])
                # Wykonanie predykcji
                predicted_place_time5k = predict_results(df_user,PATH_MODEL_PLACE_TIME5K)
                predicted_place5k_time5k = predict_results(df_user,PATH_MODEL_PLACE5K_TIME5K)
                predicted_time_time5k = predict_results(df_user,PATH_MODEL_TIME_TIME5K)
                st.session_state["predicted"]["time_sek"] = int(f"{predicted_time_time5k:.0f}")
                predicted_place_group_time5k = predict_results(df_user,PATH_MODEL_PGROUP_TIME5K)
                st.session_state["predicted"]["place"] = int(f"{predicted_place_time5k:.0f}")
                st.session_state["predicted"]["place_5k"] = int(f"{predicted_place5k_time5k:.0f}")
                st.session_state["predicted"]["place_group"] = int(f"{predicted_place_group_time5k:.0f}")
               
                predicted_placemin_time5k = predict_results(df_user_min, PATH_MODEL_PLACE_TIME5K)
                # predicted_place5kmin_time5k = predict_results(df_user_min,PATH_MODEL_PLACE5K_TIME5K)
                predicted_timemin_time5k = predict_results(df_user_min, PATH_MODEL_TIME_TIME5K)
                predicted_placemin_group_time5k = predict_results(df_user_min,PATH_MODEL_PGROUP_TIME5K)
                st.session_state["predicted"]["place_min"] = int(f"{predicted_placemin_time5k:.0f}")
                # st.session_state["predicted"]["place_5k_min"] = int(f"{predicted_place5kmin_time5k:.0f}")
                st.session_state["predicted"]["place_group_min"] = int(f"{predicted_placemin_group_time5k:.0f}")
                

                # Wyświetlenie wyników
                if predicted_place_time5k is not None:
                    predicted_tempo_time5k = predict_results(df_user,PATH_MODEL_TEMPO5K_TIME5K)
                    predicted_tempomin_time5k = predict_results(df_user_min,PATH_MODEL_TEMPO5K_TIME5K)
                    # st.session_state["predicted"]["user_tempo_5k"] = int(f"{predicted_tempo5k:.0f}")
                    df_user_tempo_5k = pd.DataFrame([{
                        'rok': year,
                        'płeć': gender_numeric,
                        'rocznik': yborn,
                        'tempo': predicted_tempo_time5k  
                    }])
                    predicted_place_tempo5k = predict_results(df_user_tempo_5k,PATH_MODEL_PLACE_TEMPO5K)
                    st.session_state["predicted"]["place_tempo5k"] = int(f"{predicted_place_tempo5k:.0f}")
                    # st.success(f"**Miejsce - Przewidywany wynik:** {predicted_place_5k:.0f} (na podstawie modelu ML){predicted_place_5k_min:.0f} Tempo: {seconds_to_time(predicted_tempo5k )}/km, {predicted_place:.0f}")



                    df_user_min2 = pd.DataFrame([{
                        'rok': year,
                        'płeć': gender_numeric,
                        'rocznik': yborn,
                        '5_km_tempo': predicted_tempomin_time5k  
                    }])
                    predicted_time5k_tempo5k = predict_results(df_user_min2,PATH_MODEL_TIME5K_TEMPO5K)
                    st.session_state["predicted"]["time_5k_min"] = seconds_to_time(predicted_time5k_tempo5k)

                    df_user_min3 = pd.DataFrame([{
                        'rok': year,
                        'płeć': gender_numeric,
                        'rocznik': yborn,
                        '5_km_czas': predicted_time5k_tempo5k  
                    }])
                    
                    
                    predicted_place5kmin_time5k = predict_results(df_user_min3,PATH_MODEL_PLACE5K_TIME5K)
                    st.session_state["predicted"]["place_5k_min"] = int(f"{predicted_place5kmin_time5k:.0f}")


                
                st.session_state["predicted"]["year"] = year
                st.session_state["predicted"]["name"] = st.session_state.name
                st.session_state["predicted"]["gender"] = gender
                st.session_state["predicted"]["yborn"] = st.session_state.yborn
                st.session_state["predicted"]["time_5k"] = st.session_state.time5km
                st.session_state["predicted"]["tempo_5k"] = seconds_to_time(predicted_tempo_time5k)
                st.session_state["predicted"]["tempo_5k_min"] = seconds_to_time(predicted_tempomin_time5k)
                st.session_state["predicted"]["time"] = seconds_to_time(predicted_time_time5k)
                st.session_state["predicted"]["time_min"] = seconds_to_time(predicted_timemin_time5k)
               
                motivation = ""
                motivation = generate_ai_motivation()
                st.session_state["predicted"]["motivation"] = motivation

                st.session_state.active_tab = "t3"
                st.rerun()

                # st.write(st.session_state["predicted"])

                # predicted = st.session_state["predicted"]
                # st.write(predicted["name"])
