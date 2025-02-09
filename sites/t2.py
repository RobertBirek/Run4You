import streamlit as st
import pandas as pd
from datetime import datetime
import re
import os
from dotenv import load_dotenv
from pycaret.regression import load_model, predict_model
# from openai import OpenAI 
from langfuse.decorators import observe
from langfuse.openai import OpenAI

load_dotenv()

# Połączenie z OpenAI
# openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # ✅ POPRAWNE

openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Inicjalizacja `session_state` dla `predicted`
if "predicted" not in st.session_state:
    st.session_state["predicted"] = {}

# Opcje płci
gender_options = ["Kobieta", "Mężczyzna"]

# Generowanie listy lat urodzenia
current_year = datetime.now().year
years = list(range(1923, current_year - 15))

# Folder na modele
LOCAL_CURRENT_FOLDER = "data/current/"

# Ścieżki do modeli
MODEL_PATH_5KPLACE = os.path.join(LOCAL_CURRENT_FOLDER, "model_5kplace")
MODEL_PATH_TEMPO = os.path.join(LOCAL_CURRENT_FOLDER, "model_tempo")
MODEL_PATH_PLACE = os.path.join(LOCAL_CURRENT_FOLDER, "model_place")

####################################
def update_name():
    st.session_state.name = st.session_state.input_name
    st.session_state["predicted"]["name"] = st.session_state.name
####################################
def update_gender():
    st.session_state.gender = gender_options.index(st.session_state.selected_gender)
    st.session_state["predicted"]["gender"] = st.session_state.gender
####################################
def update_yborn():
    st.session_state.yborn = st.session_state.selected_yborn
    st.session_state["predicted"]["yborn"] = st.session_state.yborn
####################################
def update_time():
    st.session_state.time5km = st.session_state.input_time5km
    st.session_state["predicted"]["time5km"] = st.session_state.time5km
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
            predicted_name = st.session_state["predicted"].get("name", None)
            predicted_gender = st.session_state["predicted"].get("gender", None)
            predicted_yborn = st.session_state["predicted"].get("yborn", None)
            predicted_time5k = st.session_state["predicted"].get("time5k", None)
            predicted_place_5k = st.session_state["predicted"].get("user_place_5k", None)
            predicted_place_5k_min = st.session_state["predicted"].get("user_place_5k_min", None)
            # predicted_tempo5k = seconds_to_time(st.session_state["predicted"].get("user_tempo_5k", 0))
            predicted_tempo5k = st.session_state["predicted"].get("tempo5k", None)
            predicted_place = st.session_state["predicted"].get("place", None)


        persona = """
        Jesteś trenerem personalnym, który od 20 lat trenuje przyszłych marytończyków i maratonczyki,
        pomagając im osiągnąć ich cele.   
        """.strip()
        question = f"""
        Jednym z Twoich podopiecznych jest {predicted_name}, jest {predicted_gender}, jest z rocznika {predicted_yborn} który biega 5 km w czasie {predicted_time5k}.
        Naiszybszy biegacz w jego kategorii biegnie 5 km w czasie {predicted_place_5k_min}.
        Według Twoich obliczeń, {predicted_name} powinien ukończyć bieg na miejscu {predicted_place_5k}, osiągając tempo {predicted_tempo5k}.
        Jeżeli {predicted_name} utrzyma tempo {predicted_tempo5k} przez cały bieg, to powinien ukończyć bieg na miejscu {predicted_place}.
        Jakie rady mógłbyś dać {predicted_name} przed biegiem?
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
        # Sprawdzenie, czy model istnieje
        if not os.path.exists(MODEL_PATH_5KPLACE + ".pkl"):
            st.error("Model nie istnieje! Najpierw wygeneruj model.")
            return None
        else:
            with st.spinner("Generowanie wyniku..."):
                # Konwersja płci (Kobieta = 0, Mężczyzna = 1)
                gender_numeric = 0 if gender == "Kobieta" else 1
                # Konwersja czasu na sekundy
                time5_seconds = time_to_seconds(time5) # Czas w hh:mm:ss
                
                df_user = pd.DataFrame([{
                    'płeć': gender_numeric,
                    'rocznik': yborn,
                    '5_km_czas': time5_seconds  
                }])
                
                df_user_min = pd.DataFrame([{
                    'płeć': gender_numeric,
                    'rocznik': yborn,
                    '5_km_czas': 600  
                }])
                # Wykonanie predykcji
                predicted_place_5k = predict_results(df_user,MODEL_PATH_5KPLACE)
                st.session_state["predicted"]["user_place_5k"] = int(f"{predicted_place_5k:.0f}")
                predicted_place_5k_min = predict_results(df_user_min, MODEL_PATH_5KPLACE)
                st.session_state["predicted"]["user_place_5k_min"] = int(f"{predicted_place_5k_min:.0f}")

                # Wyświetlenie wyników
                if predicted_place_5k is not None:
                    predicted_tempo5k = predict_results(df_user,MODEL_PATH_TEMPO)
                    st.session_state["predicted"]["user_tempo_5k"] = int(f"{predicted_tempo5k:.0f}")
                    df_user_tempo_5k = pd.DataFrame([{
                        'płeć': gender_numeric,
                        'rocznik': yborn,
                        'tempo': predicted_tempo5k  
                    }])
                    predicted_place = predict_results(df_user_tempo_5k,MODEL_PATH_PLACE)
                    st.session_state["predicted"]["place"] = int(f"{predicted_place:.0f}")
                    # place = df_predicted_place = predict_results(df_user,MODEL_PATH_PLACE)
                    st.success(f"🎯 **Miejsce - Przewidywany wynik:** {predicted_place_5k:.0f} (na podstawie modelu ML){predicted_place_5k_min:.0f} Tempo: {seconds_to_time(predicted_tempo5k )}/km, {predicted_place:.0f}")

                st.session_state["predicted"]["name"] = st.session_state.name
                st.session_state["predicted"]["gender"] = st.session_state.gender
                st.session_state["predicted"]["yborn"] = st.session_state.yborn
                st.session_state["predicted"]["time5km"] = st.session_state.time5km
                st.session_state["predicted"]["tempo5k"] = seconds_to_time(int(predicted_tempo5k))
                motivation = generate_ai_motivation()
                st.session_state["predicted"]["motivation"] = motivation

                st.session_state.active_tab = "t3"
                st.rerun()
