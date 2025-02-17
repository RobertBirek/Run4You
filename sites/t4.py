import streamlit as st
import pandas as pd
import os
import glob
from dotenv import load_dotenv
import boto3
import time
from datetime import datetime
from pathlib import Path
import re
from pycaret.regression import (
    setup, compare_models, plot_model, finalize_model, 
    save_model, load_model, predict_model
)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
import numpy as np
import shutil


load_dotenv()


# Foldery lokalne
LOCAL_DATA_FOLDER = Path("data/")
LOCAL_RAW_FOLDER = LOCAL_DATA_FOLDER / "raw/"
LOCAL_CURRENT_FOLDER = LOCAL_DATA_FOLDER / "current/"
LOCAL_BACKUP_FOLDER = LOCAL_DATA_FOLDER / "backup/"


# Upewnienie się, że foldery istnieją
os.makedirs(LOCAL_DATA_FOLDER, exist_ok=True)
os.makedirs(LOCAL_RAW_FOLDER, exist_ok=True)
os.makedirs(LOCAL_CURRENT_FOLDER, exist_ok=True)
os.makedirs(LOCAL_BACKUP_FOLDER, exist_ok=True)

# Ścieżki do modeli
PATH_MODEL_PLACE_TIME5K = os.path.join(LOCAL_CURRENT_FOLDER, "model_place_time5k")
PATH_MODEL_PLACE5K_TIME5K = os.path.join(LOCAL_CURRENT_FOLDER, "model_place5k_time5k")
PATH_MODEL_TEMPO5K_TIME5K = os.path.join(LOCAL_CURRENT_FOLDER, "model_tempo5k_time5k")
PATH_MODEL_PLACE_TEMPO5K = os.path.join(LOCAL_CURRENT_FOLDER, "model_place_tempo5k")
PATH_MODEL_TIME_TIME5K = os.path.join(LOCAL_CURRENT_FOLDER, "model_time_time5k")
PATH_MODEL_PGROUP_TIME5K = os.path.join(LOCAL_CURRENT_FOLDER, "model_pgroup_time5k")

PATH_MODEL_TIME5K_TEMPO5K = os.path.join(LOCAL_CURRENT_FOLDER, "model_time5k_tempo5k")

ENDPOINT_URL = f"https://fra1.digitaloceanspaces.com"
BUCKET_NAME = "gotoit.robertbirek"
MARGED_RAW_FILE="halfmarathon_wroclaw_all__marge.csv"
MARGED_CLEAN_FILE="all_cleaned.csv"
CLEANED_FILE="data_cleaned.csv"

###################################
# Funkcja do zapisu pliku lokalnie
def save_file_locally(uploaded_file, save_path):
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
###################################
# Funkcja do usuwania pliku lokalnie
def delete_file_locally(file_path):
    try:
        os.remove(file_path)
        st.toast(f"Plik {file_path} został usunięty lokalnie.")
        time.sleep(3)
    except Exception as e:
        st.toast(f"Błąd podczas usuwania pliku lokalnego: {e}")
        time.sleep(5)
###################################
# Funkcja do pobierania plików lokalnych
def list_files_locally(folder):
    try:
        return os.listdir(folder)
    except Exception as e:
        st.toast(f"Błąd podczas pobierania plików lokalnych: {e}")
        time.sleep(5)
        return []
###################################
# Funkcja do pobierania plików z chmury
def list_files_in_cloud():
    try:
        s3_client = boto3.client("s3", endpoint_url=ENDPOINT_URL)
        response = s3_client.list_objects_v2(Bucket=BUCKET_NAME)
        if "Contents" in response:
            return [obj["Key"] for obj in response["Contents"]]
        else:
            return []
    except Exception as e:
        st.toast(f"Błąd podczas pobierania plików z chmury: {e}")
        time.sleep(5)
        return []
###################################
# Funkcja do wysyłania pliku do chmury
def upload_to_cloud(local_path, object_name):
    try:
        s3_client = boto3.client("s3", endpoint_url=ENDPOINT_URL)
        s3_client.upload_file(local_path,BUCKET_NAME, object_name)
        st.toast(f"Plik {local_path} został przesłany do chmury jako {object_name}.")
        time.sleep(3)
    except Exception as e:
        st.toast(f"Błąd podczas wysyłania pliku do chmury: {e}")
        time.sleep(5)
###################################
# Funkcja do pobierania pliku z chmury do lokalnego folderu
def download_from_cloud(object_name,local_path):
    try:
        s3_client = boto3.client("s3", endpoint_url=ENDPOINT_URL)
        s3_client.download_file(BUCKET_NAME, object_name, local_path)
        st.toast(f"Plik {object_name} został pobrany do lokalnego folderu: {local_path}.")
        time.sleep(3)
    except Exception as e:
        st.error(f"Błąd podczas pobierania pliku z chmury: {e}")
        time.sleep(5)
###################################
# Funkcja do usuwania pliku z chmury
def delete_file_from_cloud(object_name):
    try:
        s3_client = boto3.client("s3", endpoint_url=ENDPOINT_URL)
        s3_client.delete_object(Bucket=BUCKET_NAME, Key=object_name)
        st.toast(f"Plik {object_name} został usunięty z chmury.")
        time.sleep(3)
    except Exception as e:
        st.toast(f"Błąd podczas usuwania pliku z chmury: {e}")
        time.sleep(5)
###################################
def merge_csv_files(csv_files, output_path, chunksize=50000):
  
    if not csv_files:
        st.error("Brak plików CSV w folderze.")
        return False

    try:
        first_file = True  # Flaga dla pierwszego pliku
        for file in csv_files:
            chunk_iter = pd.read_csv(file, sep=";", chunksize=chunksize)
            for chunk in chunk_iter:
                # Dopisujemy pierwszą część z nagłówkami, resztę bez nagłówków
                chunk.to_csv(output_path, mode="w" if first_file else "a", index=False, sep=";", header=first_file)
                first_file = False  # Następne pliki dodajemy bez nagłówków

        st.success(f"Pliki zostały połączone i zapisane do {output_path}.")
        return True
    except Exception as e:
        st.error(f"Błąd podczas łączenia plików: {e}")
        return False
###########################################
def convert_time_to_seconds(time):
    try:
        h, m, s = map(int, time.split(':'))
        return h * 3600 + m * 60 + s
    except (ValueError, AttributeError):
        return None  
###########################################
def convert_tempo_to_seconds(tempo):
    # Obsługa przypadku, gdy tempo to NaN
    if pd.isnull(tempo):
        return None 
    try:
        # Rozdziel minuty i sekundy
        minutes = int(tempo)  # Całkowita część to minuty
        seconds = (tempo - minutes) * 100  # Część dziesiętna to sekundy
        # Zamień na sekundy
        total_seconds = int(minutes * 60 + seconds)
        return total_seconds
    except (ValueError, AttributeError):
        return None
###########################################
# Funkcja wyciągająca rok urodzenia z "Kategoria wiekowa"
def estimate_rocznik(category, current_year=2025):
    if isinstance(category, str):
        match = re.search(r'\d+', category)
        if match:
            avg_age = int(match.group()) + 5  # Środek przedziału (np. 30-39 -> 35)
            return current_year - avg_age
    return None
###########################################
# Funkcja do wyliczenia kategorii wiekowej na podstawie 'Rocznik' i 'Płeć'
def calculate_kategoria_wiekowa(plec, rocznik, current_year=2025):
    if pd.notna(rocznik) and pd.notna(plec):
        age = current_year - int(rocznik)
        category = (age // 10) * 10  # Zaokrąglenie w dół do dziesiątek
        return f"{plec}{category}"
    return None
###########################################
def clean_data(csv_files):
    if not csv_files:
        st.error("Brak plików CSV w folderze.")
        return False

    try:
        # Obecny rok
        current_year = datetime.now().year
        # Generowanie listy ostatnich 5 lat
        target_year = list(range(current_year - 4, current_year + 1))
        
        for file in csv_files:
            matching_years = [str(year) for year in target_year if str(year) in os.path.basename(file)]          
            # Sprawdzanie, czy w nazwie pliku jest któryś z lat z target_year
            if matching_years:
                st.write(f"Wczytywanie danych z roku {', '.join(matching_years)}...")
                # Wczytanie danych
                df_raw = pd.read_csv(file, sep=";")

                # liczba_wierszy, liczba_kolumn = df_raw.shape

                # st.session_state["stat"]["RAW_ROWS_COUNT"] = liczba_wierszy
                # st.session_state["stat"]["RAW_COLS_COUNT"] = liczba_kolumn

                # st.success(f"Wczytano dane: wierszy:{liczba_wierszy} kolumn:{liczba_kolumn} ")

                st.write(f"Czyszczenie danych w pliku {file} ...")
                df = df_raw.copy()
                # Zamieniam nazwy kolumn 
                df.columns = df.columns.str.replace(' ', '_').str.lower()
                # Filter out rows where 'Czas' equals 'DNS', 'DNF', or is None
                df = df[~df['czas'].isin(['DNS', 'DNF']) & df['czas'].notnull()]
                
                #usuwam kolumny
                df.drop(['imię', 'nazwisko', 'miasto', 'kraj','drużyna'], axis=1, inplace=True)
                # Zamieniam kolumny Czas = Czas w sek
                czas_columns = ['czas', '5_km_czas', '10_km_czas', '15_km_czas', '20_km_czas']
                for col in czas_columns:
                    df[col] = df[col].apply(convert_time_to_seconds)
                
                # Zamieniam kolumny tempo min/km na sek/km
                tempo_columns = ['5_km_tempo','10_km_tempo', '15_km_tempo', '20_km_tempo','tempo_stabilność','tempo']  # Replace with actual column names
                for col in tempo_columns:
                    df[col] = df[col].apply(convert_tempo_to_seconds)
                #sortuję rosnąco po miejsce
                df.sort_values(by='miejsce', ascending=True, inplace=True)
                #uzupełniam puste dane interpolując
                interp_columns = ['5_km_czas','5_km_miejsce_open','5_km_tempo', '10_km_czas','10_km_miejsce_open','10_km_tempo', '15_km_czas','15_km_miejsce_open','15_km_tempo', '20_km_czas','20_km_miejsce_open','20_km_tempo']
                for col in interp_columns:
                    df[col] = df[col].interpolate(method='linear')
                
                # Uzupełnianie brakujących wartości w kolumnie 'Rocznik'
                df['rocznik'] = df['rocznik'].fillna(
                    df['kategoria_wiekowa'].apply(estimate_rocznik)
                )
                
                # Uzupełnianie braków w kolumnie 'Kategoria wiekowa'
                df['kategoria_wiekowa'] = df['kategoria_wiekowa'].fillna(
                    df.apply(
                        lambda row: calculate_kategoria_wiekowa(row['płeć'], row['rocznik']),
                        axis=1
                    )
                )
                # Wypełnianie braków w kolumnie 'Kategoria wiekowa Miejsce'
                df['kategoria_wiekowa_miejsce'] = df.groupby('kategoria_wiekowa').cumcount() + 1

                # Uzupełnij puste wartości w kolumnie 'tempo_stabilność' odchyleniem standardowym z odcinków
                # Lista kolumn z tempem dla różnych odcinków biegu
                tempo_columns = ['5_km_tempo','10_km_tempo', '15_km_tempo', '20_km_tempo']
                # Obliczenie odchylenia standardowego tempa dla każdego uczestnika
                df['tempo_stabilność'] = df['tempo_stabilność'].fillna(df[tempo_columns].std(axis=1))
                #df['tempo_stabilność'] = df['tempo_stabilność'].fillna(df['tempo_stabilność'].median())
                
                # Zamieniam kolumny na liczby całkowite
                # Wybierz tylko kolumny liczbowe
                numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

                # Konwersja tylko kolumn liczbowych na int
                df[numeric_columns] = df[numeric_columns].astype(int)
                
                # int_columns = ['miejsce','płeć_miejsce', 'rocznik']
                # for col in int_columns:
                #     df[col] = df[col].astype(int)
                
                #zamieniam płeć K=0 M=1
                df['płeć'] = df['płeć'].map({'K': 0, 'M': 1})

                # Usuwanie gdy rocznik=0
                df = df[df["rocznik"] != 0]
                #dodaje kolumnę wiek, oblicza wiek w czasie biegu
                # Wyodrębnienie roku maratonu z nazwy pliku lub użycie aktualnego roku
                current_year = datetime.now().year
                
                match = re.search(r'\d{4}', file)
                if match:
                    marathon_year = int(match.group())
                else:
                    marathon_year = current_year

                # Obliczenie wieku uczestnika w czasie biegu
                df['wiek'] = marathon_year - df['rocznik']
                df['rok'] = marathon_year


                # Wartości odstających możemy się pozbyć używając np. tzw. IQR (Interquartile Range)
                Q1 = df["czas"].quantile(0.25)
                Q3 = df["czas"].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df_outliers = df[(df["czas"] >= lower_bound) & (df["czas"] <= upper_bound)]

                # Metoda odchylenia standardowego Obliczenie średniej i odchylenia standardowego
                # mean = df["czas"].mean()
                # std_dev = df["czas"].std()
                # stlower_bound = mean - 3 * std_dev
                # stupper_bound = mean + 3 * std_dev
                # df_outliers_st = df[(df["czas"] >= stlower_bound) & (df["czas"] <= stupper_bound)]

                fig, ax = plt.subplots(figsize=(10, 5))
                # Dane z wartościami odstającymi
                sns.scatterplot(data=df, x="czas", y="rocznik", ax=ax, color="red", linewidth=0, alpha=0.5, label="Z wartościami odstającymi")
                # Dane bez wartości odstających
                sns.scatterplot(data=df_outliers, x="czas", y="rocznik", ax=ax, color="blue", linewidth=0, alpha=0.5, label="Bez wartości odstających")
                # Dodanie tytułu i legendy
                ax.set_title("Porównanie danych z i bez wartości odstających")
                ax.legend()
                plt.tight_layout()
                st.pyplot(fig)

                fig, ax = plt.subplots(figsize=(10, 5))
                # Dane z wartościami odstającymi
                sns.scatterplot(data=df, x="czas", y="miejsce", ax=ax, color="red", linewidth=0, alpha=0.5, label="Z wartościami odstającymi")
                # Dane bez wartości odstających
                sns.scatterplot(data=df_outliers, x="czas", y="miejsce", ax=ax, color="blue", linewidth=0, alpha=0.5, label="Bez wartości odstających")
                # Dodanie tytułu i legendy
                ax.set_title("Porównanie danych z i bez wartości odstających")
                ax.legend()
                plt.tight_layout()
                st.pyplot(fig)



                #zapisuję plik
                output_file_name = os.path.basename(file).replace(".csv", "_cleaned.csv")
                output_file_path = os.path.join(LOCAL_CURRENT_FOLDER, output_file_name)
                df.to_csv(output_file_path, index=False, sep=";")
                # df_outliers.to_csv(output_file_path, index=False, sep=";")

            else:
                st.error(f"Plik {file} nie zawiera w nazwie żadnego z lat {target_year}. Pomijam ten plik.")
    except Exception as e:
        st.error(f"Błąd podczas czyszczenia danych: {e}")
###########################################
def create_model(df, target, model_path):
    category_f = ['płeć','rok', 'rocznik']
    exp = setup(
            data= df,
            target= target,
            categorical_features=category_f,
            session_id=132,
            verbose=False
        )
    best_models = exp.compare_models(n_select=3)
    best_model = best_models[0] if isinstance(best_models, list) else best_models
    exp.plot_model(best_model, plot='error',display_format='streamlit')
    exp.plot_model(best_model, plot='feature',display_format='streamlit')
    exp.plot_model(best_model, plot='learning',display_format='streamlit')

    exp.predict_model(best_model)

    new_model = finalize_model(best_model)

    exp.save_model(new_model, model_path)
    st.success(f"Model na podstawie {target} zapisany lokalnie w: {model_path}.pkl")
######################################
def backup_models():
    """Tworzy backup istniejących plików w folderze `current/`."""
    if not os.path.exists(LOCAL_BACKUP_FOLDER):
        os.makedirs(LOCAL_BACKUP_FOLDER)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    backup_subfolder = os.path.join(LOCAL_BACKUP_FOLDER, f"backup_{timestamp}")

    os.makedirs(backup_subfolder)
    for file in glob.glob(os.path.join(LOCAL_CURRENT_FOLDER, "*")):
        shutil.move(file, os.path.join(backup_subfolder, os.path.basename(file)))

    st.toast(f"Backup został zapisany w: `{backup_subfolder}`.")
    time.sleep(5)
#######################################
def clear_current_folder():
    """Usuwa wszystkie pliki z folderu `current/` przed zapisaniem nowych modeli."""
    for file in glob.glob(os.path.join(LOCAL_CURRENT_FOLDER, "*")):
        os.remove(file)
    st.toast("Wyczyściłem folder `current/` przed zapisaniem nowych modeli.")
    time.sleep(5)
#######################################
def create_models():
    file = os.path.join(LOCAL_CURRENT_FOLDER, MARGED_CLEAN_FILE)
    try:
        df = pd.read_csv(file, sep=";")

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.scatterplot(data=df, x="czas", y="miejsce", ax=ax, color="red", edgecolor="red", linewidth=0, alpha=0.5)
        ax.set_title("wyniki Półmaraton Wrocław")
        ax.set_xlabel("Czas")
        ax.set_ylabel("miejsce")
        plt.tight_layout()
        st.pyplot(fig)

        # df = df[df["rok"] == df["rok"].max()]
        # df = df.sample(n=1000, random_state=42)
        
        df_place_time5k = df[['rok','płeć', 'rocznik', '5_km_czas', 'miejsce']]
        df_place5k_time5k = df[['rok','płeć', 'rocznik', '5_km_czas', '5_km_miejsce_open']]
        df_tempo5k_time5k = df[['rok','płeć', 'rocznik', '5_km_czas', '5_km_tempo']]
        df_place_tempo5k = df[['rok','płeć', 'rocznik', 'tempo', 'miejsce']]
        df_time_time5k = df[['rok','płeć', 'rocznik', '5_km_czas', 'czas']]
        df_pgoup_time5k = df[['rok','płeć', 'rocznik', '5_km_czas', 'kategoria_wiekowa_miejsce']]

        df_time5k_tempo5k = df[['rok','płeć', 'rocznik', '5_km_tempo', '5_km_czas']]


        st.write("Wykres danych do modelu")
        # wykres danych
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.scatterplot(data=df, x="czas", y="rocznik", ax=ax, color="red", edgecolor="red", linewidth=0, alpha=0.5)
        ax.set_title("wyniki Półmaraton Wrocław")
        ax.set_xlabel("Czas")
        ax.set_ylabel("Rocznik")
        plt.tight_layout()
        st.pyplot(fig)

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.scatterplot(data=df, x="czas", y="miejsce", ax=ax, color="red", edgecolor="red", linewidth=0, alpha=0.5)
        ax.set_title("wyniki Półmaraton Wrocław")
        ax.set_xlabel("Czas")
        ax.set_ylabel("Miejsce")
        plt.tight_layout()
        st.pyplot(fig)

        ########################
        create_model(df_place_time5k, "miejsce", PATH_MODEL_PLACE_TIME5K)
        create_model(df_place5k_time5k, "5_km_miejsce_open", PATH_MODEL_PLACE5K_TIME5K)
        create_model(df_tempo5k_time5k, "5_km_tempo", PATH_MODEL_TEMPO5K_TIME5K)
        create_model(df_place_tempo5k, "miejsce", PATH_MODEL_PLACE_TEMPO5K)
        create_model(df_time_time5k, "czas", PATH_MODEL_TIME_TIME5K)
        create_model(df_pgoup_time5k, "kategoria_wiekowa_miejsce", PATH_MODEL_PGROUP_TIME5K)

        create_model(df_time5k_tempo5k, "5_km_czas", PATH_MODEL_TIME5K_TEMPO5K)

        ########################
        
    except Exception as e:
        st.error(f"Błąd przygotowywania modelu: {e}")

###########################################
@st.dialog("⚠ Model już istnieje!")
def confirm_overwrite():
    """Pokazuje dialog z pytaniem, czy nadpisać istniejący model."""
    st.write("Model istnieje. Czy na pewno chcesz stworzyć nowy?")
    
    c1,c2 = st.columns(2)
    with c1:
        if st.button("Tak, nadpisz model", key="overwrite_model", type="primary"):
            st.session_state.model_overwrite = True
            st.rerun()  # Przeładowanie aplikacji po wyborze
    with c2:
        if st.button("Nie, zachowaj obecny model", key="keep_model"):
            st.session_state.model_overwrite = False
            st.rerun()  # Przeładowanie aplikacji po wyborze

###########################################
def show_page():
    u1,u2,u3,u4 = st.tabs(["Dane lokalne","Dane do modelu","Dane w chmurze","Tworzenie modelu"])
    with u1:
        # Wyświetlanie plików lokalnych
        st.header("Pliki lokalne")
        local_files = list_files_locally(LOCAL_RAW_FOLDER)
        if local_files:
            for file in local_files:
                col1, col2, col3, col4 = st.columns([10,1,1,1])
                with col1:
                    st.text(file)
                with col2:
                    # Przygotowanie pliku do pobrania
                    file_path = os.path.join(LOCAL_RAW_FOLDER, file)
                    with open(file_path, "rb") as f:
                        file_data = f.read()
                    st.download_button(
                        label="🔽",
                        data=file_data,
                        file_name=file,
                        mime="application/octet-stream",
                        key=f"get_{file}",
                        help="Pobierz plik na swój komputer"
                    )
                with col3:
                    if st.button(f"☁️", key=f"upload_{file}", help="Wyślij plik do chmury"):
                        upload_to_cloud(os.path.join(LOCAL_RAW_FOLDER, file), f"raw/{file}")
                        st.rerun()
                with col4:
                    if st.button(f"🗑️", key=f"delete_local_{file}", type="primary", help="Usuń plik lokalnie"):
                        delete_file_locally(os.path.join(LOCAL_RAW_FOLDER, file))
                        st.rerun()
                # st.write("---")

        else:
            st.write("Brak plików lokalnych.")
        
        uploaded_file = st.file_uploader("Załaduj plik z danymi maratonu:")
        if uploaded_file is not None:
            # Zapis pliku lokalnie
            local_file_path = os.path.join(LOCAL_RAW_FOLDER, uploaded_file.name)
            save_file_locally(uploaded_file, local_file_path)
            st.success(f"Plik został zapisany lokalnie w: {local_file_path}")
            st.rerun()
            # # Prześlij plik do chmury
            # object_name = f"raw/{uploaded_file.name}"  # Ścieżka w bucketu
            # if upload_to_digitalcloud(local_file_path, object_name):
            #     st.success(f"Plik został przesłany do chmury jako: {object_name}")
    with u2:
        # Wyświetlanie plików lokalnych oczyszczonych
        st.header("Pliki lokalne używane do modelu")
        local_files = list_files_locally(LOCAL_CURRENT_FOLDER)
        if local_files:
            for file in local_files:
                col1, col2, col3, col4 = st.columns([10,1,1,1])
                with col1:
                    st.text(file)
                with col2:
                    # Przygotowanie pliku do pobrania
                    file_path = os.path.join(LOCAL_CURRENT_FOLDER, file)
                    with open(file_path, "rb") as f:
                        file_data = f.read()
                    st.download_button(
                        label="🔽",
                        data=file_data,
                        file_name=file,
                        mime="application/octet-stream",
                        key=f"get_{file}",
                        help="Pobierz plik na swój komputer"
                    )
                with col3:
                    if st.button(f"☁️", key=f"upload_{file}", help="Wyślij plik do chmury"):
                        upload_to_cloud(os.path.join(LOCAL_CURRENT_FOLDER, file), f"current/{file}")
                        st.rerun()
                with col4:
                    if st.button(f"🗑️", key=f"delete_local_{file}", type="primary", help="Usuń plik lokalnie"):
                        delete_file_locally(os.path.join(LOCAL_CURRENT_FOLDER, file))
                        st.rerun()
                # st.write("---")

        else:
            st.write("Brak plików lokalnych.")
    with u3:
        # Wyświetlanie plików w chmurze
        st.header("Pliki w chmurze")
        cloud_files = list_files_in_cloud()
        if cloud_files:
            for file in cloud_files:
                col1, col2, col3, col4 = st.columns([10,1, 1,1])
                with col1:
                    st.text(file)
                with col3:
                    if st.button(f"🔽", key=f"download_{file}", help="Pobierz plik do programu"):
                        download_from_cloud(file, os.path.join(LOCAL_RAW_FOLDER, os.path.basename(file)))
                        st.rerun()
                with col4:
                    if st.button("🗑️", key=f"delete_cloud_{file}", type="primary", help="Usuń plik z chmury"):
                        delete_file_from_cloud(file)
                        st.rerun()
        else:
            st.write("Brak plików w chmurze.")
    with u4:
        # Czyszczenie danych i tworzenie modelu
        st.header("Tworzenie modelu")
        
        if "model_overwrite" not in st.session_state:
            st.session_state.model_overwrite = False #= not model_exists  # Jeśli model nie istnieje, automatycznie = True
        
        # st.write(st.session_state.model_overwrite)        
        
        ok = st.button("Przygotuj model", type="primary", use_container_width=True)
        if ok or st.session_state.model_overwrite == True:
            model_exists = os.path.exists(f"{PATH_MODEL_PLACE_TIME5K}.pkl")
            if model_exists and st.session_state.model_overwrite == False :
                confirm_overwrite()
            elif not model_exists or st.session_state.model_overwrite == True:
                if model_exists:
                    pass
                    backup_models()  # Tworzenie backupu przed nadpisaniem
                    clear_current_folder()  # Czyszczenie folderu `current/`

                with st.status("Tworzenie modelu..."):
                    st.subheader("Czyszczenie i przygotowywanie danych...")
                    csv_clean_files = glob.glob(os.path.join(LOCAL_RAW_FOLDER, "*.csv"))
                    clean_data(csv_clean_files)
                    time.sleep(2)
                    st.subheader("Sprawdzanie plików i łączenie...")
                    # Pobierz listę plików .csv
                    csv_clean_files = glob.glob(os.path.join(LOCAL_CURRENT_FOLDER, "*.csv")) #pliki zawirające final
                    output_file = os.path.join(LOCAL_CURRENT_FOLDER, MARGED_CLEAN_FILE)
                    merge_csv_files(csv_clean_files,output_file)
                    time.sleep(2)
                    st.subheader("Tworzę model...")
                    with st.spinner("Tworzę model. Daj mi chwilkę"):
                        # Tworzenie modelu
                        create_models()
                    
                    st.session_state.model_overwrite = False