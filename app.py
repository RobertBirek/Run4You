import streamlit as st
import os
import glob
from pathlib import Path
from datetime import datetime
from sites import menu, t1, t2, t3, t4

import shutil


LOCAL_DATA_FOLDER = Path("data/")
LOCAL_RAW_FOLDER = LOCAL_DATA_FOLDER / "raw/"
LOCAL_CURRENT_FOLDER = LOCAL_DATA_FOLDER / "current/"
LOCAL_BACKUP_FOLDER = LOCAL_DATA_FOLDER / "backup/"

os.makedirs(LOCAL_DATA_FOLDER, exist_ok=True)
os.makedirs(LOCAL_RAW_FOLDER, exist_ok=True)
os.makedirs(LOCAL_CURRENT_FOLDER, exist_ok=True)
os.makedirs(LOCAL_BACKUP_FOLDER, exist_ok=True)


st.set_page_config(page_title="Run4you",page_icon="logo.png")
st.logo("logo.png")

###########################################
# Session state
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "t1"
if "name" not in st.session_state:
    st.session_state.name = "Nieznane"
if "gender" not in st.session_state:
    st.session_state.gender = 0
if "yborn" not in st.session_state:
    st.session_state.yborn = datetime.now().year - 30
if "time5km" not in st.session_state:
    st.session_state.time5km = "00:00:00"
if "button_disabled" not in st.session_state:
    st.session_state.button_disabled = True
if "stat" not in st.session_state:
    st.session_state["stat"] = {}
if "predicted" not in st.session_state:
    st.session_state["predicted"] = {}
###########################################
# Funkcja do ustawienia aktywnej zakładki
def set_active_tab(tab_name):
    st.session_state.active_tab = tab_name
###########################################
def check_csv_files(folder_path):
    # Szukaj plików .csv w folderze
    csv_files = glob.glob(os.path.join(LOCAL_RAW_FOLDER, "*.csv"))
  
    # Zwróć wynik
    return bool(csv_files), csv_files
###########################################
# Funkcja dialogowa informująca o braku plików
@st.dialog("Brak plików CSV")
def show_missing_files_dialog():
    st.write("Nie znaleziono plików .csv w katalogu `data/raw/`.")
    st.write("Załaduj pliki z danymi maratonu")
    if st.button("OK"):
        st.session_state.active_tab = "t4"
        st.rerun()

###########################################
exists, csv_files =  check_csv_files(LOCAL_RAW_FOLDER)

# Jeśli pliki nie istnieją, wyświetl dialog
if not exists and st.session_state.active_tab != "t4":
    show_missing_files_dialog()

if os.path.exists("/tmp/"):
    print("✅ Katalog `/tmp/` istnieje.")
    total, used, free = shutil.disk_usage("/tmp")
    st.write(f"📂 Dostępne miejsce w `/tmp/`:")
    st.write(f"💾 Całkowita przestrzeń: {total / (1024**3):.2f} GB")
    st.write(f"📊 Wykorzystane: {used / (1024**3):.2f} GB")
    st.write(f"🟢 Wolne miejsce: {free / (1024**3):.2f} GB")
    if os.access("/tmp/", os.W_OK):
        print("✅ Można zapisywać w `/tmp/`")
    else:
        print("❌ `/tmp/` jest tylko do odczytu!")
else:
    print("❌ Katalog `/tmp/` NIE istnieje!")


# Wywołanie menu
menu.show_menu()

if st.session_state.active_tab == "t1":
    t1.show_page()
if st.session_state.active_tab == "t2":
    t2.show_page()
if st.session_state.active_tab == "t3":
    t3.show_page()
if st.session_state.active_tab == "t4":
    t4.show_page()