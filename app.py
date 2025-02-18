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



st.set_page_config(page_title="Run4you",page_icon="sites/logo.png")
st.logo("sites/logo.png")

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
def check_files(folder_path):
    # Szukaj plików .pkl w folderze
    files = glob.glob(os.path.join(folder_path, "*.pkl"))
  
    # Zwróć wynik
    return bool(files), files
###########################################
# Funkcja dialogowa informująca o braku plików
@st.dialog("Brak plików")
def show_missing_files_dialog():
    st.write(f"Nie znaleziono plików .pkl w katalogu `{LOCAL_CURRENT_FOLDER}`.")
    st.write("Załaduj pliki z danymi maratonu")
    if st.button("OK"):
        st.session_state.active_tab = "t4"
        st.rerun()

###########################################
exists, csv_files =  check_files(LOCAL_CURRENT_FOLDER)

# Jeśli pliki nie istnieją, wyświetl dialog
if not exists and st.session_state.active_tab != "t4":
    show_missing_files_dialog()

# testdir = "/tmp/"
# testdir = "data/"
# os.makedirs(testdir, exist_ok=True)
# if os.path.exists(testdir):
#     st.write(f"✅ Katalog `{testdir}` istnieje.")
#     total, used, free = shutil.disk_usage(testdir)
#     st.write(f"📂 Dostępne miejsce w `{testdir}`:")
#     st.write(f"💾 Całkowita przestrzeń: {total / (1024**3):.2f} GB")
#     st.write(f"📊 Wykorzystane: {used / (1024**3):.2f} GB")
#     st.write(f"🟢 Wolne miejsce: {free / (1024**3):.2f} GB")
#     if os.access(testdir, os.W_OK):
#         st.write(f"✅ Można zapisywać w `{testdir}`")
#     else:
#         st.write(f"❌ `{testdir}` jest tylko do odczytu!")
# else:
#     st.write(f"❌ Katalog `{testdir}` NIE istnieje!")


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