from pathlib import Path
import pandas as pd
import streamlit as st

# Helper: always resolve files relative to project root
ROOT = Path(__file__).resolve().parents[1]
print(ROOT)
DATA = ROOT / "csv"

@st.cache_data(show_spinner=False)
def load_hotel_data() -> pd.DataFrame:
    return pd.read_csv(DATA / "hotel_bookings.csv")

@st.cache_data(show_spinner=False)
def load_student_data() -> pd.DataFrame:
    # Expect a CSV like: id, name, grade, gender, gpa, attendance_pct, ...
    return pd.read_csv(DATA / "student_data.csv")

@st.cache_data(show_spinner=False)
def load_insurance_data() -> pd.DataFrame:
    return pd.read_csv(DATA / "insurance.csv")