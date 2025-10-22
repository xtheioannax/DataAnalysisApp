import streamlit as st

st.set_page_config(page_title="Data Analysis App", layout="wide")

home_page   = st.Page("./pages/Home_Page.py", title="Home", icon="🏠")
hotel_page = st.Page("./pages/Hotel_Page.py", title="Hotel Dashboard", icon="🏨")
student_page = st.Page("./pages/Student_Page.py", title="Student Dashboard", icon="🎓")
insurance_page = st.Page("./pages/Insurance_Page.py", title="Insurance Dashboard", icon="🛡️")

page = st.navigation([home_page, hotel_page, student_page, insurance_page])
page.run()
