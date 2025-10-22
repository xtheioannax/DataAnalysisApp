# pages/Home_Page.py
import streamlit as st
from streamlit_autorefresh import st_autorefresh
from datetime import datetime
from zoneinfo import ZoneInfo


st.title("Homepage")


col1, col2 = st.columns(2)

with col1:
    with st.container(border=True):
        st.page_link(
            "pages/Hotel_Page.py",
            label="🏨 **Hotel Dashboard**",
            help="Explore bookings, ADR, cancellations, and guest trends.",
        )
        st.write("Bookings, ADR, cancellations, and guest patterns.")

with col2:
    with st.container(border=True):
        st.page_link(
            "pages/Student_Page.py",
            label="🎓 **Student Dashboard**",
            help="View grades, GPA distributions, and attendance insights.",
        )
        st.write("Performance, attendance, and grade distributions.")

col1, col2 = st.columns(2)

with col1:
    with st.container(border=True):
        st.page_link(
            "pages/Insurance_Page.py",
            label="🛡️ **Insurance Dashboard**",
            help="Explore demographics and costs.",
        )
        st.write("Premiums, claims, and customer analytics.")

st.sidebar.success("Choose a page above")