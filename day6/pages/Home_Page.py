# pages/Home_Page.py
import streamlit as st
from streamlit_autorefresh import st_autorefresh
from datetime import datetime
from zoneinfo import ZoneInfo

st_autorefresh(interval=1000, key="clock_refresh")
col1, col2 = st.columns(2)
with col1:
    st.title("Homepage")
with col2:
    tz_name = st.selectbox("Select timezone", ["UTC", "Europe/Berlin", "America/New_York"])
    tz = ZoneInfo(tz_name)   # e.g. "UTC", "America/New_York", "Asia/Singapore"
    now = datetime.now(tz).strftime("%H:%M:%S")

col1, col2, col3, col4 = st.columns(4)
with col1:
    c1, c2 = st.columns(2)
    with c1:
        st.write(f"📅 {datetime.now():%B %d, %Y}")
    with c2:
        st.write(f"🕒 {now}")


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