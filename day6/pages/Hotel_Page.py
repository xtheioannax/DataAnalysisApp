import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from services.data import load_hotel_data

st.set_page_config(page_title="Hotel Bookings Dashboard", layout="wide")
st.page_link("pages/Home_Page.py", label="↩︎ Back to Home")


# --- Load data ---
hotel_data = load_hotel_data()

# --- Sidebar filters ---
st.sidebar.header("Filters")

years = sorted(hotel_data['arrival_date_year'].dropna().unique())
months = ["January","February","March","April","May","June",
          "July","August","September","October","November","December"]
hotels = sorted(hotel_data['hotel'].dropna().unique())

sel_years = st.sidebar.multiselect("Select year(s)", years, default=years)
sel_month = st.sidebar.multiselect("Select months", months, default=months)
sel_hotel = st.sidebar.multiselect("Select hotel type", hotels, default=hotels)

adr_min, adr_max = st.sidebar.slider(
    "ADR range (€)", 
    float(hotel_data['adr'].min()), 
    float(hotel_data['adr'].max()), 
    (float(hotel_data['adr'].min()), float(hotel_data['adr'].max()))
)

lead_min, lead_max = st.sidebar.slider(
    "Lead time (days)", 
    int(hotel_data['lead_time'].min()), 
    int(hotel_data['lead_time'].max()), 
    (int(hotel_data['lead_time'].min()), int(hotel_data['lead_time'].max()))
)

# --- Apply filters safely ---
filtered = hotel_data.copy()

if sel_years:
    filtered = filtered[filtered['arrival_date_year'].isin(sel_years)]

if sel_month:
    filtered = filtered[filtered['arrival_date_month'].isin(sel_month)]

if sel_hotel:
    filtered = filtered[filtered['hotel'].isin(sel_hotel)]

filtered = filtered[
    (filtered['adr'].between(adr_min, adr_max)) &
    (filtered['lead_time'].between(lead_min, lead_max))
]

st.title("Hotel Bookings Dashboard")
st.caption(f"Filtered rows: {len(filtered)}")

if filtered.empty:
    st.warning("No data matches your current filters. Please adjust filters in the sidebar.")
    st.stop()

### =============== ###
###     Metrics     ###
### =============== ###


total_bookings = len(filtered)
cancel_rate = filtered['is_canceled'].mean() * 100 if len(filtered) > 0 else 0
avg_adr = filtered['adr'].mean() if len(filtered) > 0 else 0
avg_lead = filtered['lead_time'].mean() if len(filtered) > 0 else 0
repeat_guests = filtered['is_repeated_guest'].mean() * 100 if len(filtered) > 0 else 0

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Bookings", f"{total_bookings:,}")
col2.metric("Cancellation Rate", f"{cancel_rate:.1f}%")
col3.metric("Avg. ADR", f"€{avg_adr:.2f}")
col4.metric("Avg. Lead Time", f"{avg_lead:.1f} days")
col5.metric("Repeated Guests", f"{repeat_guests:.1f}%")


### =============== ###
###    Dashboard    ###
### =============== ###

month_order = ["January","February","March","April","May","June",
               "July","August","September","October","November","December"]

# Row 1 
col1, col2 = st.columns(2)

with col1:
    st.subheader("Hotel Popularity")
    fig, ax = plt.subplots()
    filtered['hotel'].value_counts().plot(kind='bar', ax=ax, color=['skyblue', 'salmon'])
    ax.set_xlabel("Hotel")
    ax.set_ylabel("Bookings")
    ax.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig)

with col2:
    st.subheader("Cancellation Rate by Hotel (%)")
    fig, ax = plt.subplots()
    (filtered.groupby('hotel')['is_canceled'].mean() * 100).plot(kind='bar', ax=ax, color='orange')
    ax.set_xlabel("Hotel")
    ax.set_ylabel("Canceled (%)")
    ax.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig)

# Row 2
col1, col2 = st.columns(2)

with col1:
    st.subheader("Monthly Booking Trend")
    fig, ax = plt.subplots()

    month_map = {m: i+1 for i, m in enumerate(month_order)}

    for h in filtered['hotel'].unique():
        grouped = (
            filtered[filtered['hotel'] == h]
            .groupby(['arrival_date_year', 'arrival_date_month'])
            .size()
            .reset_index(name='bookings')
        )
        grouped['month_num'] = grouped['arrival_date_month'].map(month_map)
        grouped = grouped.sort_values(['arrival_date_year', 'month_num'])
        

        x_labels = grouped['arrival_date_year'].astype(str) + " " + grouped['arrival_date_month']
        
        ax.plot(x_labels, grouped['bookings'], marker='o', label=h)

    ax.legend()
    ax.set_xlabel("Year / Month")
    ax.set_ylabel("Bookings")
    ax.set_title("Monthly Booking Trend (across selected years)")
    plt.xticks(rotation=90)
    ax.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig)

with col2:
    st.subheader("Lead Time Distribution")
    fig, ax = plt.subplots()
    ax.hist(filtered['lead_time'], bins=30, color='purple', alpha=0.7)
    ax.set_xlabel("Days before arrival")
    ax.set_ylabel("Bookings")
    ax.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig)

# Row 3
col1, col2 = st.columns(2)

with col1:
    st.subheader("ADR Distribution (€)")
    fig, ax = plt.subplots()
    ax.hist(filtered['adr'], bins=40, color='green', alpha=0.7)
    ax.set_xlabel("Average Daily Rate (€)")
    ax.set_ylabel("Frequency")
    ax.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig)

with col2:
    st.subheader("Market Segment Distribution")
    fig, ax = plt.subplots()
    filtered['market_segment'].value_counts().plot(kind='bar', ax=ax, color='teal')
    ax.set_xlabel("Market Segment")
    ax.set_ylabel("Bookings")
    ax.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig)

# Row 4
col1, col2 = st.columns(2)

with col1:
    st.subheader("Top 10 Countries of Origin")
    fig, ax = plt.subplots()
    filtered['country'].value_counts().head(10).plot(kind='bar', ax=ax, color='darkcyan')
    ax.set_xlabel("Country")
    ax.set_ylabel("Bookings")
    ax.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig)

with col2:
    st.subheader("Room Type Changes (Reserved vs Assigned)")
    fig, ax = plt.subplots()
    room_changes = (filtered[filtered['reserved_room_type'] != filtered['assigned_room_type']]
                    .groupby('hotel').size())
    if not room_changes.empty:
        room_changes.plot(kind='bar', ax=ax, color='red')
    ax.set_xlabel("Hotel")
    ax.set_ylabel("Changes")
    ax.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig)

# Row 5
col1, col2 = st.columns(2)

with col1:
    st.subheader("Repeated Guests")
    fig, ax = plt.subplots()
    data = filtered['is_repeated_guest'].replace({0:"Not Repeated", 1:"Repeated"}).value_counts()
    if not data.empty:
        data.plot(kind='pie', autopct='%1.1f%%', ax=ax, startangle=90, colors=['lightcoral','lightgreen'])
    ax.set_ylabel("")
    st.pyplot(fig)

with col2:
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots()
    corr = filtered[['lead_time', 'adr', 'is_canceled']].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=ax)
    ax.set_title("Correlation: ADR, Lead Time, Cancellations")
    st.pyplot(fig)

# Row 6: Weekend vs Weekday Stays
col1, _ = st.columns([1,1])
with col1:
    st.subheader("Weekend vs Weekday Nights")
    fig, ax = plt.subplots()
    weekend_total = filtered['stays_in_weekend_nights'].sum()
    weekday_total = filtered['stays_in_week_nights'].sum()
    stay_data = pd.Series({
        'Weekend Nights': weekend_total,
        'Weekday Nights': weekday_total
    })
    stay_data.plot(kind='bar', ax=ax, color=['cornflowerblue', 'lightgray'])
    ax.set_ylabel("Total Nights")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_title("Distribution of Weekend vs Weekday Stays")
    st.pyplot(fig)
