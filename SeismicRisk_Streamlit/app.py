import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="AI Earthquake Risk Dashboard",
    layout="wide"
)

@st.cache_data
def load_data():
    df = pd.read_csv("data/earthquakes.csv")
    return df

df = load_data()

st.sidebar.title("🌍 Seismic Risk")
page = st.sidebar.radio(
    "Menu",
    [
        "Home",
        "Latest Earthquakes",
        "Earthquake Map",
        "Markov Forecast",
        "Omori Forecast",
        "Notifications Statistics"
    ]
)

# ---------------- HOME ----------------
if page == "Home":
    st.title("🌍 AI Earthquake Risk Monitoring System")

    st.write("""
    This project analyzes earthquake data, displays seismic activity,
    and provides simple AI-based risk indicators.
    """)

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Earthquakes", len(df))
    col2.metric("Max Magnitude", df["mag"].max())
    col3.metric("Average Depth", round(df["depth"].mean(), 2))

    st.subheader("Project Objective")
    st.write("""
    The objective is to help users understand earthquake activity
    using data visualization, risk indicators, and forecasting methods.
    """)

# ---------------- LATEST EARTHQUAKES ----------------
elif page == "Latest Earthquakes":
    st.title("📋 Latest Earthquakes")

    min_mag = st.slider("Minimum Magnitude", 0.0, 10.0, 4.0)

    filtered = df[df["mag"] >= min_mag]

    st.dataframe(filtered)

    st.metric("Earthquakes Found", len(filtered))

# ---------------- MAP ----------------
elif page == "Earthquake Map":
    st.title("🗺️ Earthquake Map")

    fig = px.scatter_mapbox(
        df,
        lat="lat",
        lon="lon",
        size="mag",
        color="mag",
        hover_name="place",
        hover_data=["date", "mag", "depth"],
        zoom=1,
        height=600
    )

    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

    st.plotly_chart(fig, use_container_width=True)

# ---------------- MARKOV ----------------
elif page == "Markov Forecast":
    st.title("🤖 Markov Forecast")

    st.write("""
    This page shows a simple Markov-style risk classification
    based on the latest maximum earthquake magnitude.
    """)

    max_mag = df["mag"].max()

    if max_mag >= 6:
        risk = "High"
    elif max_mag >= 4.5:
        risk = "Medium"
    else:
        risk = "Low"

    st.metric("Current Risk Level", risk)
    st.metric("Maximum Magnitude", max_mag)

    st.info("Later we will replace this with your real Markov model.")

# ---------------- OMORI ----------------
elif page == "Omori Forecast":
    st.title("📉 Omori Aftershock Forecast")

    st.write("""
    Omori law is used to estimate how aftershock activity decreases over time.
    This is a simplified demo for the final project.
    """)

    days = list(range(1, 11))
    aftershock_rate = [round(10 / d, 2) for d in days]

    omori_df = pd.DataFrame({
        "Day": days,
        "Estimated Aftershock Rate": aftershock_rate
    })

    st.line_chart(omori_df.set_index("Day"))

    st.dataframe(omori_df)

# ---------------- NOTIFICATIONS ----------------
elif page == "Notifications Statistics":
    st.title("🔔 Notifications Statistics")

    st.write("""
    This page will show alert statistics such as number of users,
    notifications sent, and high-risk events.
    """)

    total_alerts = len(df[df["mag"] >= 5])
    high_risk = len(df[df["mag"] >= 6])

    col1, col2 = st.columns(2)

    col1.metric("Notifications Sent", total_alerts)
    col2.metric("High Risk Alerts", high_risk)

    st.info("Later we can connect this page to your notificationpreferences table.")