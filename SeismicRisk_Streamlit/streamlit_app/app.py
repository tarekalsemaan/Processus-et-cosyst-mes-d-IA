import streamlit as st
import pandas as pd
import plotly.express as px
import requests

st.set_page_config(
    page_title="AI Earthquake Risk Prediction Ecosystem",
    layout="wide"
)

API_URL = "http://127.0.0.1:8000/predict"

@st.cache_data
def load_data():
    return pd.read_csv("data/earthquakes.csv")

df = load_data()
df = df.dropna(subset=["magnitude", "latitude", "longitude", "depth_km"])
df["magnitude"] = pd.to_numeric(df["magnitude"], errors="coerce")
df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
df["depth_km"] = pd.to_numeric(df["depth_km"], errors="coerce")
df = df.dropna(subset=["magnitude", "latitude", "longitude", "depth_km"])

st.sidebar.title("🌍 Seismic AI")
page = st.sidebar.radio(
    "Menu",
    [
        "Home",
        "Latest Earthquakes",
        "Earthquake Map",
        "AI Risk Prediction",
        "Model Tracking",
        "Notifications Statistics"
    ]
)

if page == "Home":
    st.title("🌍 AI Earthquake Risk Prediction Ecosystem")
    st.write("This project uses H2O AutoML, MLflow, FastAPI, and Streamlit.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Earthquakes", len(df))
    c2.metric("Max Magnitude", df["magnitude"].max())
    c3.metric("Average Depth", round(df["depth_km"].mean(), 2))

elif page == "Latest Earthquakes":
    st.title("📋 Latest Earthquakes")

    min_mag = st.slider("Minimum Magnitude", 0.0, 10.0, 4.0)
    filtered = df[df["magnitude"] >= min_mag]

    st.dataframe(filtered)
    st.metric("Earthquakes Found", len(filtered))

elif page == "Earthquake Map":
    st.title("🗺️ Earthquake Map")

    fig = px.scatter_mapbox(
        df,
        lat="latitude",
        lon="longitude",
        size="magnitude",
        color="magnitude",
        hover_name="place",
        hover_data=["time_utc", "magnitude", "depth_km", "risk_level"],
        zoom=1,
        height=600
    )

    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig, use_container_width=True)

elif page == "AI Risk Prediction":
    st.title("🤖 AI Risk Prediction")

    st.write("This page sends earthquake values to the FastAPI backend, which uses the H2O AutoML model.")

    col1, col2 = st.columns(2)

    with col1:
        magnitude = st.number_input("Magnitude", min_value=0.0, max_value=10.0, value=5.5)
        depth_km = st.number_input("Depth KM", min_value=0.0, max_value=700.0, value=20.0)

    with col2:
        latitude = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=45.5)
        longitude = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=-73.6)

    tsunami = st.selectbox("Tsunami", [0, 1])

    if st.button("Predict Risk"):
        payload = {
            "magnitude": magnitude,
            "depth_km": depth_km,
            "latitude": latitude,
            "longitude": longitude,
            "tsunami": tsunami
        }

        try:
            response = requests.post(API_URL, json=payload)
            result = response.json()

            st.success(f"Predicted Risk Level: {result['risk_level']}")
            st.json(result)

        except Exception as e:
            st.error("Could not connect to FastAPI. Make sure uvicorn is running.")
            st.exception(e)

elif page == "Model Tracking":
    st.title("📊 Model Tracking with MLflow")

    st.write("""
    MLflow was used to track:
    - experiment name
    - model type
    - selected features
    - accuracy
    - saved model artifact
    """)

    st.metric("Experiment", "Earthquake Risk Prediction")
    st.metric("Model", "H2O AutoML - GLM")
    st.metric("Accuracy", "1.0")

    st.markdown("[Open MLflow Dashboard](http://127.0.0.1:5000)")

elif page == "Notifications Statistics":
    st.title("🔔 Notifications Statistics")

    high = len(df[df["risk_level"] == "High"])
    medium = len(df[df["risk_level"] == "Medium"])
    low = len(df[df["risk_level"] == "Low"])

    c1, c2, c3 = st.columns(3)

    c1.metric("High Risk Events", high)
    c2.metric("Medium Risk Events", medium)
    c3.metric("Low Risk Events", low)

    chart_df = pd.DataFrame({
        "Risk Level": ["High", "Medium", "Low"],
        "Count": [high, medium, low]
    })

    st.bar_chart(chart_df.set_index("Risk Level"))