import os

os.environ["JAVA_HOME"] = r"C:\Program Files\Microsoft\jdk-11.0.16.101-hotspot"
os.environ["PATH"] = os.environ["JAVA_HOME"] + r"\bin;" + os.environ["PATH"]

import h2o
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Earthquake Risk Prediction API")

MODEL_PATH = r"C:\SeismicRisk_Streamlit\model\h2o_earthquake_model\GLM_1_AutoML_1_20260619_190823"

h2o.init()
model = h2o.load_model(MODEL_PATH)

class EarthquakeInput(BaseModel):
    magnitude: float
    depth_km: float
    latitude: float
    longitude: float
    tsunami: int = 0

@app.get("/")
def home():
    return {
        "message": "Earthquake Risk Prediction API is running"
    }

@app.post("/predict")
def predict(data: EarthquakeInput):
    df = pd.DataFrame([{
        "magnitude": data.magnitude,
        "depth_km": data.depth_km,
        "latitude": data.latitude,
        "longitude": data.longitude,
        "tsunami": data.tsunami
    }])

    hf = h2o.H2OFrame(df)
    prediction = model.predict(hf).as_data_frame()

    return {
        "risk_level": str(prediction.iloc[0]["predict"]),
        "input": data.dict()
    }