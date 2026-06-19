import os

os.environ["JAVA_HOME"] = r"C:\Program Files\Microsoft\jdk-11.0.16.101-hotspot"
os.environ["PATH"] = os.environ["JAVA_HOME"] + r"\bin;" + os.environ["PATH"]

import h2o
import mlflow
import pandas as pd
from h2o.automl import H2OAutoML

# Load real earthquake data
df = pd.read_csv("data/earthquakes.csv")

# Keep only useful columns
features = [
    "magnitude",
    "depth_km",
    "latitude",
    "longitude",
    "tsunami"
]

target = "risk_level"

df = df.dropna(subset=features + [target])

# Start H2O
h2o.init()

# Convert pandas dataframe to H2O frame
hf = h2o.H2OFrame(df)

hf[target] = hf[target].asfactor()

train, test = hf.split_frame(ratios=[0.8], seed=42)

mlflow.set_experiment("Earthquake Risk Prediction")

with mlflow.start_run():
    aml = H2OAutoML(
        max_models=5,
        seed=42,
        max_runtime_secs=120
    )

    aml.train(
        x=features,
        y=target,
        training_frame=train
    )

    leader = aml.leader
    perf = leader.model_performance(test)

    accuracy = perf.accuracy()[0][1]

    mlflow.log_param("model_type", "H2O AutoML")
    mlflow.log_param("features", ",".join(features))
    mlflow.log_metric("accuracy", accuracy)

    model_path = "model/h2o_earthquake_model"
    h2o.save_model(model=leader, path=model_path, force=True)

    mlflow.log_artifacts(model_path)

    print("Best model saved.")
    print("Accuracy:", accuracy)
    print("Leader model:", leader.model_id)

h2o.shutdown(prompt=False)