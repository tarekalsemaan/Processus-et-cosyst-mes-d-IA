@echo off

start "FastAPI" cmd /k "cd /d C:\SeismicRisk_Streamlit && uvicorn api.main:app --reload"

timeout /t 5 >nul

start "MLflow" cmd /k "cd /d C:\SeismicRisk_Streamlit && mlflow ui"

timeout /t 5 >nul

start "Streamlit" cmd /k "cd /d C:\SeismicRisk_Streamlit && streamlit run streamlit_app\app.py"