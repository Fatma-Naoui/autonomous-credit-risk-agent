import streamlit as st
import pandas as pd
import json
import os
from PIL import Image

st.set_page_config(page_title="🏦 Autonomous Risk Agent Dashboard", layout="wide")
st.title("🏦 Autonomous Risk Agent Dashboard")

model_path = "../agent_core/models"

# Leaderboard Tab
st.header("📊 Model Leaderboard")
leaderboard_file = os.path.join(model_path, "model_leaderboard.csv")
if os.path.exists(leaderboard_file):
    df_leaderboard = pd.read_csv(leaderboard_file)
    st.dataframe(df_leaderboard)
else:
    st.warning("Leaderboard not found. Run the MCP pipeline first.")

# Metrics Tab
st.header("📈 Evaluation Metrics")
metrics_file = os.path.join(model_path, "model_metrics.json")
if os.path.exists(metrics_file):
    with open(metrics_file, "r") as f:
        metrics = json.load(f)
    st.json(metrics)
else:
    st.warning("Metrics file not found. Run the MCP pipeline first.")

# SHAP Visualization
st.header("🔍 SHAP Feature Importance")
shap_file = os.path.join(model_path, "shap_summary.png")
if os.path.exists(shap_file):
    image = Image.open(shap_file)
    st.image(image, caption="SHAP Feature Importance")
else:
    st.warning("SHAP plot not found. Run the MCP pipeline first.")

st.success(" MCP visualization ready. Update your pipeline, retrain, and refresh this dashboard to track progress.")
