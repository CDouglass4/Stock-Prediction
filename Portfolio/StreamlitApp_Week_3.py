import os, sys, warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import tempfile

import boto3
import sagemaker
from sagemaker.predictor import Predictor
from sagemaker.serializers import NumpySerializer
from sagemaker.deserializers import NumpyDeserializer

import shap

# =========================
# Setup
# =========================
warnings.simplefilter("ignore")

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.feature_utils import extract_features

# =========================
# Secrets
# =========================
aws_id = st.secrets["aws_credentials"]["AWS_ACCESS_KEY_ID"]
aws_secret = st.secrets["aws_credentials"]["AWS_SECRET_ACCESS_KEY"]
aws_token = st.secrets["aws_credentials"]["AWS_SESSION_TOKEN"]
aws_bucket = st.secrets["aws_credentials"]["AWS_BUCKET"]
aws_endpoint = st.secrets["aws_credentials"]["AWS_ENDPOINT"]

# =========================
# AWS Session
# =========================
@st.cache_resource
def get_session(_id, _secret, _token):
    return boto3.Session(
        aws_access_key_id=_id,
        aws_secret_access_key=_secret,
        aws_session_token=_token,
        region_name="us-east-1",
    )

session = get_session(aws_id, aws_secret, aws_token)
sm_session = sagemaker.Session(boto_session=session)

# =========================
# Data
# =========================
df_features = extract_features()

# =========================
# Model Config
# =========================
MODEL_INFO = {
    "endpoint": aws_endpoint,

    # Must match your upload:
    # s3://sagemaker-us-east-1-684398918081/sklearn-pipeline-deployment/explainer.shap
    "explainer_s3_key": "sklearn-pipeline-deployment/explainer.shap",
    "explainer_local_name": "explainer.shap",

    "keys": [
        "JPM","MS","C","WFC","BAC","COF",
        "DEXJPUS","DEXUSUK","SP500","DJIA","VIXCLS",
        "GS_mom5","GS_vol20","GS_hl_range","GS_ma10_50_gap"
    ],

    "ui_keys": [
        "JPM","MS","C","WFC","BAC","COF",
        "DEXJPUS","DEXUSUK","SP500","DJIA","VIXCLS"
    ],

    "inputs": [
        {"name": k, "min": -1.0, "max": 1.0, "default": 0.0, "step": 0.01}
        for k in [
            "JPM","MS","C","WFC","BAC","COF",
            "DEXJPUS","DEXUSUK","SP500","DJIA","VIXCLS"
        ]
    ]
}

# =========================
# S3 Utility
# =========================
def load_shap_explainer(_session, bucket, key, local_path):
    s3 = _session.client("s3")

    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    if not os.path.exists(local_path):
        try:
            s3.download_file(Bucket=bucket, Key=key, Filename=local_path)
        except Exception as e:
            if hasattr(e, "response"):
                code = e.response.get("Error", {}).get("Code", "Unknown")
                msg = e.response.get("Error", {}).get("Message", "No message")
                st.error(f"S3 download failed: {code} — {msg}")
            else:
                st.error(f"S3 download failed: {repr(e)}")

            st.error(f"Bucket: {bucket}")
            st.error(f"Key attempted: {key}")
            raise

    # ✅ Correct way to load
    return shap.Explainer.load(local_path)

# =========================
# SageMaker Prediction
# =========================
def call_model_api(input_df: pd.DataFrame):
    predictor = Predictor(
        endpoint_name=MODEL_INFO["endpoint"],
        sagemaker_session=sm_session,
        serializer=NumpySerializer(),
        deserializer=NumpyDeserializer(),
    )

    try:
        X = input_df.to_numpy(dtype=np.float32)
        raw_pred = predictor.predict(X)
        pred_val = np.array(raw_pred).reshape(-1)[0]
        return round(float(pred_val), 4), 200
    except Exception as e:
        return f"Error: {str(e)}", 500

# =========================
# SHAP Display
# =========================
def display_explanation(input_df, _session, _bucket):
    s3_key = MODEL_INFO["explainer_s3_key"]
    local_path = os.path.join(
        tempfile.gettempdir(),
        MODEL_INFO["explainer_local_name"]
    )

    explainer = load_shap_explainer(_session, _bucket, s3_key, local_path)
    shap_values = explainer(input_df)

    st.subheader("🔍 Decision Transparency (SHAP)")

    shap.plots.waterfall(shap_values[0], max_display=10, show=False)
    st.pyplot(plt.gcf(), clear_figure=True)

    vals = shap_values[0].values
    names = shap_values[0].feature_names
    top_idx = int(np.argmax(np.abs(vals)))

    st.info(
        f"**Business Insight:** "
        f"The most influential factor was **{names[top_idx]}**."
    )

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="ML Deployment", layout="wide")
st.title("👨‍💻 ML Deployment")

with st.form("pred_form"):
    st.subheader("Inputs")

    cols = st.columns(2)
    user_inputs = {}

    for i, inp in enumerate(MODEL_INFO["inputs"]):
        with cols[i % 2]:
            user_inputs[inp["name"]] = st.number_input(
                inp["name"],
                min_value=inp["min"],
                max_value=inp["max"],
                value=inp["default"],
                step=inp["step"]
            )

    submitted = st.form_submit_button("Run Prediction")

# =========================
# Run Prediction
# =========================
if submitted:
    row = df_features.iloc[-1].copy()

    # overwrite 11 UI features
    for k in MODEL_INFO["ui_keys"]:
        row[k] = user_inputs[k]

    missing = [k for k in MODEL_INFO["keys"] if k not in row.index]
    if missing:
        st.warning(
            f"Missing engineered features in df_features "
            f"(filled with 0.0): {missing}"
        )

    row_aligned = row.reindex(MODEL_INFO["keys"], fill_value=0.0)
    input_df = pd.DataFrame([row_aligned], columns=MODEL_INFO["keys"])

    result, status = call_model_api(input_df)

    if status == 200:
        st.metric("Prediction Result", result)

        try:
            display_explanation(input_df, session, aws_bucket)
        except Exception:
            st.warning(
                "Prediction succeeded, but SHAP explainer "
                "could not be loaded from S3."
            )
    else:
        st.error(result)
