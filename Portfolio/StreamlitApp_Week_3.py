import os, sys, warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import posixpath
import tempfile

import boto3
import sagemaker
from sagemaker.predictor import Predictor
from sagemaker.serializers import NumpySerializer
from sagemaker.deserializers import NumpyDeserializer

import shap

# =========================
# Setup & Path Configuration
# =========================
warnings.simplefilter("ignore")

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.feature_utils import extract_features

# =========================
# Secrets (DO NOT hardcode)
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
def get_session(_aws_id, _aws_secret, _aws_token):
    return boto3.Session(
        aws_access_key_id=_aws_id,
        aws_secret_access_key=_aws_secret,
        aws_session_token=_aws_token,
        region_name="us-east-1",
    )

session = get_session(aws_id, aws_secret, aws_token)
sm_session = sagemaker.Session(boto_session=session)

# =========================
# Data & Model Config
# =========================
df_features = extract_features()

MODEL_INFO = {
    "endpoint": aws_endpoint,
    "explainer": "explainer.shap",

    # What the SageMaker endpoint expects (15 features)
    "keys": [
        "JPM","MS","C","WFC","BAC","COF",
        "DEXJPUS","DEXUSUK","SP500","DJIA","VIXCLS",
        "GS_mom5","GS_vol20","GS_hl_range","GS_ma10_50_gap"
    ],

    # What you want users to type in (11 features)
    "ui_keys": ["JPM","MS","C","WFC","BAC","COF","DEXJPUS","DEXUSUK","SP500","DJIA","VIXCLS"],

    "inputs": [
        {"name": k, "type": "number", "min": -1.0, "max": 1.0, "default": 0.0, "step": 0.01}
        for k in ["JPM","MS","C","WFC","BAC","COF","DEXJPUS","DEXUSUK","SP500","DJIA","VIXCLS"]
    ]
}

# =========================
# S3 Diagnostics (Optional)
# =========================
def list_s3_keys(bucket: str, prefix: str):
    """Lists keys under a prefix. Useful for confirming explainer path."""
    s3 = session.client("s3")
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    return [o["Key"] for o in resp.get("Contents", [])]

# =========================
# SHAP Explainer Loader (with diagnostics)
# =========================
def load_shap_explainer(_session, bucket, key, local_path):
    s3_client = _session.client("s3")

    if not os.path.exists(local_path):
        try:
            s3_client.download_file(Filename=local_path, Bucket=bucket, Key=key)
        except Exception as e:
            # Streamlit may redact; show our own details
            if hasattr(e, "response"):
                code = e.response.get("Error", {}).get("Code", "Unknown")
                msg = e.response.get("Error", {}).get("Message", "No message")
                st.error(f"S3 download failed: {code} — {msg}")
            else:
                st.error(f"S3 download failed: {repr(e)}")

            st.error(f"Bucket: {bucket}")
            st.error(f"Key attempted: {key}")
            raise

    with open(local_path, "rb") as f:
        return shap.Explainer.load(f)

# =========================
# Prediction Logic
# =========================
def call_model_api(input_df: pd.DataFrame):
    predictor = Predictor(
        endpoint_name=MODEL_INFO["endpoint"],
        sagemaker_session=sm_session,
        serializer=NumpySerializer(),
        deserializer=NumpyDeserializer(),
    )

    try:
        # NumpySerializer expects numpy array
        x = input_df.to_numpy(dtype=np.float32)  # shape (1, 15)
        raw_pred = predictor.predict(x)
        pred_val = np.array(raw_pred).reshape(-1)[0]
        return round(float(pred_val), 4), 200
    except Exception as e:
        return f"Error: {str(e)}", 500

# =========================
# Local Explainability
# =========================
def display_explanation(input_df: pd.DataFrame, _session, _aws_bucket):
    explainer_name = MODEL_INFO["explainer"]

    # ✅ FIXED: this is where you actually uploaded it
    # s3://sagemaker-us-east-1-724944527346/sklearn-pipeline-deployment/explainer.shap
    explainer_s3_key = "sklearn-pipeline-deployment/explainer.shap"

    local_path = os.path.join(tempfile.gettempdir(), explainer_name)
    explainer = load_shap_explainer(_session, _aws_bucket, explainer_s3_key, local_path)

    # IMPORTANT: keep as DataFrame so feature names align
    shap_values = explainer(input_df)

    st.subheader("🔍 Decision Transparency (SHAP)")

    # SHAP draws on current matplotlib figure; show=False prevents double-render
    shap.plots.waterfall(shap_values[0], max_display=10, show=False)
    st.pyplot(plt.gcf(), clear_figure=True)

    # Top feature (based on absolute SHAP magnitude)
    vals = shap_values[0].values
    names = shap_values[0].feature_names
    top_idx = int(np.argmax(np.abs(vals)))
    top_feature = names[top_idx]
    st.info(f"**Business Insight:** The most influential factor in this decision was **{top_feature}**.")

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="ML Deployment", layout="wide")
st.title("👨‍💻 ML Deployment")

with st.expander("🔧 Debug (S3 Explorer)", expanded=False):
    st.caption("Lists keys under a prefix in your bucket to confirm file paths.")
    prefix = st.text_input("Prefix to list", value="sklearn-pipeline-deployment")
    if st.button("List S3 keys"):
        try:
            keys = list_s3_keys(aws_bucket, prefix)
            st.write(keys if keys else "No objects found under that prefix.")
        except Exception as e:
            st.error(f"Could not list objects: {e}")

with st.form("pred_form"):
    st.subheader("Inputs")
    cols = st.columns(2)
    user_inputs = {}

    for i, inp in enumerate(MODEL_INFO["inputs"]):
        with cols[i % 2]:
            user_inputs[inp["name"]] = st.number_input(
                inp["name"].replace("_", " ").upper(),
                min_value=inp["min"],
                max_value=inp["max"],
                value=inp["default"],
                step=inp["step"],
            )

    submitted = st.form_submit_button("Run Prediction")

# =========================
# Run Prediction
# =========================
if submitted:
    row = df_features.iloc[-1].copy()

    # Overwrite only user-provided 11 inputs
    for k in MODEL_INFO["ui_keys"]:
        row[k] = user_inputs[k]

    # Align to 15 expected keys; fill missing engineered cols with 0.0
    missing = [k for k in MODEL_INFO["keys"] if k not in row.index]
    if missing:
        st.warning(f"Missing engineered features in df_features (filled with 0.0): {missing}")

    row_aligned = row.reindex(MODEL_INFO["keys"], fill_value=0.0)
    input_df = pd.DataFrame([row_aligned], columns=MODEL_INFO["keys"])

    # Call endpoint
    res, status = call_model_api(input_df)

    if status == 200:
        st.metric("Prediction Result", res)

        # SHAP should NOT crash the app; show diagnostics if it fails
        try:
            display_explanation(input_df, session, aws_bucket)
        except Exception:
            st.warning(
                "Prediction succeeded, but SHAP explainer could not be loaded from S3. "
                "Open Debug (S3 Explorer) above to confirm the key/path."
            )
    else:
        st.error(res)
