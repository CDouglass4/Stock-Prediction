import os, sys, warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import posixpath

import joblib
import tarfile
import tempfile

import boto3
import sagemaker
from sagemaker.predictor import Predictor
from sagemaker.serializers import NumpySerializer
from sagemaker.deserializers import NumpyDeserializer

import shap

# Setup & Path Configuration
warnings.simplefilter("ignore")

# Fix path for Streamlit Cloud (ensure 'src' is findable)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.feature_utils import extract_features

# Access the secrets
aws_id = st.secrets["aws_credentials"]["AWS_ACCESS_KEY_ID"]
aws_secret = st.secrets["aws_credentials"]["AWS_SECRET_ACCESS_KEY"]
aws_token = st.secrets["aws_credentials"]["AWS_SESSION_TOKEN"]
aws_bucket = st.secrets["aws_credentials"]["AWS_BUCKET"]
aws_endpoint = st.secrets["aws_credentials"]["AWS_ENDPOINT"]

# AWS Session Management
@st.cache_resource
def get_session(aws_id, aws_secret, aws_token):
    return boto3.Session(
        aws_access_key_id=aws_id,
        aws_secret_access_key=aws_secret,
        aws_session_token=aws_token,
        region_name='us-east-1'
    )

session = get_session(aws_id, aws_secret, aws_token)
sm_session = sagemaker.Session(boto_session=session)

# Data & Model Configuration
df_features = extract_features()

MODEL_INFO = {
    "endpoint": aws_endpoint,
    "explainer": "explainer.shap",
    "pipeline": "finalized_model.tar.gz",

    # EXACT 15 features expected by SageMaker model (order matters!)
    "keys": [
        "JPM","MS","C","WFC","BAC","COF",
        "DEXJPUS","DEXUSUK","SP500","DJIA","VIXCLS",
        "GS_mom5","GS_vol20","GS_hl_range","GS_ma10_50_gap"
    ],

    # Only show sliders for the 11 “base” inputs in the UI
    "ui_keys": ["JPM","MS","C","WFC","BAC","COF","DEXJPUS","DEXUSUK","SP500","DJIA","VIXCLS"],

    "inputs": [
        {"name": k, "type": "number", "min": -1.0, "max": 1.0, "default": 0.0, "step": 0.01}
        for k in ["JPM","MS","C","WFC","BAC","COF","DEXJPUS","DEXUSUK","SP500","DJIA","VIXCLS"]
    ]
}

def load_shap_explainer(_session, bucket, key, local_path):
    s3_client = _session.client('s3')

    if not os.path.exists(local_path):
        s3_client.download_file(Filename=local_path, Bucket=bucket, Key=key)

    with open(local_path, "rb") as f:
        return shap.Explainer.load(f)

# ---------- CORE FIX: Build a 1x15 payload in the right order ----------
def build_payload_row(df_features: pd.DataFrame, user_inputs: dict) -> pd.DataFrame:
    """
    Returns a single-row DataFrame with exactly the 15 model features in MODEL_INFO['keys'] order.

    Strategy:
    - Start with 0.0 defaults for all 15 expected features
    - If df_features contains any of those feature columns, use the LAST row values as baseline
      (this preserves engineered features like GS_mom5, etc.)
    - Overwrite the 11 UI features with the user-provided values
    """
    feature_cols = MODEL_INFO["keys"]

    # start with zeros
    row = {c: 0.0 for c in feature_cols}

    # if df_features has matching columns, seed from its last row
    if isinstance(df_features, pd.DataFrame) and len(df_features) > 0:
        last = df_features.iloc[-1]
        for c in feature_cols:
            if c in df_features.columns:
                val = last[c]
                row[c] = 0.0 if pd.isna(val) or np.isinf(val) else float(val)

    # overwrite base inputs from UI
    for k, v in user_inputs.items():
        if k in row:
            row[k] = float(v)

    payload_df = pd.DataFrame([row], columns=feature_cols)

    # sanitize
    payload_df = payload_df.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    return payload_df

# Prediction Logic (send ONLY 1x15)
def call_model_api(payload_df: pd.DataFrame):
    predictor = Predictor(
        endpoint_name=MODEL_INFO["endpoint"],
        sagemaker_session=sm_session,
        serializer=NumpySerializer(),
        deserializer=NumpyDeserializer()
    )

    try:
        # IMPORTANT: send numpy array of shape (1, 15) in correct order
        X = payload_df.to_numpy(dtype=np.float32)

        raw_pred = predictor.predict(X)  # raw_pred could be np array
        pred_val = float(np.array(raw_pred).reshape(-1)[0])

        return round(pred_val, 4), 200
    except Exception as e:
        return f"Error: {str(e)}", 500

# Local Explainability (use the same 1-row payload)
def display_explanation(payload_df, session, aws_bucket):
    # S3 key for the SHAP explainer file (set this in Streamlit secrets)
    explainer_key = st.secrets["aws_credentials"]["AWS_EXPLAINER_KEY"]

    explainer = load_shap_explainer(
        session,
        aws_bucket,
        explainer_key,
        os.path.join(tempfile.gettempdir(), os.path.basename(explainer_key))
    )

    shap_values = explainer(payload_df)

    st.subheader("🔍 Decision Transparency (SHAP)")
    fig, ax = plt.subplots(figsize=(10, 4))
    shap.plots.waterfall(shap_values[0], max_display=10, show=False)
    st.pyplot(fig)

    top_feature = shap_values[0].feature_names[0]
    st.info(f"**Business Insight:** The most influential factor in this decision was **{top_feature}**.")

# Streamlit UI
st.set_page_config(page_title="ML Deployment", layout="wide")
st.title("👨‍💻 ML Deployment")

with st.form("pred_form"):
    st.subheader("Inputs")
    cols = st.columns(2)
    user_inputs = {}

    for i, inp in enumerate(MODEL_INFO["inputs"]):
        with cols[i % 2]:
            user_inputs[inp['name']] = st.number_input(
                inp['name'].replace('_', ' ').upper(),
                min_value=inp['min'],
                max_value=inp['max'],
                value=inp['default'],
                step=inp['step']
            )

    submitted = st.form_submit_button("Run Prediction")

if submitted:

    # Prepare data
    base_df = df_features

    # Build full 15-feature row
    full_feature_row = {k: user_inputs.get(k, 0.0) for k in MODEL_INFO["keys"]}

    payload_df = pd.DataFrame([full_feature_row], columns=MODEL_INFO["keys"])

    res, status = call_model_api(payload_df)

    if status == 200:
        st.metric("Prediction Result", res)

        try:
            display_explanation(payload_df, session, aws_bucket)
        except Exception as e:
            st.warning("Prediction worked, but SHAP explanation could not be loaded from S3.")
            st.write(str(e))

    else:
        st.error(res)


