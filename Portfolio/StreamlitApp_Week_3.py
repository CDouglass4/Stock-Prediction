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

# -------------------- Setup & Path Configuration --------------------
warnings.simplefilter("ignore")

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.feature_utils import extract_features

# -------------------- Secrets --------------------
aws_id = st.secrets["aws_credentials"]["AWS_ACCESS_KEY_ID"]
aws_secret = st.secrets["aws_credentials"]["AWS_SECRET_ACCESS_KEY"]
aws_token = st.secrets["aws_credentials"]["AWS_SESSION_TOKEN"]
aws_bucket = st.secrets["aws_credentials"]["AWS_BUCKET"]
aws_endpoint = st.secrets["aws_credentials"]["AWS_ENDPOINT"]

# -------------------- AWS Session --------------------
@st.cache_resource
def get_session(aws_id, aws_secret, aws_token):
    return boto3.Session(
        aws_access_key_id=aws_id,
        aws_secret_access_key=aws_secret,
        aws_session_token=aws_token,
        region_name="us-east-1",
    )

session = get_session(aws_id, aws_secret, aws_token)
sm_session = sagemaker.Session(boto_session=session)

# -------------------- Data & Model Configuration --------------------
df_features = extract_features()

MODEL_INFO = {
    "endpoint": aws_endpoint,
    "explainer": "explainer.shap",           # S3 key for SHAP explainer
    "pipeline": "finalized_model.tar.gz",
    "keys": [
        "JPM","MS","C","WFC","BAC","COF",
        "DEXJPUS","DEXUSUK","SP500","DJIA","VIXCLS",
        "GS_mom5","GS_vol20","GS_hl_range","GS_ma10_50_gap"
    ],
    "ui_keys": ["JPM","MS","C","WFC","BAC","COF","DEXJPUS","DEXUSUK","SP500","DJIA","VIXCLS"],
    "inputs": [
        {"name": k, "type": "number", "min": -1.0, "max": 1.0, "default": 0.0, "step": 0.01}
        for k in ["JPM","MS","C","WFC","BAC","COF","DEXJPUS","DEXUSUK","SP500","DJIA","VIXCLS"]
    ],
}

# -------------------- FIX 1: Correct SHAP loader (path, not file handle) --------------------
def load_shap_explainer(_session, bucket, key, local_path):
    s3_client = _session.client("s3")

    parent = os.path.dirname(local_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    if not os.path.exists(local_path):
        s3_client.download_file(Bucket=bucket, Key=key, Filename=local_path)

    # ✅ pass file path
    return shap.Explainer.load(local_path)

# -------------------- Build 1x15 Payload in Correct Order --------------------
def build_payload_row(df_features: pd.DataFrame, user_inputs: dict) -> pd.DataFrame:
    feature_cols = MODEL_INFO["keys"]

    # start with zeros
    row = {c: 0.0 for c in feature_cols}

    # seed engineered features from last row of df_features if available
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
    payload_df = payload_df.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return payload_df

# -------------------- SageMaker Prediction (send ONLY 1x15) --------------------
def call_model_api(payload_df: pd.DataFrame):
    predictor = Predictor(
        endpoint_name=MODEL_INFO["endpoint"],
        sagemaker_session=sm_session,
        serializer=NumpySerializer(),
        deserializer=NumpyDeserializer(),
    )
    try:
        X = payload_df.to_numpy(dtype=np.float32)   # shape (1, 15)
        raw_pred = predictor.predict(X)
        pred_val = float(np.array(raw_pred).reshape(-1)[0])
        return round(pred_val, 4), 200
    except Exception as e:
        return f"Error: {str(e)}", 500

# -------------------- FIX 2: SHAP explanation loads from S3 --------------------
def display_explanation(payload_df: pd.DataFrame, session, aws_bucket):
    st.subheader("🔍 Decision Transparency (SHAP)")

    local_path = os.path.join(tempfile.gettempdir(), MODEL_INFO["explainer"])
    explainer = load_shap_explainer(session, aws_bucket, MODEL_INFO["explainer"], local_path)

    shap_values = explainer(payload_df)

    fig = plt.figure(figsize=(10, 4))
    shap.plots.waterfall(shap_values[0], max_display=10, show=False)
    st.pyplot(fig)

    # Optional insight
    if hasattr(shap_values, "feature_names") and shap_values.feature_names:
        st.info(f"**Business Insight:** strongest driver: **{shap_values.feature_names[0]}**")

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="ML Deployment", layout="wide")
st.title("👨‍💻 ML Deployment")

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

if submitted:
    # ✅ FIX 3: Use correct 15-feature payload builder (keeps engineered features)
    payload_df = build_payload_row(df_features, user_inputs)

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
