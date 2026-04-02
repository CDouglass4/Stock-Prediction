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
from botocore.exceptions import ClientError

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
# Model Config
# =========================
MODEL_INFO = {
    "endpoint": aws_endpoint,

    # MUST match your upload exactly:
    # s3://<AWS_BUCKET>/sklearn-pipeline-deployment/explainer.shap
    "explainer_s3_key": "sklearn-pipeline-deployment/explainer.shap",
    "explainer_local_name": "explainer.shap",

    # 15 model features (order matters!)
    "keys": [
        "JPM","MS","C","WFC","BAC","COF",
        "DEXJPUS","DEXUSUK","SP500","DJIA","VIXCLS",
        "GS_mom5","GS_vol20","GS_hl_range","GS_ma10_50_gap"
    ],

    # 11 UI features
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
    ],
}

# =========================
# Data
# =========================
df_features = extract_features()

# =========================
# S3 Debug helpers
# =========================
def s3_exists(bucket: str, key: str) -> bool:
    s3 = session.client("s3")
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError:
        return False

def s3_list(bucket: str, prefix: str, max_keys: int = 50):
    s3 = session.client("s3")
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=max_keys)
    return [o["Key"] for o in resp.get("Contents", [])]

# =========================
# ✅ SHAP loader (UPDATED FIX)
# In your Streamlit env, shap.Explainer.load expects a file-like object.
# =========================
def load_shap_explainer(_session, bucket: str, key: str, local_path: str):
    s3 = _session.client("s3")
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    if not os.path.exists(local_path):
        s3.download_file(Bucket=bucket, Key=key, Filename=local_path)

    # ✅ FIX: pass file handle (rb), not a path string
    with open(local_path, "rb") as f:
        return shap.Explainer.load(f)

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

    X = input_df.to_numpy(dtype=np.float32)  # (1, 15)
    raw_pred = predictor.predict(X)
    pred_val = float(np.array(raw_pred).reshape(-1)[0])
    return round(pred_val, 4), 200

# =========================
# SHAP display
# =========================
def display_explanation(input_df: pd.DataFrame):
    s3_key = MODEL_INFO["explainer_s3_key"]
    local_path = os.path.join(tempfile.gettempdir(), MODEL_INFO["explainer_local_name"])

    explainer = load_shap_explainer(session, aws_bucket, s3_key, local_path)
    shap_values = explainer(input_df)

    st.subheader("🔍 Decision Transparency (SHAP)")
    shap.plots.waterfall(shap_values[0], max_display=10, show=False)
    st.pyplot(plt.gcf(), clear_figure=True)

    vals = shap_values[0].values
    names = shap_values[0].feature_names
    top_idx = int(np.argmax(np.abs(vals)))
    st.info(f"**Business Insight:** The most influential factor was **{names[top_idx]}**.")

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="ML Deployment", layout="wide")
st.title("👨‍💻 ML Deployment")

with st.expander("🔧 Debug SHAP in S3 (open if SHAP fails)", expanded=False):
    st.write("AWS_BUCKET secret:", aws_bucket)
    st.write("Explainer key:", MODEL_INFO["explainer_s3_key"])
    st.write("Explainer exists?:", s3_exists(aws_bucket, MODEL_INFO["explainer_s3_key"]))
    st.write("Keys under sklearn-pipeline-deployment/:")
    st.write(s3_list(aws_bucket, "sklearn-pipeline-deployment/"))

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
                step=inp["step"],
            )

    submitted = st.form_submit_button("Run Prediction")

# =========================
# Run Prediction
# =========================
if submitted:
    row = df_features.iloc[-1].copy()

    # overwrite 11 UI inputs
    for k in MODEL_INFO["ui_keys"]:
        row[k] = user_inputs[k]

    # warn on missing engineered features
    missing = [k for k in MODEL_INFO["keys"] if k not in row.index]
    if missing:
        st.warning(f"Missing engineered features in df_features (filled with 0.0): {missing}")

    # align to 15 features in correct order
    row_aligned = row.reindex(MODEL_INFO["keys"], fill_value=0.0)
    input_df = pd.DataFrame([row_aligned], columns=MODEL_INFO["keys"])

    # call endpoint
    res, status = call_model_api(input_df)

    if status == 200:
        st.metric("Prediction Result", res)

        # show SHAP or show exact error
        try:
            display_explanation(input_df)
        except Exception as e:
            st.error("Prediction succeeded, but SHAP failed with this exact error:")
            st.exception(e)
    else:
        st.error(res)
