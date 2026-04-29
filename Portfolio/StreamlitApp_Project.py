import os, sys, warnings, tarfile, tempfile
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import posixpath
import joblib
import boto3
import sagemaker
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
import shap

warnings.simplefilter("ignore")

# ── Custom classes (must match training definitions exactly) ───────────────────
class DataCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, freq_cols=None, drop_cols=None):
        self.freq_cols   = freq_cols or ["card4","card6","ProductCD",
                                          "P_emaildomain","R_emaildomain",
                                          "DeviceType","DeviceInfo"]
        self.drop_cols   = drop_cols or ["TransactionID"]
        self.freq_maps_  = {}
        self.label_maps_ = {}
        self.median_vals_= {}
        self.cat_cols_   = []
        self.num_cols_   = []

    @staticmethod
    def _recode_tf(df):
        for c in [col for col in df.columns if col.startswith("M")]:
            df[c] = df[c].map({"T": 1, "F": 0})
        return df

    def fit(self, X, y=None):
        X = X.copy()
        X.drop(columns=[c for c in self.drop_cols if c in X.columns], inplace=True)
        X = self._recode_tf(X)
        for c in self.freq_cols:
            if c in X.columns:
                self.freq_maps_[c] = X[c].value_counts(normalize=True).to_dict()
        for c, m in self.freq_maps_.items():
            X[c] = X[c].map(m)
        self.num_cols_ = X.select_dtypes(include="number").columns.tolist()
        self.cat_cols_ = X.select_dtypes(exclude="number").columns.tolist()
        for c in self.num_cols_:
            self.median_vals_[c] = X[c].median()
        for c in self.cat_cols_:
            X[c] = X[c].fillna("missing")
            le = LabelEncoder()
            le.fit(X[c].astype(str))
            self.label_maps_[c] = le
        return self

    def transform(self, X):
        X = X.copy()
        X.drop(columns=[c for c in self.drop_cols if c in X.columns], inplace=True)
        X = self._recode_tf(X)
        for c, m in self.freq_maps_.items():
            if c in X.columns:
                X[c] = X[c].map(m).fillna(0.0)
        for c in self.num_cols_:
            if c in X.columns:
                X[c] = X[c].fillna(self.median_vals_.get(c, 0))
        for c in self.cat_cols_:
            if c in X.columns:
                X[c] = X[c].fillna("missing").astype(str)
                le = self.label_maps_[c]
                known = set(le.classes_)
                X[c] = X[c].apply(lambda v: v if v in known else "missing")
                X[c] = le.transform(X[c])
        return X


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, missing_thresh=0.50, const_thresh=0.95):
        self.missing_thresh = missing_thresh
        self.const_thresh   = const_thresh
        self.drop_missing_  = []
        self.drop_const_    = []
        self.card1_freq_    = {}

    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy()
        null_ratio = X.isnull().mean()
        self.drop_missing_ = null_ratio[null_ratio > self.missing_thresh].index.tolist()
        self.drop_const_ = []
        for c in X.columns:
            if c in self.drop_missing_:
                continue
            if X[c].value_counts(normalize=True, dropna=False).iloc[0] > self.const_thresh:
                self.drop_const_.append(c)
        if "card1" in X.columns:
            self.card1_freq_ = X["card1"].value_counts(normalize=True).to_dict()
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        drop = list(set(self.drop_missing_ + self.drop_const_))
        X.drop(columns=[c for c in drop if c in X.columns], inplace=True)
        if "TransactionAmt" in X.columns:
            X["TransactionAmt_log"] = np.log1p(X["TransactionAmt"])
        if "TransactionDT" in X.columns:
            X["hour"]        = (X["TransactionDT"] // 3600) % 24
            X["day_of_week"] = (X["TransactionDT"] // 86400) % 7
        if "TransactionAmt" in X.columns and "card1" in X.columns:
            X["amt_x_card1"] = X["TransactionAmt"] * X["card1"]
        if "TransactionAmt" in X.columns and "C1" in X.columns:
            X["amt_per_C1"] = X["TransactionAmt"] / (X["C1"] + 1e-6)
        if "card1" in X.columns:
            X["card1_freq_enc"] = X["card1"].map(self.card1_freq_).fillna(0)
        return X


class DropCollinear(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.95):
        self.threshold  = threshold
        self.drop_cols_ = []

    def fit(self, X, y=None):
        X_df  = pd.DataFrame(X)
        num   = X_df.select_dtypes(include="number")
        corr  = num.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        self.drop_cols_ = [c for c in upper.columns if any(upper[c] > self.threshold)]
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X)
        return X_df.drop(columns=[c for c in self.drop_cols_ if c in X_df.columns])


# ── Secrets & session ──────────────────────────────────────────────────────────
aws_id       = st.secrets["aws_credentials"]["AWS_ACCESS_KEY_ID"]
aws_secret   = st.secrets["aws_credentials"]["AWS_SECRET_ACCESS_KEY"]
aws_token    = st.secrets["aws_credentials"]["AWS_SESSION_TOKEN"]
aws_bucket   = st.secrets["aws_credentials"]["AWS_BUCKET"]
aws_endpoint = st.secrets["aws_credentials"]["AWS_ENDPOINT"]

@st.cache_resource
def get_session(aws_id, aws_secret, aws_token):
    return boto3.Session(
        aws_access_key_id=aws_id,
        aws_secret_access_key=aws_secret,
        aws_session_token=aws_token,
        region_name='us-east-1'
    )

session    = get_session(aws_id, aws_secret, aws_token)
sm_session = sagemaker.Session(boto_session=session)

# ── Load X_train from S3 for default input values ─────────────────────────────
@st.cache_data
def load_dataset(_session, bucket):
    s3  = _session.client('s3')
    tmp = os.path.join(tempfile.gettempdir(), 'X_train.csv')
    if not os.path.exists(tmp):
        s3.download_file(Bucket=bucket, Key='Portfolio/X_train.csv', Filename=tmp)
    df = pd.read_csv(tmp)
    return df.loc[:, ~df.columns.str.contains('^Unnamed')]

dataset = load_dataset(session, aws_bucket)

# ── Model config ───────────────────────────────────────────────────────────────
INPUT_KEYS = ['TransactionAmt', 'card1', 'card3', 'C1', 'C12', 'TransactionDT']

MODEL_INFO = {
    "endpoint" : aws_endpoint,
    "explainer": "explainer_sentiment.shap",
    "pipeline" : "finalized_fraud_model.tar.gz",
    "inputs"   : [
        {"name": "TransactionAmt", "min": 0.0,    "max": 10000.0, "default": 50.0,  "step": 1.0},
        {"name": "card1",          "min": 0.0,    "max": 20000.0, "default": 9500.0,"step": 1.0},
        {"name": "card3",          "min": 100.0,  "max": 231.0,   "default": 150.0, "step": 1.0},
        {"name": "C1",             "min": 0.0,    "max": 5000.0,  "default": 1.0,   "step": 1.0},
        {"name": "C12",            "min": 0.0,    "max": 5000.0,  "default": 0.0,   "step": 1.0},
        {"name": "TransactionDT",  "min": 0.0,    "max": 15811200.0, "default": 86400.0, "step": 1.0},
    ]
}

# ── Loaders ────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_pipeline(_session, bucket):
    s3       = _session.client('s3')
    tar_path = os.path.join(tempfile.gettempdir(), 'finalized_fraud_model.tar.gz')
    s3.download_file(Bucket=bucket,
                     Key=f"sklearn-pipeline-deployment/finalized_fraud_model.tar.gz",
                     Filename=tar_path)
    extract_dir = tempfile.gettempdir()
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=extract_dir)
        joblib_file = [f for f in tar.getnames() if f.endswith('.joblib')][0]
    return joblib.load(os.path.join(extract_dir, joblib_file))

@st.cache_resource
def load_shap_explainer(_session, bucket):
    s3         = _session.client('s3')
    local_path = os.path.join(tempfile.gettempdir(), 'explainer_sentiment.shap')
    if not os.path.exists(local_path):
        s3.download_file(Bucket=bucket,
                         Key='explainer/explainer_sentiment.shap',
                         Filename=local_path)
    with open(local_path, "rb") as f:
        return joblib.load(f)

# ── Prediction ─────────────────────────────────────────────────────────────────
def call_model_api(input_dict):
    predictor = Predictor(
        endpoint_name=MODEL_INFO["endpoint"],
        sagemaker_session=sm_session,
        serializer=JSONSerializer(),
        deserializer=JSONDeserializer()
    )
    try:
        raw_pred = predictor.predict(input_dict)
        pred_val = int(raw_pred[0]) if isinstance(raw_pred, list) else int(raw_pred)
        return {0: "Legitimate", 1: "Fraud"}.get(pred_val, str(pred_val)), 200
    except Exception as e:
        return f"Error: {str(e)}", 500

# ── SHAP explanation ───────────────────────────────────────────────────────────
def display_explanation(input_dict):
    pipeline = load_pipeline(session, aws_bucket)
    explainer = load_shap_explainer(session, aws_bucket)

    # Build preprocessing-only pipeline (all steps except the final model)
    preprocessing = Pipeline(steps=pipeline.steps[:-1])
    input_df      = pd.DataFrame([input_dict])
    transformed   = preprocessing.transform(input_df)

    shap_values = explainer(transformed)

    st.subheader("🔍 Decision Transparency (SHAP)")
    fig, ax = plt.subplots(figsize=(10, 4))
    shap.plots.waterfall(shap_values[0, :, 1], show=False)
    st.pyplot(fig)
    plt.close(fig)

    top_feature = (pd.Series(shap_values[0, :, 1].values,
                             index=shap_values[0, :, 1].feature_names)
                   .abs().idxmax())
    st.info(f"**Business Insight:** The most influential factor in this decision was **{top_feature}**.")

# ── UI ─────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Fraud Detection", layout="wide")
st.title("🔐 IEEE-CIS Fraud Detection")
st.caption("Enter transaction details below and click **Run Prediction** to score the transaction.")

with st.form("pred_form"):
    st.subheader("Transaction Inputs")
    cols = st.columns(2)
    user_inputs = {}
    for i, inp in enumerate(MODEL_INFO["inputs"]):
        with cols[i % 2]:
            user_inputs[inp["name"]] = st.number_input(
                inp["name"].replace("_", " ").upper(),
                min_value=float(inp["min"]),
                max_value=float(inp["max"]),
                value=float(inp["default"]),
                step=float(inp["step"])
            )
    submitted = st.form_submit_button("Run Prediction")

if submitted:
    # Merge user inputs on top of a full row of defaults from X_train
    input_dict = dataset.iloc[0].to_dict()
    input_dict.update(user_inputs)

    with st.spinner("Scoring transaction..."):
        res, status = call_model_api(input_dict)

    if status == 200:
        color = "🔴" if res == "Fraud" else "🟢"
        st.metric("Prediction Result", f"{color} {res}")
        display_explanation(input_dict)
    else:
        st.error(res)
