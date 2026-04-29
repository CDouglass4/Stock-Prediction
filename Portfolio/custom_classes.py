import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder


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
