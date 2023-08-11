import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class OneHotEncoderColumns(BaseEstimator, TransformerMixin):

    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.columns is None:
            return X

        X_encoded = pd.get_dummies(X, columns=self.columns)
        return X_encoded