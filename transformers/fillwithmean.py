import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FillWithMean(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        self.numeric_columns = X.select_dtypes(include=['number']).columns
        self.column_means = X[self.numeric_columns].mean()
        return self

    def transform(self, X):
        X_filled = X.copy()
        X_filled[self.numeric_columns] = X_filled[self.numeric_columns].fillna(self.column_means)
        return X_filled