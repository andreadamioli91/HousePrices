import pandas as pd


class ColumnSelector:

    def columns_with_more_than_nans(df, threshold=400):
        columns_with_nans = df.columns[df.isna().sum() > threshold]
        return columns_with_nans.tolist()
