import pandas as pd


class ColumnSelector:

    def many_nans_columns(df, threshold=400):
        columns_with_nans = df.columns[df.isna().sum() > threshold]
        return columns_with_nans.tolist()

    def low_cardinality_columns(df, threshold=5):
        low_cardinality_cols = []
        for column in df.columns:
            unique_values = df[column].nunique()
            if unique_values < threshold:
                low_cardinality_cols.append(column)
        return low_cardinality_cols
