import pandas as pd
from sklearn.pipeline import Pipeline

from loader.csvloader import CsvLoader
from loader.datavisualizer import DataVisualizer
from transformers.dropnacolumns import DropColumns
from transformers.fillwithmean import FillWithMean
from utils.columnselector import ColumnSelector

TRAIN_DATASET = "data/train.csv"
TEST_DATASET = "data/test.csv"

loader = CsvLoader()
data_visualizer = DataVisualizer()

if __name__ == '__main__':
    train_initial = loader.load_data(TRAIN_DATASET)
    test_initial = loader.load_data(TEST_DATASET)

    cols_with_many_nan = ColumnSelector.columns_with_more_than_nans(train_initial, 400)

    pd.set_option('display.max_columns', None)

    transformer_pipeline = Pipeline([
        ("dropIfTooManyNa", DropColumns(cols_with_many_nan)),
        ("fillWithMean", FillWithMean())
    ])

    train_transformed = transformer_pipeline.fit_transform(train_initial)


    print("Finished")

