import pandas as pd
from sklearn.pipeline import Pipeline

from loader.csvloader import CsvLoader
from loader.datavisualizer import DataVisualizer
from transformers.dropnacolumns import DropColumns
from transformers.fillwithmean import FillWithMean
from transformers.onehotencoder import OneHotEncoderColumns
from utils.columnselector import ColumnSelector

TRAIN_DATASET = "data/train.csv"
TEST_DATASET = "data/test.csv"

loader = CsvLoader()
data_visualizer = DataVisualizer()

if __name__ == '__main__':
    train_initial = loader.load_data(TRAIN_DATASET)
    test_initial = loader.load_data(TEST_DATASET)

    many_nans_columns = ColumnSelector.many_nans_columns(train_initial, 400)
    low_card_columns = ColumnSelector.low_cardinality_columns(train_initial, 4)
    low_card_columns = [item for item in low_card_columns if item not in many_nans_columns]

    pd.set_option('display.max_columns', None)

    transformer_pipeline = Pipeline([
        ("dropIfTooManyNa", DropColumns(many_nans_columns)),
        ("fillWithMean", FillWithMean()),
        ("oneHotEncoder", OneHotEncoderColumns(low_card_columns)),
    ])

    train_transformed = transformer_pipeline.fit_transform(train_initial)

    print(train_transformed.shape)

    print("Finished")
