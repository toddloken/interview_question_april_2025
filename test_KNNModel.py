from dev_code.ModelKNearestNeighbors import KNNModel

from typing import List
from typing import Tuple
from dev_code.DataProcessor import DataProcessor
from TrainTestSplit import TrainTestSplit

import pandas

SALES_PATH = "../data/kc_house_data.csv"  # path to CSV with home sale data
DEMOGRAPHICS_PATH = "../data/kc_house_data.csv"  # path to CSV with demographics
# List of columns (subset) that will be taken from home sale data
SALES_COLUMN_SELECTION = [
    'price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
    'sqft_above', 'sqft_basement', 'zipcode'
]
OUTPUT_DIR = "../model"  # Directory where output artifacts will be saved


def load_data(
    sales_path: str, demographics_path: str, sales_column_selection: List[str]
) -> Tuple[pandas.DataFrame, pandas.Series]:
    """Load the target and feature data by merging sales and demographics.

    Args:
        sales_path: path to CSV file with home sale data
        demographics_path: path to CSV file with home sale data
        sales_column_selection: list of columns from sales data to be used as
            features

    Returns:
        Tuple containing with two elements: a DataFrame and a Series of the same
        length.  The DataFrame contains features for machine learning, the
        series contains the target variable (home sale price).

    """
    data = pandas.read_csv(sales_path,
                           usecols=sales_column_selection,
                           dtype={'zipcode': str})
    demographics = pandas.read_csv("../data/zipcode_demographics.csv",
                                   dtype={'zipcode': str})

    merged_data = data.merge(demographics, how="left",
                             on="zipcode").drop(columns="zipcode")
    # # Remove the target variable from the dataframe, features will remain
    # y = merged_data.pop('price')
    # x = merged_data
    # df.rename(columns={'A': 'X'}, inplace=True)


    return merged_data


"""Load data, train model, and export artifacts."""
df = load_data(SALES_PATH, DEMOGRAPHICS_PATH, SALES_COLUMN_SELECTION)
processor = DataProcessor(df)
df = processor.process()
ttsplit = TrainTestSplit(df)
X_train, X_test, y_train, y_test = ttsplit.balance_and_split()
print('BuildingKNNModel')
model = KNNModel(X_train, X_test, y_train, y_test)
knn_predictions = model.train_model()


