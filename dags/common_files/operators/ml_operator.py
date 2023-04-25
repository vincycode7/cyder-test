# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
import pandas as pd
from typing import List, Optional, Tuple

class FeatureExtractor:
    def __init__(self, input_file: str=None):
        self.input_file = input_file

    def extract_ml_features_target(self) -> Tuple[pd.DataFrame, pd.Series]:
        # Extract features and target from the processed data
        pass


class DataSplitter:
    def __init__(self, X: pd.DataFrame = None, y: pd.Series = None):
        self.X = X
        self.y = y

    def split_train_test(self, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        # Split data into train and test sets
        pass


class ModelTrainer:
    def __init__(self, X_train: pd.DataFrame = None, y_train: pd.Series = None):
        self.X_train = X_train
        self.y_train = y_train

    def train(self):
        # Train a model on the train set
        pass


class ModelTester:
    def __init__(self, model = None, X_test: pd.DataFrame = None, y_test: pd.Series = None):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test

    def test(self) -> float:
        # Test the trained model on the test set and return accuracy score
        pass