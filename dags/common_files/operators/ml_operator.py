from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from typing import List, Optional, Tuple
from sklearn.metrics import accuracy_score

class FeatureExtractor:
    def __init__(self, input_file: str=None):
        self.input_file = input_file

    def extract_ml_features_target(self) -> Tuple[pd.DataFrame, pd.Series]:
        # Extract features and target from the processed data
        df = pd.read_csv(self.input_file)
        X = df.drop(['metadata.name', 'metadata.content', 'interests', 'is_finance_related'], axis=1)
        y = df['is_finance_related']
        return X, y
    
class DataProcessorForML:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def process_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        # Extract relevant columns
        df = self.data[['metadata.content', 'is_finance_related']].copy()

        # Convert target to binary
        target = df['is_finance_related'].apply(lambda x: 1 if x else 0)

        # Process text column
        vectorizer = CountVectorizer(stop_words='english')
        features = vectorizer.fit_transform(df['metadata.content']).todense()

        # Encode target
        le = LabelEncoder()
        target = le.fit_transform(target)

        # Create feature and target dataframes
        self.features_df = pd.DataFrame(features, columns=vectorizer.get_feature_names())
        self.target_series = pd.Series(target)

        return self.features_df, self.target_series

class DataSplitter:
    def __init__(self, X: pd.DataFrame = None, y: pd.Series = None):
        self.X = X
        self.y = y

    def split_train_test(self, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        # Split data into train and test sets with stratification
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=42, stratify=self.y
        )
        return self.X_train, self.X_test, self.y_train, self.y_test

class ModelTrainer:
    def __init__(self, X_train: pd.DataFrame = None, y_train: pd.Series = None):
        self.X_train = X_train
        self.y_train = y_train
        self.model = None

    def train(self):
        # Train a logistic regression model on the train set
        model = LogisticRegression()
        # model.fit(self.X_train, self.y_train)
        self.model = model


class ModelTester:
    def __init__(self, model = None, X_test: pd.DataFrame = None, y_test: pd.Series = None):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test

    def test(self) -> float:
        # Test the trained model on the test set and return accuracy score
        y_pred = self.model.predict(self.X_test)
        return accuracy_score(self.y_test, y_pred)