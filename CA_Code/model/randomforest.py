import numpy as np
import pandas as pd
from model.base import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import random

#Set consistent seed for repeatable results
seed = 0
np.random.seed(seed)
random.seed(seed)

#Console display settings
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 200)

class RandomForest(BaseModel):
    def __init__(self, name: str, features: np.ndarray, labels: np.ndarray) -> None:
        """
        Constructor to set up the Random Forest model.

        Parameters:
        name: Identifier for the model (used in logs)
        features: Input feature matrix
        labels: Target labels
        """
        super(RandomForest, self).__init__()
        self.name = name
        self.features = features
        self.labels = labels

        # RandomForestClassifier setup
        self.model = RandomForestClassifier(
            n_estimators=1000,
            random_state=seed,
            class_weight='balanced_subsample'
        )

        self.output = None
        self.preprocess_data()  # Optional preprocessing (currently unused)

    def train(self, dataset) -> None:
        """
        Train the classifier using the training subset of the dataset.
        """
        self.model.fit(dataset.X_train, dataset.y_train)

    def predict(self, X_test: pd.Series):
        """
        Predict labels for the given test data.

        Returns:
        Predictions as a NumPy array
        """
        self.output = self.model.predict(X_test)
        return self.output

    def print_results(self, dataset):
        """
        Print performance metrics for evaluation.
        """
        print(f"\nEvaluation Summary for Model: {self.name}")
        print(classification_report(dataset.y_test, self.output))

    def preprocess_data(self) -> None:
        """
        Optional preprocessing placeholder (currently unused).
        """
        pass

    def data_transform(self) -> None:
        """
        Implementation of abstract method from BaseModel.
        Required to avoid instantiation error.
        """
        pass
