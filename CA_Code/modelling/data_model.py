import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import random
from Config import *

#Ensure consistent results across runs
seed = 0
random.seed(seed)
np.random.seed(seed)

class ProcessedData:
    def __init__(self, features: np.ndarray, records: pd.DataFrame, label_col='y', vectorizer=None):
        self.records = records
        self.label_col = label_col
        self.vectorizer = vectorizer
        self.features = features

        #Extract label values
        labels = records[label_col].to_numpy()
        label_series = pd.Series(labels)

        #Only keep classes with at least 3 records
        frequent_classes = label_series.value_counts()[label_series.value_counts() >= 3].index

        if len(frequent_classes) < 1:
            print("Skipping - No class has 3 or more records.")
            self.X_train = None
            return

        #Mask to filter only good classes
        valid_mask = label_series.isin(frequent_classes).values
        filtered_labels = labels[valid_mask]
        filtered_features = features[valid_mask]

        #Adjust test size based on original data size
        adjusted_test_size = features.shape[0] * 0.2 / filtered_features.shape[0]

        #Split into training and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            filtered_features, filtered_labels, test_size=adjusted_test_size,
            random_state=0, stratify=filtered_labels
        )

        self.final_labels = filtered_labels
        self.class_names = frequent_classes

    def get_labels(self, specific_label=None):
        if specific_label:
            return self.records[specific_label].to_numpy()
        return self.final_labels

    def get_X_train(self):
        return self.X_train

    def get_X_test(self):
        return self.X_test

    def get_y_train(self):
        return self.y_train

    def get_y_test(self):
        return self.y_test

    def get_features(self):
        return self.features

    def clone_with_new_features(self, updated_features, new_label):
        """
        When we create new text (e.g., combined intent + tone),
        we generate new TF-IDF features.
        This function wraps them in a new ProcessedData object.
        """
        last_n_rows = updated_features.shape[0]
        matching_records = self.records.iloc[-last_n_rows:].copy()

        return ProcessedData(
            updated_features, matching_records,
            label_col=new_label,
            vectorizer=self.vectorizer
        )
