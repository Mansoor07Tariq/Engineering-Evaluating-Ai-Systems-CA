from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

class BaseModel(ABC):
    def __init__(self) -> None:
        """
        Base class for ML models. Child classes must implement core methods.
        """
        pass

    @abstractmethod
    def train(self) -> None:
        """
        Train the model on input data.
        To be implemented by each specific model (e.g., RandomForest).
        """
        pass

    @abstractmethod
    def predict(self) -> int:
        """
        Generate predictions from test data.
        Should return model outputs (typically as a NumPy array).
        """
        pass

    @abstractmethod
    def data_transform(self) -> None:
        """
        Placeholder for any data preprocessing (optional).
        Can be left empty if not required.
        """
        return

    def build(self, values={}):
        """
        Update internal settings of the model using a dictionary of parameters.
        """
        if not isinstance(values, dict):
            from utils import string2any  # Only used if values passed as string
            values = string2any(values)

        # Apply default settings first (if defined), then update with given values
        self.__dict__.update(getattr(self, 'defaults', {}))
        self.__dict__.update(values)
        return self
