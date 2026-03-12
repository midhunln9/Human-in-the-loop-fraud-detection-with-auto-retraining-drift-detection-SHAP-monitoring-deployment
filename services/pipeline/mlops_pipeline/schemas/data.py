import pandas as pd
from dataclasses import dataclass

@dataclass
class SplitDatasets():
    """Immutable container returned by the transformation stage."""
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series

@dataclass
class PreprocessedDatasets():
    """Immutable container returned by the preprocessing stage."""
    X_train: pd.DataFrame 
    X_val: pd.DataFrame 
    X_test: pd.DataFrame 
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series

