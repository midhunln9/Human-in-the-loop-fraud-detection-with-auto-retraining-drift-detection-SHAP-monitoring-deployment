from typing import NamedTuple

import pandas as pd


class SplitDatasets(NamedTuple):
    """Immutable container returned by the transformation stage."""
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.DataFrame
    y_val: pd.DataFrame
    y_test: pd.DataFrame


class PreprocessedDatasets(NamedTuple):
    """Immutable container returned by the preprocessing stage."""
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.DataFrame
    y_val: pd.DataFrame
    y_test: pd.DataFrame
    columns: list[str]
