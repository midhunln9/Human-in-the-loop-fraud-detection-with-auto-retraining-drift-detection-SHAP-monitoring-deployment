import logging
from sklearn.model_selection import train_test_split
import pandas as pd

from mlops_pipeline.configs.transformation_config import TransformationConfig
from mlops_pipeline.src.types import SplitDatasets
logger = logging.getLogger(__name__)

class DataTransformation:
    def __init__(self, config: TransformationConfig):
        self.config = config
    
    def transform_data(self, df: pd.DataFrame):
        """Perform train tet split and store data at the data contract inside types.py
        Args:
            df: pd.DataFrame
        Returns:
            SplitDatasets: The split datasets.
        """
        try : 
            X = df.drop(columns=[self.config.target_column])
            logger.info(f"Target column dropped successfully")
        except Exception as e:
            logger.error(f"Error dropping target column: {e}")
            raise ValueError(f"check the target column: {self.config.target_column}")
        y = df[self.config.target_column]
        logger.info(f"Splitting data into train, validation and test sets")
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=self.config.test_size
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=self.config.val_size
        )
        logger.info(f"Data split completed successfully")
        return SplitDatasets(X_train=X_train, X_val=X_val, X_test=X_test, y_train=y_train, y_val=y_val, y_test=y_test)
    