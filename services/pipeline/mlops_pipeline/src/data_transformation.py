import logging
from sklearn.model_selection import train_test_split
import pandas as pd

from mlops_pipeline.configs.transformation_config import TransformationConfig
from mlops_pipeline.schemas.data import SplitDatasets
from mlops_pipeline.exceptions import TransformationError

logger = logging.getLogger(__name__)


class DataTransformation:
    """Handles data transformation including train/validation/test splitting.

    This class splits the raw data into training, validation, and test sets
    according to the configured ratios, separating features from the target column.

    Attributes:
        config: Configuration containing split ratios and target column name.
    """

    def __init__(self, config: TransformationConfig):
        """Initialize the DataTransformation with configuration.

        Args:
            config: Transformation configuration specifying test/validation sizes,
                   random state, and target column.
        """
        self.config = config

    def transform_data(self, df: pd.DataFrame) -> SplitDatasets:
        """Split the input DataFrame into train, validation, and test sets.

        Separates features from the target column and performs stratified
        splitting to create training, validation, and test datasets.

        Args:
            df: The input DataFrame containing features and target column.

        Returns:
            A SplitDatasets dataclass containing X_train, X_val, X_test,
            y_train, y_val, and y_test.

        Raises:
            TransformationError: If the target column is missing or splitting fails.
        """
        try : 
            X = df.drop(columns=[self.config.target_column])
            logger.info(f"Target column dropped successfully")
        except Exception as e:
            logger.error(f"Error dropping target column: {e}")
            raise TransformationError(f"check the target column: {self.config.target_column}") from e
        y = df[self.config.target_column]
        logger.info(f"Splitting data into train, validation and test sets")
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=self.config.test_size, random_state = self.config.random_state
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=self.config.val_size, random_state = self.config.random_state
        )
        logger.info(f"Data split completed successfully")
        return SplitDatasets(X_train=X_train, X_val=X_val, X_test=X_test, y_train=y_train, y_val=y_val, y_test=y_test)
    