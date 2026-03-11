from mlops_pipeline.configs.s3_storage_config import S3StorageConfig
from mlops_pipeline.protocols.storage_protocol import StorageProtocol
import logging

logger = logging.getLogger(__name__)

class DataIngestion:
    """DataIngestion class to ingest csv data as dataframe from S3."""
    def __init__(self, config : S3StorageConfig, repository: StorageProtocol):
        """Initialize the DataIngestion class.
        Args:
            config : S3StorageConfig
            repository : StorageProtocol
        """
        self.config = config
        self.repository = repository
    
    def ingest_data(self):
        """Ingest data from S3 and return a dataframe.
        Returns:
            pd.DataFrame: The dataframe containing the ingested data.
        """
        try:
            df = self.repository.stream_download_dataframe(self.config.raw_data_key)
            return df
        except Exception as e:
            logger.error(f"Error ingesting data: {e}")
            raise ValueError(f"check the data key: {self.config.raw_data_key}")
    
