from mlops_pipeline.configs.s3_storage_config import S3StorageConfig
from mlops_pipeline.protocols.storage_protocol import StorageProtocol

class DataIngestion:
    def __init__(self, config : S3StorageConfig, repository: StorageProtocol):
        self.config = config
        self.repository = repository
    
    def ingest_data(self):
        df = self.repository.stream_download_dataframe(self.config.raw_data_key)
        return df
    
    
