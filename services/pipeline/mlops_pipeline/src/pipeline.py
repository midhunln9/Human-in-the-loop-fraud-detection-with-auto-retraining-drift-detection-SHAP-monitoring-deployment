from mlops_pipeline.configs.s3_storage_config import S3StorageConfig
from mlops_pipeline.settings import Settings
from mlops_pipeline.protocols.storage_protocol import StorageProtocol
from mlops_pipeline.src.data_ingestion import DataIngestion
import logging

logger = logging.getLogger(__name__)


class PipelineRunner:
    def __init__(self, config : S3StorageConfig, settings: Settings, repository: StorageProtocol):
        self.config = config
        self.settings = settings
        self.repository = repository
    
    def run(self):
        data_ingestion = DataIngestion(self.config, self.repository)
        df = data_ingestion.ingest_data()
        logger.info("Data ingestion completed successfully")
        return df
        
