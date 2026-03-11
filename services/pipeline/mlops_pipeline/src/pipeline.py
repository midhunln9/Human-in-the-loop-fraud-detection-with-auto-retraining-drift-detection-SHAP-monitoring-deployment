from mlops_pipeline.configs.s3_storage_config import S3StorageConfig
from mlops_pipeline.configs.transformation_config import TransformationConfig
from mlops_pipeline.settings import Settings
from mlops_pipeline.protocols.storage_protocol import StorageProtocol
from mlops_pipeline.src.data_ingestion import DataIngestion
from mlops_pipeline.src.data_transformation import DataTransformation
import logging

logger = logging.getLogger(__name__)


class PipelineRunner:
    def __init__(self, s3_config : S3StorageConfig, 
    settings: Settings, 
    repository: StorageProtocol, 
    transformation_config: TransformationConfig):
        self.s3_config = s3_config
        self.settings = settings
        self.repository = repository
        self.transformation_config = transformation_config
    
    def run(self):
        data_ingestion = DataIngestion(self.s3_config, self.repository)
        df = data_ingestion.ingest_data()
        transformation = DataTransformation(self.transformation_config)
        split_datasets = transformation.transform_data(df)
        logger.info(f"Data transformation completed successfully")

    
        
