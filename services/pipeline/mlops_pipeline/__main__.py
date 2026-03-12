from mlops_pipeline.utils.logger import setup_logging
from mlops_pipeline.settings import Settings
from mlops_pipeline.repositories.s3_storage import S3Storage
from mlops_pipeline.src.pipeline import PipelineRunner
from mlops_pipeline.configs.s3_storage_config import S3StorageConfig
from dotenv import load_dotenv, find_dotenv
from mlops_pipeline.configs.transformation_config import TransformationConfig
from mlops_pipeline.configs.preprocessing_config import PreprocessingConfig
from mlops_pipeline.repositories.wandb_repository import WandbRepository
from mlops_pipeline.configs.wandb_config import WandbConfig
from mlops_pipeline.src.xgboost_strategy import XGBoostStrategy
from mlops_pipeline.src.lightgbm_strategy import LightGBMStrategy
from mlops_pipeline.schemas.data import PreprocessedDatasets



def main():
    setup_logging()
    load_dotenv(find_dotenv())
    settings = Settings()
    s3_storage = S3Storage(settings)
    transformation_config = TransformationConfig()
    preprocessing_config = PreprocessingConfig()
    s3_config = S3StorageConfig()
    wandb_config = WandbConfig()
    wandb_repository = WandbRepository(wandb_config)
    strategies = [XGBoostStrategy, LightGBMStrategy]
    pipeline_runner = PipelineRunner(s3_config, settings, s3_storage, wandb_repository, 
    transformation_config, preprocessing_config, strategies)
    pipeline_runner.run()
    

if __name__ == "__main__":
    main()