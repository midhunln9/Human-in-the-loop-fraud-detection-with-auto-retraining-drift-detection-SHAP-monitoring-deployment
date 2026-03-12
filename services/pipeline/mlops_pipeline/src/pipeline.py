from mlops_pipeline.configs.s3_storage_config import S3StorageConfig
from mlops_pipeline.configs.transformation_config import TransformationConfig
from mlops_pipeline.configs.preprocessing_config import PreprocessingConfig
from mlops_pipeline.settings import Settings
from mlops_pipeline.protocols.storage_protocol import StorageProtocol
from mlops_pipeline.src.data_ingestion import DataIngestion
from mlops_pipeline.src.data_transformation import DataTransformation
from mlops_pipeline.src.data_preprocessing import DataPreprocessing
from mlops_pipeline.protocols.model_versioning_protocol import ModelVersioningProtocol
import logging
from typing import List
from mlops_pipeline.src.master_tuner import MasterTuner
from mlops_pipeline.strategies.base_strategy import BaseModelStrategy
from mlops_pipeline.src.model_promotion import ModelPromotion
from mlops_pipeline.configs.wandb_config import WandbConfig
from mlops_pipeline.src.model_trainer import ModelTrainer
logger = logging.getLogger(__name__)


class PipelineRunner:
    def __init__(self, s3_config : S3StorageConfig, settings: Settings, 
    storage_repository: StorageProtocol, model_versioning_repository: ModelVersioningProtocol, 
    transformation_config: TransformationConfig, preprocessing_config: PreprocessingConfig,
    strategies : List[BaseModelStrategy], wandb_config: WandbConfig):
        self.s3_config = s3_config
        self.settings = settings
        self.storage_repository = storage_repository
        self.transformation_config = transformation_config
        self.preprocessing_config = preprocessing_config
        self.model_versioning_repository = model_versioning_repository
        self.strategies = strategies
        self.wandb_config = wandb_config
    
    def run(self):
        data_ingestion = DataIngestion(self.s3_config, self.storage_repository)
        df = data_ingestion.ingest_data()

        transformation = DataTransformation(self.transformation_config)
        split_datasets = transformation.transform_data(df)

        preprocessing = DataPreprocessing(self.preprocessing_config, 
        split_datasets, self.model_versioning_repository)
        preprocessed_datasets = preprocessing.preprocess_data()

        del split_datasets # clear memory of unwanted datasets

        self.storage_repository.stream_upload_dataframe(preprocessed_datasets.X_train, self.s3_config.preprocessed_train_data_key_features)
        self.storage_repository.stream_upload_dataframe(preprocessed_datasets.X_val, self.s3_config.preprocessed_validation_data_key_features)
        self.storage_repository.stream_upload_dataframe(preprocessed_datasets.X_test, self.s3_config.preprocessed_test_data_key_features)
        self.storage_repository.stream_upload_dataframe(preprocessed_datasets.y_train, self.s3_config.preprocessed_train_labels_key)
        self.storage_repository.stream_upload_dataframe(preprocessed_datasets.y_val, self.s3_config.preprocessed_validation_labels_key)
        self.storage_repository.stream_upload_dataframe(preprocessed_datasets.y_test, self.s3_config.preprocessed_test_labels_key)

        self.strategies = [strategy(preprocessed_datasets) for strategy in self.strategies]
        hyperparameter_tuner = MasterTuner(strategies = self.strategies, model_versioning_repository = self.model_versioning_repository)
        best_result = hyperparameter_tuner.start_hyperparameter_tuning()
        trainer = ModelTrainer(preprocessed_datasets, best_result, self.model_versioning_repository)
        best_model, pr_auc_score = trainer.combine_data_and_train_model()
        model_promotion = ModelPromotion(best_model, pr_auc_score, self.model_versioning_repository, self.wandb_config)
        model_promotion.promote_model()
        logger.info(f"Pipeline completed successfully")








    
        
