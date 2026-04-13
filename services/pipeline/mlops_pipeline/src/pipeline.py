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
from mlops_pipeline.src.model_evaluation import ModelEvaluation
from mlops_pipeline.exceptions import PipelineError
import time

logger = logging.getLogger(__name__)


class PipelineRunner:
    """Orchestrates the complete MLOps pipeline execution.

    This class manages the end-to-end machine learning pipeline including
    data ingestion, transformation, preprocessing, hyperparameter tuning,
    model training, evaluation, and promotion.

    Attributes:
        s3_config: Configuration for S3 storage paths and keys.
        settings: Application settings including credentials.
        storage_repository: Storage protocol implementation for S3 operations.
        transformation_config: Configuration for data transformation.
        preprocessing_config: Configuration for data preprocessing.
        model_versioning_repository: Model versioning protocol implementation.
        strategies: List of model strategy classes for hyperparameter tuning.
        wandb_config: Weights & Biases configuration for model versioning.
    """

    def __init__(self, s3_config: S3StorageConfig, settings: Settings,
                 storage_repository: StorageProtocol, model_versioning_repository: ModelVersioningProtocol,
                 transformation_config: TransformationConfig, preprocessing_config: PreprocessingConfig,
                 strategies: List[BaseModelStrategy], wandb_config: WandbConfig):
        """Initialize the PipelineRunner with all required configurations and repositories.

        Args:
            s3_config: S3 storage configuration for data and model paths.
            settings: Application settings containing AWS and W&B credentials.
            storage_repository: Storage implementation for S3 operations.
            model_versioning_repository: Model versioning implementation for artifact management.
            transformation_config: Configuration for train/test/validation splitting.
            preprocessing_config: Configuration for feature preprocessing.
            strategies: List of model strategy classes to evaluate.
            wandb_config: Weights & Biases configuration for artifact logging.
        """
        self.s3_config = s3_config
        self.settings = settings
        self.storage_repository = storage_repository
        self.transformation_config = transformation_config
        self.preprocessing_config = preprocessing_config
        self.model_versioning_repository = model_versioning_repository
        self.strategies = strategies
        self.wandb_config = wandb_config

    def run(self):
        """Execute the complete MLOps pipeline.

        Runs all pipeline stages sequentially:
        1. Data ingestion from S3
        2. Data transformation (train/test/validation split)
        3. Data preprocessing (scaling, imputation)
        4. Upload preprocessed data to S3
        5. Hyperparameter tuning with multiple strategies
        6. Model training on combined data
        7. Upload trained model to S3
        8. Model evaluation with Evidently reports
        9. Model promotion to production

        Logs timing information for each stage.

        Returns:
            None
        """
        # step 0 : ingest data from S3
        t_start = time.perf_counter()
        data_ingestion = DataIngestion(self.s3_config, self.storage_repository)
        df = data_ingestion.ingest_data()
        t1 = time.perf_counter()
        logger.info(f"Data ingestion completed in {t1 - t_start} seconds")

        # step 1 : transform data
        t0 = time.perf_counter()
        transformation = DataTransformation(self.transformation_config)
        split_datasets = transformation.transform_data(df)
        t1 = time.perf_counter()
        logger.info(f"Data transformation completed in {t1 - t0} seconds")

        # step 2 : preprocess data
        t0 = time.perf_counter()
        preprocessing = DataPreprocessing(self.preprocessing_config, 
        split_datasets, self.model_versioning_repository)
        preprocessed_datasets = preprocessing.preprocess_data()
        t1 = time.perf_counter()
        logger.info(f"Data preprocessing completed in {t1 - t0} seconds")

        # step 3 : upload data to S3
        t0 = time.perf_counter()
        self.storage_repository.stream_upload_dataframe(preprocessed_datasets.X_train, self.s3_config.preprocessed_train_data_key_features)
        self.storage_repository.stream_upload_dataframe(preprocessed_datasets.X_val, self.s3_config.preprocessed_validation_data_key_features)
        self.storage_repository.stream_upload_dataframe(preprocessed_datasets.X_test, self.s3_config.preprocessed_test_data_key_features)
        self.storage_repository.stream_upload_dataframe(preprocessed_datasets.y_train, self.s3_config.preprocessed_train_labels_key)
        self.storage_repository.stream_upload_dataframe(preprocessed_datasets.y_val, self.s3_config.preprocessed_validation_labels_key)
        self.storage_repository.stream_upload_dataframe(preprocessed_datasets.y_test, self.s3_config.preprocessed_test_labels_key)
        t1 = time.perf_counter()
        logger.info(f"Data uploaded to S3 in {t1 - t0} seconds")

        # step 4 : hyperparameter tuning
        t0 = time.perf_counter()
        self.strategies = [strategy(preprocessed_datasets) for strategy in self.strategies]
        hyperparameter_tuner = MasterTuner(strategies = self.strategies, model_versioning_repository = self.model_versioning_repository)
        best_result = hyperparameter_tuner.start_hyperparameter_tuning()
        trainer = ModelTrainer(preprocessed_datasets, best_result, self.model_versioning_repository)
        t1 = time.perf_counter()
        logger.info(f"Hyperparameter tuning completed in {t1 - t0} seconds")

        # step 5 : train model
        t0 = time.perf_counter()
        best_model = trainer.combine_data_and_train_model()
        t1 = time.perf_counter()
        logger.info(f"Model trained in {t1 - t0} seconds")

        # step 6 : upload model to S3
        t0 = time.perf_counter()
        self.storage_repository.upload_object(best_model, self.s3_config.model_key)
        t1 = time.perf_counter()
        logger.info(f"Model uploaded to S3 in {t1 - t0} seconds")

        # step 7 : evaluate model
        t0 = time.perf_counter()
        model_evaluation = ModelEvaluation(best_model, self.storage_repository, self.s3_config, preprocessed_datasets, 1)
        pr_auc_test = model_evaluation.evaluate()
        t1 = time.perf_counter()
        logger.info(f"Model evaluated in {t1 - t0} seconds")

        # step 8 : promote model
        t0 = time.perf_counter()
        model_promotion = ModelPromotion(pr_auc_test, self.model_versioning_repository, self.wandb_config)
        model_promotion.promote_model()
        t1 = time.perf_counter()
        logger.info(f"Model promoted in {t1 - t0} seconds")

        logger.info(f"Pipeline completed successfully in {time.perf_counter() - t_start} seconds")