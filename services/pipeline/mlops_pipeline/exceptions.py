"""Custom exceptions for the MLOps pipeline.

This module defines the exception hierarchy used throughout the pipeline
for handling and categorizing errors during ML operations.
"""


class PipelineError(Exception):
    """Base exception class for all pipeline-related errors.

    This is the root exception class that all pipeline-specific exceptions
    inherit from. It can be used to catch any pipeline-related error.
    """


class IngestionError(PipelineError):
    """Raised when data ingestion from S3 fails.

    This exception is raised when there are issues downloading or reading
    raw data from the configured S3 storage.
    """


class TransformationError(PipelineError):
    """Raised when data transformation fails.

    This exception is raised when there are issues during train/test/validation
    splitting or when the target column cannot be found or separated.
    """


class PreprocessingError(PipelineError):
    """Raised when data preprocessing fails.

    This exception is raised when there are issues during feature scaling,
    imputation, or when logging preprocessor artifacts fails.
    """


class TuningError(PipelineError):
    """Raised when hyperparameter tuning fails.

    This exception is raised when Optuna optimization fails or when there are
    issues during the hyperparameter search process.
    """


class EvaluationError(PipelineError):
    """Raised when model evaluation fails.

    This exception is raised when there are issues calculating metrics,
    generating Evidently reports, or uploading evaluation results.
    """


class PromotionError(PipelineError):
    """Raised when model promotion fails.

    This exception is raised when there are issues comparing production
    and staging models or when promoting artifacts between aliases.
    """


class StorageError(PipelineError):
    """Raised when S3 storage operations fail.

    This exception is raised when there are issues uploading or downloading
    dataframes, models, or reports to/from S3 storage.
    """


class ArtifactError(PipelineError):
    """Raised when model versioning operations fail.

    This exception is raised when there are issues creating, logging,
    loading, or promoting artifacts in the W&B model registry.
    """


class TrainingError(PipelineError):
    """Raised when model training fails.

    This exception is raised when there are issues during model fitting
    or when the best model from hyperparameter tuning cannot be instantiated.
    """
