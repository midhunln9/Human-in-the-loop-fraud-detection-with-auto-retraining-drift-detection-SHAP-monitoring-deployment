class PipelineError(Exception):
    """Base exception for all pipeline errors."""


class IngestionError(PipelineError):
    """Raised when data ingestion fails."""


class TransformationError(PipelineError):
    """Raised when data transformation fails."""


class PreprocessingError(PipelineError):
    """Raised when data preprocessing fails."""


class TuningError(PipelineError):
    """Raised when hyperparameter tuning fails."""


class EvaluationError(PipelineError):
    """Raised when model evaluation fails."""


class PromotionError(PipelineError):
    """Raised when model promotion fails."""


class StorageError(PipelineError):
    """Raised when storage operations fail."""


class ArtifactError(PipelineError):
    """Raised when artifact repository operations fail."""

class TrainingError(PipelineError):
    """Raised when model training fails."""
