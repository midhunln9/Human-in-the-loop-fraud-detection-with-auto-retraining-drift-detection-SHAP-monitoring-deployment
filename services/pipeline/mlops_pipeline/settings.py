from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    This class manages all application configuration including database connections,
    AWS S3 credentials, and Weights & Biases settings. Values are loaded from
    the .env file with support for environment variable overrides.

    Attributes:
        db_link: Database connection string.
        bucket: AWS S3 bucket name for data and model storage.
        aws_access_key: AWS access key for S3 authentication.
        aws_secret_key: AWS secret key for S3 authentication.
        region: AWS region for S3 services.
        wandb_api_key: API key for Weights & Biases authentication.
        wandb_dir: Directory for W&B local files.
        wandb_cache_dir: Cache directory for W&B artifacts.
        wandb_silent: Whether to suppress W&B console output.
    """

    model_config = SettingsConfigDict(env_file="services/pipeline/.env", env_file_encoding="utf-8", extra="ignore")

    db_link: str = Field(..., validation_alias="db_link")

    bucket: str = Field(..., validation_alias="bucket")
    aws_access_key: str = Field(..., validation_alias="aws_access")
    aws_secret_key: str = Field(..., validation_alias="aws_secret")
    region: str = Field(..., validation_alias="region")

    wandb_api_key: str = Field(..., validation_alias="WANDB_API_KEY")
    wandb_dir: str = Field(..., validation_alias="WANDB_DIR")
    wandb_cache_dir: str = Field(..., validation_alias="WANDB_CACHE_DIR")
    wandb_silent: bool = Field(..., validation_alias="WANDB_SILENT")