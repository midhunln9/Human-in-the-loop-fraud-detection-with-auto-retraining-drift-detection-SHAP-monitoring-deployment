from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file="services/mlops_pipeline/.env", env_file_encoding="utf-8", extra="ignore")

    db_link: str = Field(..., alias="db_link")
    
    bucket: str = Field(..., alias="bucket")
    aws_access_key: str = Field(..., alias="aws_access")
    aws_secret_key: str = Field(..., alias="aws_secret")
    region: str = Field(..., alias="region")
    
    wandb_api_key: str = Field(..., alias="WANDB_API_KEY")
    wandb_dir: str = Field(..., alias="WANDB_DIR")
    wandb_cache_dir: str = Field(..., alias="WANDB_CACHE_DIR")
    wandb_silent: bool = Field(..., alias="WANDB_SILENT")