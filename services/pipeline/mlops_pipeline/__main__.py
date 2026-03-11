from mlops_pipeline.utils.logger import setup_logging
from mlops_pipeline.settings import Settings
from mlops_pipeline.repositories.s3_storage import S3Storage
from mlops_pipeline.src.pipeline import PipelineRunner
from mlops_pipeline.configs.s3_storage_config import S3StorageConfig
from dotenv import load_dotenv, find_dotenv



def main():
    setup_logging()
    load_dotenv(find_dotenv())
    settings = Settings()
    s3_storage = S3Storage(settings)
    pipeline_runner = PipelineRunner(S3StorageConfig(), settings, s3_storage)
    df = pipeline_runner.run()
    print(df.head())

if __name__ == "__main__":
    main()