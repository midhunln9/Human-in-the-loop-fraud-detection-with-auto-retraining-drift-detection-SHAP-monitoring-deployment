from mlops_pipeline.protocols.storage_protocol import StorageProtocol
from mlops_pipeline.settings import Settings

from io import BytesIO
import boto3
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class S3Storage(StorageProtocol):
    def __init__(self, settings: Settings):
        self.settings = settings
        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id=self.settings.aws_access_key,
            aws_secret_access_key=self.settings.aws_secret_key,
            region_name=self.settings.region
        )

    def stream_upload_dataframe(self, dataframe: pd.DataFrame, key: str) -> None:
        try:    
            csv_buffer = BytesIO()
            dataframe.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)
            self.s3_client.upload_fileobj(csv_buffer, self.settings.bucket, key)
            logger.info(f"Dataframe uploaded to S3: {key}")
        except Exception as e:
            logger.error(f"Error uploading dataframe to S3: {e}")
            raise 

    def stream_download_dataframe(self, key: str) -> pd.DataFrame:
        try:
            response = self.s3_client.get_object(Bucket=self.settings.bucket, Key=key)
            dataframe = pd.read_csv(response['Body'], sep=',')
            logger.info(f"Dataframe downloaded from S3: {key}")
            return dataframe
        except Exception as e:
            logger.error(f"Error downloading dataframe from S3: {e}")
            raise 