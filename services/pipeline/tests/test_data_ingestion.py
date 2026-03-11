import pandas as pd
from mlops_pipeline.src.data_ingestion import DataIngestion
import pytest

def test_for_data_ingestion(fake_storage, s3_config):
    # upload data into fake storage first
    df = pd.DataFrame({
        "col1": [1, 2, 3],
        "col2": [4, 5, 6]
    })
    fake_storage.storage[s3_config.raw_data_key] = df.copy()
    # start to test ingestion of data
    data_ingestion = DataIngestion(s3_config, fake_storage)
    df = data_ingestion.ingest_data()
    # assert the data is ingested correctly
    assert df.equals(fake_storage.storage[s3_config.raw_data_key])

