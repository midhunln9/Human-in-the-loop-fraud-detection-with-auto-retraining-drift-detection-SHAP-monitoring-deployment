from mlops_pipeline.protocols.storage_protocol import StorageProtocol
from typing import Dict
import pandas as pd

class FakeStorageRepository(StorageProtocol):
    def __init__(self):
        self.storage : Dict[str, pd.DataFrame]= {}
    
    def stream_upload_dataframe(self, dataframe: pd.DataFrame, key: str) -> None:
        self.storage[key] = dataframe.copy()
    
    def stream_download_dataframe(self, key: str) -> pd.DataFrame:
        if key not in self.storage:
            raise FileNotFoundError(f"Key not found in storage")
        return self.storage[key].copy()