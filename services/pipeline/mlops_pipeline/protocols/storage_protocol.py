from typing import Protocol
import pandas as pd

class StorageProtocol(Protocol):
    def stream_upload_dataframe(self, dataframe: pd.DataFrame, key: str) -> None:
        ...

    def stream_download_dataframe(self, key : str) -> pd.DataFrame:
        ...