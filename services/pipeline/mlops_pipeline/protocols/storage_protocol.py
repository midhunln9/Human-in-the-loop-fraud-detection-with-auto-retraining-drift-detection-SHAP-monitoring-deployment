from typing import Protocol
import pandas as pd
from typing import Any

class StorageProtocol(Protocol):
    def stream_upload_dataframe(self, dataframe: pd.DataFrame, key: str) -> None:
        ...

    def stream_download_dataframe(self, key : str) -> pd.DataFrame:
        ...
    def upload_object(self, object: Any, key: str) -> None:
        ...