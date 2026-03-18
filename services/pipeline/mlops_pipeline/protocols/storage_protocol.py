from typing import Protocol
import pandas as pd
from typing import Any


class StorageProtocol(Protocol):
    """Protocol defining the interface for storage operations.

    This protocol specifies the contract that any storage implementation
    must fulfill to be used within the MLOps pipeline for data persistence.
    """

    def stream_upload_dataframe(self, dataframe: pd.DataFrame, key: str) -> None:
        """Upload a pandas DataFrame to storage.

        Args:
            dataframe: The DataFrame to upload.
            key: The storage key/path where the DataFrame should be saved.

        Returns:
            None
        """
        ...

    def stream_download_dataframe(self, key: str) -> pd.DataFrame:
        """Download a pandas DataFrame from storage.

        Args:
            key: The storage key/path of the DataFrame to download.

        Returns:
            The downloaded DataFrame.
        """
        ...

    def upload_object(self, obj: Any, key: str) -> None:
        """Upload a Python object to storage.

        Args:
            obj: The Python object to upload (typically a serialized model).
            key: The storage key/path where the object should be saved.

        Returns:
            None
        """
        ...

    def upload_html(self, html: str, key: str) -> None:
        """Upload an HTML file to storage.

        Args:
            html: The file path to the HTML file to upload.
            key: The storage key/path where the HTML should be saved.

        Returns:
            None
        """
        ...
    