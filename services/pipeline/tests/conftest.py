import pytest
from mlops_pipeline.protocols.storage_protocol import StorageProtocol
from tests.fakes.fake_storage import FakeStorageRepository
from mlops_pipeline.configs.s3_storage_config import S3StorageConfig
from mlops_pipeline.settings import Settings

# storage Fixtures
@pytest.fixture
def fake_storage() -> StorageProtocol:
    return FakeStorageRepository()

# s3 config Fixtures
@pytest.fixture
def s3_config() -> S3StorageConfig:
    return S3StorageConfig()

# settings Fixtures
@pytest.fixture
def settings() -> Settings:
    return Settings()
