import json
import pytest
from unittest.mock import MagicMock, patch

from app.sns_publish import SNSPublish


class TestSNSPublish:
    """Unit tests for SNSPublish class."""

    @pytest.fixture
    def sns_config(self):
        """Fixture providing SNS configuration."""
        return {
            "aws_access_key_id": "test-access-key",
            "aws_secret_access_key": "test-secret-key",
            "region_name": "us-east-1",
            "aws_sns_arn": "arn:aws:sns:us-east-1:123456789:test-topic",
        }

    def test_initialization_stores_credentials(self, sns_config):
        """Test that SNSPublish stores all credentials on initialization."""
        with patch("boto3.client") as mock_boto_client:
            mock_sns = MagicMock()
            mock_boto_client.return_value = mock_sns

            publisher = SNSPublish(**sns_config)

            assert publisher.aws_access_key_id == sns_config["aws_access_key_id"]
            assert publisher.aws_secret_access_key == sns_config["aws_secret_access_key"]
            assert publisher.region_name == sns_config["region_name"]
            assert publisher.aws_sns_arn == sns_config["aws_sns_arn"]

    def test_initialization_creates_sns_client(self, sns_config):
        """Test that SNSPublish creates boto3 SNS client on initialization."""
        with patch("boto3.client") as mock_boto_client:
            mock_sns = MagicMock()
            mock_boto_client.return_value = mock_sns

            publisher = SNSPublish(**sns_config)

            mock_boto_client.assert_called_once_with(
                "sns", region_name=sns_config["region_name"]
            )
            assert publisher.sns_client == mock_sns

    def test_publish_sends_message_to_correct_topic(self, sns_config):
        """Test that publish sends message to the correct SNS topic ARN."""
        with patch("boto3.client") as mock_boto_client:
            mock_sns = MagicMock()
            mock_boto_client.return_value = mock_sns

            publisher = SNSPublish(**sns_config)
            message = {"features": [1.0, 2.0, 3.0], "prediction": 1}
            publisher.publish(message)

            mock_sns.publish.assert_called_once()
            call_args = mock_sns.publish.call_args
            assert call_args.kwargs["TopicArn"] == sns_config["aws_sns_arn"]

    def test_publish_serializes_message_to_json(self, sns_config):
        """Test that publish serializes the message dictionary to JSON."""
        with patch("boto3.client") as mock_boto_client:
            mock_sns = MagicMock()
            mock_boto_client.return_value = mock_sns

            publisher = SNSPublish(**sns_config)
            message = {"features": [1.0, 2.0, 3.0], "prediction": 1}
            publisher.publish(message)

            call_args = mock_sns.publish.call_args
            published_message = call_args.kwargs["Message"]

            assert isinstance(published_message, str)
            deserialized = json.loads(published_message)
            assert deserialized == message

    def test_publish_with_nested_dict(self, sns_config):
        """Test that publish handles nested dictionary structures."""
        with patch("boto3.client") as mock_boto_client:
            mock_sns = MagicMock()
            mock_boto_client.return_value = mock_sns

            publisher = SNSPublish(**sns_config)
            message = {
                "features": [1.0, 2.0],
                "prediction": 1,
                "metadata": {"timestamp": "2024-01-01", "model_version": "v1"},
            }
            publisher.publish(message)

            call_args = mock_sns.publish.call_args
            published_message = json.loads(call_args.kwargs["Message"])

            assert published_message["metadata"]["timestamp"] == "2024-01-01"
            assert published_message["metadata"]["model_version"] == "v1"

    def test_publish_with_empty_message(self, sns_config):
        """Test that publish handles empty dictionary."""
        with patch("boto3.client") as mock_boto_client:
            mock_sns = MagicMock()
            mock_boto_client.return_value = mock_sns

            publisher = SNSPublish(**sns_config)
            publisher.publish({})

            call_args = mock_sns.publish.call_args
            published_message = call_args.kwargs["Message"]

            assert published_message == "{}"
            assert json.loads(published_message) == {}

    def test_publish_with_list_in_message(self, sns_config):
        """Test that publish handles lists in message."""
        with patch("boto3.client") as mock_boto_client:
            mock_sns = MagicMock()
            mock_boto_client.return_value = mock_sns

            publisher = SNSPublish(**sns_config)
            message = {"features": list(range(29)), "prediction": 0}
            publisher.publish(message)

            call_args = mock_sns.publish.call_args
            published_message = json.loads(call_args.kwargs["Message"])

            assert published_message["features"] == list(range(29))
            assert len(published_message["features"]) == 29

    def test_publish_with_various_data_types(self, sns_config):
        """Test that publish handles various data types."""
        with patch("boto3.client") as mock_boto_client:
            mock_sns = MagicMock()
            mock_boto_client.return_value = mock_sns

            publisher = SNSPublish(**sns_config)
            message = {
                "string_field": "test",
                "int_field": 42,
                "float_field": 3.14,
                "bool_field": True,
                "null_field": None,
                "list_field": [1, 2, 3],
            }
            publisher.publish(message)

            call_args = mock_sns.publish.call_args
            published_message = json.loads(call_args.kwargs["Message"])

            assert published_message["string_field"] == "test"
            assert published_message["int_field"] == 42
            assert published_message["float_field"] == 3.14
            assert published_message["bool_field"] is True
            assert published_message["null_field"] is None
            assert published_message["list_field"] == [1, 2, 3]

    def test_multiple_publish_calls(self, sns_config):
        """Test that multiple publish calls work correctly."""
        with patch("boto3.client") as mock_boto_client:
            mock_sns = MagicMock()
            mock_boto_client.return_value = mock_sns

            publisher = SNSPublish(**sns_config)

            message1 = {"prediction": 1}
            message2 = {"prediction": 0}

            publisher.publish(message1)
            publisher.publish(message2)

            assert mock_sns.publish.call_count == 2

            call1_args = mock_sns.publish.call_args_list[0].kwargs
            call2_args = mock_sns.publish.call_args_list[1].kwargs

            assert json.loads(call1_args["Message"]) == message1
            assert json.loads(call2_args["Message"]) == message2
