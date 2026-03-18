import pytest
from pydantic import ValidationError

from app.schemas.real_time_prediction import RealTimePredictionRequest
from app.schemas.batch_prediction import BatchPredictionRequest


class TestRealTimePredictionRequest:
    """Unit tests for RealTimePredictionRequest schema."""

    def test_valid_request_creation(self, sample_prediction_data):
        """Test creating a valid prediction request with all required fields."""
        request = RealTimePredictionRequest(**sample_prediction_data)

        assert request.V1 == sample_prediction_data["V1"]
        assert request.V28 == sample_prediction_data["V28"]
        assert request.Amount == sample_prediction_data["Amount"]

    def test_all_fields_are_floats(self):
        """Test that all V1-V28 and Amount fields accept float values."""
        data = {f"V{i}": float(i) for i in range(1, 29)}
        data["Amount"] = 100.50

        request = RealTimePredictionRequest(**data)

        for i in range(1, 29):
            assert getattr(request, f"V{i}") == float(i)
        assert request.Amount == 100.50

    def test_negative_values_accepted(self):
        """Test that negative float values are accepted."""
        data = {f"V{i}": -float(i) for i in range(1, 29)}
        data["Amount"] = -50.25

        request = RealTimePredictionRequest(**data)

        assert request.V1 == -1.0
        assert request.Amount == -50.25

    def test_zero_values_accepted(self):
        """Test that zero values are accepted."""
        data = {f"V{i}": 0.0 for i in range(1, 29)}
        data["Amount"] = 0.0

        request = RealTimePredictionRequest(**data)

        assert request.V1 == 0.0
        assert request.Amount == 0.0

    def test_integer_values_coerced_to_float(self):
        """Test that integer values are coerced to float."""
        data = {f"V{i}": i for i in range(1, 29)}
        data["Amount"] = 100

        request = RealTimePredictionRequest(**data)

        assert isinstance(request.V1, float)
        assert isinstance(request.Amount, float)
        assert request.V1 == 1.0
        assert request.Amount == 100.0

    def test_missing_v1_field_raises_error(self):
        """Test that missing V1 field raises validation error."""
        data = {f"V{i}": float(i) for i in range(2, 29)}
        data["Amount"] = 100.0

        with pytest.raises(ValidationError) as exc_info:
            RealTimePredictionRequest(**data)

        assert "V1" in str(exc_info.value)

    def test_missing_amount_field_raises_error(self):
        """Test that missing Amount field raises validation error."""
        data = {f"V{i}": float(i) for i in range(1, 29)}

        with pytest.raises(ValidationError) as exc_info:
            RealTimePredictionRequest(**data)

        assert "Amount" in str(exc_info.value)

    def test_missing_v28_field_raises_error(self):
        """Test that missing V28 field raises validation error."""
        data = {f"V{i}": float(i) for i in range(1, 28)}
        data["Amount"] = 100.0

        with pytest.raises(ValidationError) as exc_info:
            RealTimePredictionRequest(**data)

        assert "V28" in str(exc_info.value)

    def test_model_dump_returns_dict(self, sample_prediction_data):
        """Test that model_dump() returns a dictionary with all values."""
        request = RealTimePredictionRequest(**sample_prediction_data)
        dumped = request.model_dump()

        assert isinstance(dumped, dict)
        assert len(dumped) == 29
        assert all(key in dumped for key in sample_prediction_data.keys())
        assert dumped["V1"] == sample_prediction_data["V1"]
        assert dumped["Amount"] == sample_prediction_data["Amount"]

    def test_invalid_string_value_raises_error(self):
        """Test that string values raise validation error."""
        data = {f"V{i}": float(i) for i in range(1, 29)}
        data["Amount"] = "not_a_number"

        with pytest.raises(ValidationError) as exc_info:
            RealTimePredictionRequest(**data)

        assert "Amount" in str(exc_info.value)

    def test_invalid_none_value_raises_error(self):
        """Test that None values raise validation error."""
        data = {f"V{i}": float(i) for i in range(1, 29)}
        data["Amount"] = None

        with pytest.raises(ValidationError) as exc_info:
            RealTimePredictionRequest(**data)

        assert "Amount" in str(exc_info.value)


class TestBatchPredictionRequest:
    """Unit tests for BatchPredictionRequest schema."""

    def test_valid_batch_request_creation(self, sample_batch_data):
        """Test creating a valid batch prediction request."""
        request = BatchPredictionRequest(**sample_batch_data)

        assert len(request.data) == 2
        assert isinstance(request.data[0], RealTimePredictionRequest)
        assert isinstance(request.data[1], RealTimePredictionRequest)

    def test_single_item_batch(self, sample_prediction_data):
        """Test batch request with a single item."""
        data = {"data": [sample_prediction_data]}
        request = BatchPredictionRequest(**data)

        assert len(request.data) == 1
        assert request.data[0].V1 == sample_prediction_data["V1"]

    def test_empty_batch_is_valid(self):
        """Test that empty batch is valid (schema allows empty list)."""
        data = {"data": []}

        request = BatchPredictionRequest(**data)
        assert request.data == []

    def test_missing_data_field_raises_error(self):
        """Test that missing data field raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            BatchPredictionRequest()

        assert "data" in str(exc_info.value)

    def test_invalid_item_in_batch_raises_error(self, sample_prediction_data):
        """Test that invalid item in batch raises validation error."""
        invalid_item = sample_prediction_data.copy()
        del invalid_item["V1"]

        data = {"data": [sample_prediction_data, invalid_item]}

        with pytest.raises(ValidationError) as exc_info:
            BatchPredictionRequest(**data)

    def test_model_dump_returns_dict_with_list(self, sample_batch_data):
        """Test that model_dump() returns correct structure."""
        request = BatchPredictionRequest(**sample_batch_data)
        dumped = request.model_dump()

        assert isinstance(dumped, dict)
        assert "data" in dumped
        assert isinstance(dumped["data"], list)
        assert len(dumped["data"]) == 2
        assert dumped["data"][0]["V1"] == sample_batch_data["data"][0]["V1"]
