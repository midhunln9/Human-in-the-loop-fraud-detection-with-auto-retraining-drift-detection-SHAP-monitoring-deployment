from pydantic import BaseModel
from app.schemas.real_time_prediction import RealTimePredictionRequest
from typing import List

class BatchPredictionRequest(BaseModel):
    data: List[RealTimePredictionRequest]
