from fastapi import APIRouter, Request, HTTPException
from app.schemas.real_time_prediction import RealTimePredictionRequest
from app.schemas.batch_prediction import BatchPredictionRequest
import pandas as pd 
import logging
router = APIRouter()

logger = logging.getLogger(__name__)

@router.post("/predict")
async def predict(real_time_prediction_request: RealTimePredictionRequest, request: Request):
    logger.info("payload received for real time prediction")
    model = request.app.state.model
    if model is None:
        logger.error("model not loaded")
        raise HTTPException(status_code=500, detail="Model not loaded")
    df = pd.DataFrame([real_time_prediction_request.model_dump()])
    df_preprocessed = request.app.state.preprocessor.transform(df)
    logger.info("data preprocessed successfully")
    prediction = model.predict(df_preprocessed)
    logger.info("prediction made successfully")
    probability = model.predict_proba(df_preprocessed)[:,1]
    logger.info("probability calculated successfully")
    return {"prediction": int(prediction[0]), "probability": float(probability[0])}

@router.post("/batch-predict")
async def batch_predict(batch_prediction_request: BatchPredictionRequest, request: Request):
    logger.info("payload received for batch prediction")
    model = request.app.state.model
    if model is None:
        logger.error("model not loaded")
        raise HTTPException(status_code=500, detail="Model not loaded")
    data = []
    for row in batch_prediction_request.data:
        data.append(row.model_dump())
    df = pd.DataFrame(data)
    df_preprocessed = request.app.state.preprocessor.transform(df)
    logger.info("data preprocessed successfully")
    predictions = model.predict(df_preprocessed)
    logger.info("predictions made successfully")
    probability = model.predict_proba(df_preprocessed)[:,1]
    logger.info("probabilities calculated successfully")
    return {"predictions": predictions.tolist(), "probability": probability.tolist()}