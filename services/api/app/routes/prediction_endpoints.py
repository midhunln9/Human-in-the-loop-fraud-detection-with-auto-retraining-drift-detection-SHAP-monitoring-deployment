from fastapi import APIRouter, Request, HTTPException
from app.schemas.real_time_prediction import RealTimePredictionRequest
from app.schemas.batch_prediction import BatchPredictionRequest
import pandas as pd 
import logging
import boto3
from app.sns_publish import SNSPublish
import os
import uuid

router = APIRouter()

logger = logging.getLogger(__name__)

sns_publish = SNSPublish(aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"), 
aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"), 
region_name=os.getenv("AWS_REGION"),
aws_sns_arn=os.getenv("AWS_SNS_ARN"))

def create_sns_message_format(message : dict):
    event = {}
    for x,y in message.items():
        if x == "features":
            j = 0
            for i in y:
                event[f"V{j}"] = i
                j+=1
        elif x == "prediction":
            event["prediction"] = y
        else:
            event["transaction_id"] = y
    return event



@router.post("/predict")
async def predict(real_time_prediction_request: RealTimePredictionRequest, request: Request):
    logger.info("payload received for real time prediction")
    model = request.app.state.model
    if model is None:
        logger.error("model not loaded")
        raise HTTPException(status_code=500, detail="Model not loaded")
    transaction_id =str(uuid.uuid4())
    df = pd.DataFrame([real_time_prediction_request.model_dump()])
    df_preprocessed = request.app.state.preprocessor.transform(df)
    logger.info("data preprocessed successfully")
    prediction = model.predict(df_preprocessed)
    logger.info("prediction made successfully")
    probability = model.predict_proba(df_preprocessed)[:,1]

    features = list(real_time_prediction_request.model_dump().values())
    pred_label = int(prediction[0])

    payload = {
        "transaction_id" : transaction_id,
        "features": features,
        "prediction": pred_label
    }

    refined_payload = create_sns_message_format(payload)

    sns_publish.publish(message = refined_payload)

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