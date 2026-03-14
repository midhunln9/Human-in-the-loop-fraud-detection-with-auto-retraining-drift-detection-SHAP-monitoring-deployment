from fastapi import FastAPI
from app.routes.prediction_endpoints import router
from contextlib import asynccontextmanager
import wandb
import tempfile
import os
import joblib
from sklearn.base import BaseEstimator
import json
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from app.utils.logger import setup_logging
import logging

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(BASE_DIR, ".env")
load_dotenv(env_path, override=True)

def load_app_secrets(secret_name: str = "fraud_detection/api", region_name: str = "eu-north-1") -> dict:
    client = boto3.client("secretsmanager", region_name=region_name,
    aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"))
    try:
        response = client.get_secret_value(SecretId=secret_name)
        secret_str = response["SecretString"]
        os.environ["WANDB_API_KEY"] = json.loads(secret_str)["WANDB_API_KEY"]
        return json.loads(secret_str)
    except ClientError as e:
        raise RuntimeError(f"Failed to load secret {secret_name}") from e

def load_model(project_name : str, entity : str, artifact_name : str, file_name : str):
    with wandb.init(project = project_name, entity = entity) as run:
            artifact = run.use_artifact(f"{artifact_name}:production")
            with tempfile.TemporaryDirectory() as temp_dir:
                artifact.download(temp_dir)
                model = joblib.load(os.path.join(temp_dir, file_name))
    return model


@asynccontextmanager
async def on_start_up(app: FastAPI):
    setup_logging()
    logger = logging.getLogger(__name__)
    try:
        secrets = load_app_secrets()
        logger.info("successfully loaded the secrets")
        app.state.model = load_model(project_name=secrets["wandb_project"], entity=secrets["wandb_entity"], artifact_name=secrets["model_artifact_name"], file_name=secrets["model_file_name"])
        logger.info("successfully loaded the model")
        app.state.preprocessor = load_model(project_name=secrets["wandb_project"], entity=secrets["wandb_entity"], artifact_name=secrets["preprocessor_artifact_name"], file_name=secrets["preprocessor_file_name"])
        logger.info("successfully loaded the preprocessor")
        if app.state.model is None or app.state.preprocessor is None:
            raise RuntimeError("Error when loading model or preprocessor")
    except Exception as e:
        logger.error(f"Error starting or shutting down the application: {e}")
        raise RuntimeError("Error starting or shutting down the application") from e
    yield
    logger.info("successfull shut down")
    

app = FastAPI(title="Credit Card Fraud Detection API", 
description="API for credit card fraud detection",
lifespan=on_start_up)

app.include_router(router)

@app.get("/health")
async def health_check():
    return {"status": "ok", "model_loaded" : hasattr(app.state, "model") 
    and isinstance(app.state.model, BaseEstimator)}