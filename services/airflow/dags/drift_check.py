from __future__ import annotations

import io
import logging
from datetime import datetime

import pandas as pd
from airflow import DAG
from airflow.decorators import task
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.providers.amazon.aws.operators.ecs import EcsRunTaskOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.providers.standard.decorators.short_circuit import short_circuit_task
from airflow.operators.empty import EmptyOperator

from evidently import Report
from evidently.presets import DataDriftPreset


AWS_CONN_ID = "aws_default"
POSTGRES_CONN_ID = "rds_postgres_default"

S3_BUCKET = "your-training-data-bucket"
S3_KEY = "training.csv"

ECS_CLUSTER = "your-ecs-cluster"
ECS_TASK_DEFINITION = "your-ecs-task-definition"
ECS_CONTAINER_NAME = "your-container-name"
AWS_REGION = "ap-south-1"

SUBNETS = ["subnet-xxxxxxxx", "subnet-yyyyyyyy"]
SECURITY_GROUPS = ["sg-xxxxxxxx"]

DRIFT_SHARE_THRESHOLD = 0.5  


def _load_training_data_from_s3(bucket: str, key: str, aws_conn_id: str) -> pd.DataFrame:
    """Read training.csv from S3 into a pandas DataFrame."""
    s3_hook = S3Hook(aws_conn_id=aws_conn_id)
    csv_text = s3_hook.read_key(key=key, bucket_name=bucket)
    if not csv_text:
        raise ValueError(f"S3 object s3://{bucket}/{key} is empty or not readable.")
    return pd.read_csv(io.StringIO(csv_text))


def _load_production_data_from_rds(postgres_conn_id: str) -> pd.DataFrame:
    """Read all rows from the transactions table into a pandas DataFrame."""
    pg_hook = PostgresHook(postgres_conn_id=postgres_conn_id)
    query = "SELECT * FROM transactions;"
    df = pg_hook.get_df(sql=query, df_type="pandas")
    if df.empty:
        raise ValueError("RDS query returned zero rows from transactions.")
    return df


def _prepare_common_feature_frames(
    train_df: pd.DataFrame,
    prod_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Keep only common columns and align order.
    Optionally drop columns that are entirely null in either dataset.
    """
    common_cols = [c for c in train_df.columns if c in prod_df.columns]
    if not common_cols:
        raise ValueError("No common columns found between training.csv and transactions table.")

    ref = train_df[common_cols].copy()
    cur = prod_df[common_cols].copy()

    # Drop columns that are fully null in either dataset
    valid_cols = [
        c for c in common_cols
        if not ref[c].isna().all() and not cur[c].isna().all()
    ]
    if not valid_cols:
        raise ValueError("All common columns are empty in at least one dataset.")

    ref = ref[valid_cols]
    cur = cur[valid_cols]

    return ref, cur


def _detect_dataset_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    drift_share_threshold: float = 0.5,
) -> bool:
    """
    Run Evidently data drift and return a single boolean.
    True  -> drift detected
    False -> no drift
    """
    report = Report(
        metrics=[
            DataDriftPreset(drift_share=drift_share_threshold)
        ]
    )

    evaluation = report.run(
        current_data=current_df,
        reference_data=reference_df,
    )

    result = evaluation.dict()

    dataset_drift = None
    drifted_columns = None
    drift_share = None

    for metric in result.get("metrics", []):
        metric_result = metric.get("result", {})
        if "dataset_drift" in metric_result:
            dataset_drift = metric_result.get("dataset_drift")
            drifted_columns = metric_result.get("number_of_drifted_columns")
            drift_share = metric_result.get("share_of_drifted_columns")
            break

    if dataset_drift is None:
        raise ValueError("Could not extract dataset_drift from Evidently output.")

    logging.info(
        "Evidently result | dataset_drift=%s | drifted_columns=%s | drift_share=%s",
        dataset_drift,
        drifted_columns,
        drift_share,
    )

    return bool(dataset_drift)


with DAG(
    dag_id="data_drift_check_and_trigger_ecs",
    start_date=datetime(2025, 1, 1),
    schedule="@daily",
    catchup=False,
    tags=["ml", "drift", "evidently", "ecs"],
    doc_md="""
    DAG flow:
    1. Read training.csv from S3
    2. Read all rows from RDS transactions table
    3. Run Evidently data drift check
    4. If drift detected -> trigger ECS task
    5. If no drift -> stop
    """,
) as dag:

    start = EmptyOperator(task_id="start")

    @task(task_id="check_data_drift")
    def check_data_drift() -> bool:
        train_df = _load_training_data_from_s3(
            bucket=S3_BUCKET,
            key=S3_KEY,
            aws_conn_id=AWS_CONN_ID,
        )

        prod_df = _load_production_data_from_rds(
            postgres_conn_id=POSTGRES_CONN_ID,
        )

        reference_df, current_df = _prepare_common_feature_frames(train_df, prod_df)

        is_drift = _detect_dataset_drift(
            reference_df=reference_df,
            current_df=current_df,
            drift_share_threshold=DRIFT_SHARE_THRESHOLD,
        )

        logging.info("Final drift decision: %s", is_drift)
        return is_drift

    @short_circuit_task(task_id="proceed_only_if_drift")
    def proceed_only_if_drift(is_drift: bool) -> bool:
        """
        If False, downstream tasks are skipped.
        If True, downstream tasks execute.
        """
        return is_drift

    trigger_ecs_pipeline = EcsRunTaskOperator(
        task_id="trigger_ecs_pipeline",
        aws_conn_id=AWS_CONN_ID,
        region_name=AWS_REGION,
        cluster=ECS_CLUSTER,
        task_definition=ECS_TASK_DEFINITION,
        launch_type="FARGATE",
        overrides={
            "containerOverrides": [
                {
                    "name": ECS_CONTAINER_NAME,
                    # Replace with the actual command your pipeline container expects
                    "command": ["python", "pipeline.py"],
                }
            ]
        },
        network_configuration={
            "awsvpcConfiguration": {
                "subnets": SUBNETS,
                "securityGroups": SECURITY_GROUPS,
                "assignPublicIp": "ENABLED",
            }
        },
    )

    end = EmptyOperator(task_id="end")

    drift_result = check_data_drift()
    gate = proceed_only_if_drift(drift_result)

    start >> drift_result >> gate >> trigger_ecs_pipeline >> end