import json
import os
from typing import Any, Dict, List

from sqlalchemy import create_engine, MetaData, Table, Column
from sqlalchemy import String, Float, Integer
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.dialects.postgresql import insert


# env variables
DB_HOST = os.environ["DB_HOST"]
DB_PORT = os.environ.get("DB_PORT", "5432")
DB_NAME = os.environ["DB_NAME"]
DB_USER = os.environ["DB_USER"]
DB_PASSWORD = os.environ["DB_PASSWORD"]
DB_TABLE = os.environ.get("DB_TABLE", "transactions")

DATABASE_URL = (
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

# SQL ALCHEMY setup
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=300,
)

metadata = MetaData()

table = Table(
    DB_TABLE,
    metadata,
    Column("transaction_id", String, primary_key=True),
    *(Column(f"V{i}", Float, nullable=False, default=0.0) for i in range(29)),
    Column("prediction", Integer, nullable=False, default=0),
)

_table_initialized = False


def ensure_table_exists() -> None:
    global _table_initialized
    if not _table_initialized:
        metadata.create_all(engine)
        _table_initialized = True


def normalize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and normalize one incoming message dict.
    Ensures all V0..V28 exist and are numeric.
    """
    if "transaction_id" not in row:
        raise ValueError("Missing required field: transaction_id")

    normalized = {
        "transaction_id": str(row["transaction_id"]),
        "prediction": int(row.get("prediction", 0)),
    }

    for i in range(29):
        normalized[f"V{i}"] = float(row.get(f"V{i}", 0.0))

    return normalized


def parse_sqs_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse one SQS record body into a normalized row dict.
    Expected body is JSON string like:
    {
      "transaction_id": "...",
      "V0": 0.0,
      ...
      "V28": 0.0,
      "prediction": 0
    }
    """
    body = record.get("body", "")
    if not body:
        raise ValueError("Empty SQS message body")

    payload = json.loads(body)

    if not isinstance(payload, dict):
        raise ValueError("SQS message body must be a JSON object")

    return normalize_row(payload)


def lambda_handler(event, context):
    """
    SQS-triggered Lambda handler.

    Handles:
    - single or batched SQS messages
    - duplicate delivery using ON CONFLICT DO NOTHING
    - partial batch failures
    """
    ensure_table_exists()

    failed_items: List[Dict[str, str]] = []
    rows_to_insert: List[Dict[str, Any]] = []

    # Keep track of which SQS message produced which row
    # so if parsing fails we can mark only that item failed.
    valid_message_ids: List[str] = []

    records = event.get("Records", [])

    for record in records:
        message_id = record.get("messageId", "")
        try:
            row = parse_sqs_record(record)
            rows_to_insert.append(row)
            valid_message_ids.append(message_id)
        except Exception as exc:
            print(f"Failed to parse message {message_id}: {str(exc)}")
            failed_items.append({"itemIdentifier": message_id})

    # Insert valid rows in one DB call for efficiency
    if rows_to_insert:
        try:
            stmt = insert(table).values(rows_to_insert)
            stmt = stmt.on_conflict_do_nothing(index_elements=["transaction_id"])

            with engine.begin() as conn:
                conn.execute(stmt)

        except SQLAlchemyError as exc:
            # If the DB insert fails, mark all valid messages in this batch as failed
            # so SQS can retry them.
            print(f"Database insert failed: {str(exc)}")
            for message_id in valid_message_ids:
                failed_items.append({"itemIdentifier": message_id})

    return {
        "batchItemFailures": failed_items
    }