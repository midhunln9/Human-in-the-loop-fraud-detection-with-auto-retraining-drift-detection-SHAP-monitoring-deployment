# Credit Card Fraud Detection — Technical Design Document

> End-to-End MLOps System with Automated Retraining, Drift Detection, Human-in-the-Loop Feedback, and AWS Deployment

---

## 1. Executive Summary

This project is a **production-grade fraud detection system** that goes far beyond model training. It covers the complete MLOps lifecycle — from data ingestion and hyperparameter optimization to real-time serving, human feedback collection, data drift monitoring, and automated retraining — deployed as independently containerized microservices on AWS.

**At a glance:**

| Metric | Value |
|---|---|
| Dataset | 284,807 transactions (Kaggle Credit Card Fraud) |
| Fraud rate | 0.17% (492 fraudulent out of 284,807) |
| Model PR-AUC | 0.87 (test set) |
| Inference latency | ~40ms per prediction |
| Hyperparameter trials | 200 (100 XGBoost + 100 LightGBM via Optuna) |
| Automated tests | 84 (71 API + 13 pipeline) |
| Services | 6 independently containerized microservices |
| AWS services used | 9 (ECS, Lambda, S3, SNS, SQS, RDS, Secrets Manager, ALB, ECR) |

---

## 2. Problem Statement

Credit card fraud costs the global economy **$30B+ annually**. Traditional rule-based systems fail to keep up with evolving fraud patterns. ML models improve detection, but deploying them introduces new challenges:

- **Model decay** — fraud patterns shift over time, degrading model performance silently
- **False positives** — blocking legitimate transactions damages customer trust and revenue
- **Latency requirements** — fraud decisions must happen in real-time, not batch
- **Accountability** — financial regulations require auditability of every decision

This system addresses all four by combining **real-time ML inference**, **human-in-the-loop validation**, **automated drift detection**, and **full audit trails**.

---

## 3. System Architecture

### 3.1 High-Level Overview

The system operates in two interconnected loops:

**Loop 1 — Real-Time Prediction and Feedback**
```
Transaction → FastAPI → Model Prediction → SNS Topic
                                              ├── SQS → Lambda → PostgreSQL (production data store)
                                              └── SSE Server → Dashboard → Human Reviewer
                                                                              ├── "Correct" → dismiss
                                                                              └── "Not Correct" → wrong_predictions table
```

**Loop 2 — Automated Monitoring and Retraining**
```
Airflow DAG (daily) → Load training data (S3) + production data (PostgreSQL)
                    → Evidently drift check
                    → If drift detected → Trigger ECS Fargate task
                                        → Run 9-stage ML pipeline
                                        → Compare with production model
                                        → Promote only if better
```

### 3.2 Data Flow Diagram

```
┌─────────────┐    POST /predict    ┌──────────────────┐
│   Client    │ ──────────────────► │  FastAPI Service  │
│ (Transaction│                     │  (ECS Fargate)    │
│   System)   │ ◄────────────────── │                   │
└─────────────┘  prediction + prob  │  - Preprocesses   │
                                    │  - Predicts       │
                                    │  - Publishes SNS  │
                                    └────────┬──────────┘
                                             │
                                    ┌────────▼──────────┐
                                    │   AWS SNS Topic   │
                                    │   (Fan-out)       │
                                    └──┬─────────────┬──┘
                                       │             │
                              ┌────────▼───┐   ┌────▼───────────┐
                              │  SQS Queue │   │  SSE Server    │
                              └────────┬───┘   │  (ECS Fargate) │
                                       │       └────┬───────────┘
                              ┌────────▼───┐        │ Server-Sent Events
                              │ AWS Lambda │   ┌────▼───────────┐
                              │ (DB Writer)│   │  Dashboard     │
                              └────────┬───┘   │  (Browser)     │
                                       │       └────┬───────────┘
                              ┌────────▼───┐        │ Human Feedback
                              │ PostgreSQL │   ┌────▼───────────┐
                              │   (RDS)    │   │ Wrong Preds DB │
                              │ transactions│  │ (feedback loop)│
                              └────────────┘   └────────────────┘
```

### 3.3 Why This Architecture?

| Decision | Rationale |
|---|---|
| **SNS fan-out** | Decouples API from downstream consumers. Lambda and SSE server scale independently. Adding a new consumer (e.g., alerting service) requires zero API changes. |
| **SQS between SNS and Lambda** | Provides buffering, retry with backoff, and dead-letter queue support. Handles traffic spikes without dropping predictions. |
| **SSE over WebSocket** | Unidirectional data flow (server → client) matches the use case. SSE is simpler, auto-reconnects, and works through proxies without upgrade headers. |
| **Separate services on ECS** | Each service can be scaled, deployed, and monitored independently. A bug in the SSE server does not affect prediction serving. |

---

## 4. ML Pipeline — Deep Dive

### 4.1 Pipeline Stages

The pipeline runs as a single orchestrated flow via `PipelineRunner`, producing a timestamped audit trail in S3 for every run.

```
Stage 0: Data Ingestion        → Read raw CSV from S3
Stage 1: Data Transformation   → Stratified train/val/test split (60/20/20)
Stage 2: Data Preprocessing    → ColumnTransformer (SimpleImputer + StandardScaler)
Stage 3: Data Upload           → Upload preprocessed splits to S3
Stage 4: Hyperparameter Tuning → 100 Optuna trials × 2 model types
Stage 5: Model Training        → Retrain best model on train+val combined
Stage 6: Model Upload          → Serialize to S3 via joblib
Stage 7: Model Evaluation      → PR-AUC + Evidently ClassificationPreset report
Stage 8: Model Promotion       → Compare staging vs production → promote if better
```

**Total pipeline runtime:** ~12.5 minutes (756 seconds measured end-to-end)

### 4.2 Hyperparameter Optimization

Two model strategies compete head-to-head:

**XGBoost — tuned parameters:**
| Parameter | Search Range |
|---|---|
| max_depth | 3–10 |
| learning_rate | 0.01–0.3 |
| subsample | 0.5–1.0 |
| colsample_bytree | 0.5–1.0 |
| gamma | 0–5 |
| reg_alpha | 1e-8–10 |
| reg_lambda | 1e-8–10 |
| min_child_weight | 1–10 |

**LightGBM — tuned parameters:**
| Parameter | Search Range |
|---|---|
| max_depth | 3–10 |
| learning_rate | 0.01–0.3 |
| num_leaves | 20–150 |
| subsample | 0.5–1.0 |
| colsample_bytree | 0.5–1.0 |
| min_child_samples | 5–100 |
| reg_alpha | 1e-8–10 |
| reg_lambda | 1e-8–10 |

**Results from actual pipeline run:**

| Model | Best Optuna PR-AUC | Tuning Time |
|---|---|---|
| XGBoost | **0.9178** | 3m 56s |
| LightGBM | 0.9124 | 7m 20s |

XGBoost was selected as the best strategy. After retraining on combined train+validation data, the staging model achieved **0.9979 PR-AUC** on the combined set and **0.8661 PR-AUC** on the held-out test set.

### 4.3 Class Imbalance Handling

With only 0.17% positive class, standard accuracy is meaningless (a "predict all legitimate" model gets 99.83% accuracy).

**Strategies used:**
- **PR-AUC as primary metric** — measures the trade-off between precision and recall, which is critical for imbalanced datasets
- **Automatic scale_pos_weight** — calculated as `count(negatives) / count(positives)` = ~577, telling the model to penalize misclassifying fraud 577x more
- **Early stopping** — 50 rounds of patience to prevent overfitting on the minority class

### 4.4 Model Promotion Logic

```
if no production model exists:
    promote staging → production

elif staging_test_prauc > production_prauc:
    promote staging model → production
    promote staging preprocessor → production

else:
    keep current production model (no action)
```

This ensures the production model **never degrades** — a new model must demonstrably outperform the incumbent.

### 4.5 S3 Audit Trail

Every pipeline run produces a complete snapshot:

```
pipeline_2026_04_14_00_51_12/
├── data/
│   ├── preprocessed_train_data.csv        # Training features
│   ├── preprocessed_train_labels.csv      # Training labels
│   ├── preprocessed_validation_data.csv   # Validation features
│   ├── preprocessed_validation_labels.csv # Validation labels
│   ├── preprocessed_test_data.csv         # Test features
│   └── preprocessed_test_labels.csv       # Test labels
├── models/
│   └── model.joblib                       # Serialized trained model
└── reports/
    └── eval_report.html                   # Evidently ClassificationPreset
```

Any historical run can be fully inspected and reproduced.

---

## 5. Fraud Detection API — Deep Dive

### 5.1 Startup Sequence

```
1. Load secrets from AWS Secrets Manager (W&B keys, artifact names)
2. Initialize W&B run
3. Download production model artifact → deserialize with joblib
4. Download production preprocessor artifact → deserialize with joblib
5. Store both in app.state for request-time access
6. Start accepting requests
```

### 5.2 Prediction Flow (POST /predict)

```
1. Validate input (29 float features: V1–V28 + Amount)
2. Generate UUID for transaction_id
3. Preprocess features using production preprocessor
4. Run model.predict() → binary prediction (0 or 1)
5. Run model.predict_proba() → fraud probability
6. Construct SNS message (transaction_id + features + prediction)
7. Publish to SNS topic (async, non-blocking to response)
8. Return {prediction, probability} to client
```

**Measured latency breakdown:**
| Step | Time |
|---|---|
| Preprocessing | ~5ms |
| Prediction | ~2ms |
| Probability calculation | ~400ms |
| SNS publish | ~400ms |
| **Total (incl. SNS)** | **~40ms response** (SNS async) |

### 5.3 Endpoints

| Endpoint | Method | Description | SNS Publish |
|---|---|---|---|
| `/predict` | POST | Single real-time prediction | Yes |
| `/batch-predict` | POST | Batch inference (list of transactions) | No |
| `/health` | GET | Returns model loaded status | — |

---

## 6. Human-in-the-Loop Feedback System

### 6.1 Why Human-in-the-Loop?

In fraud detection, false positives (legitimate transactions flagged as fraud) directly impact customer experience. Automated models alone cannot achieve 100% precision. The human feedback loop:

1. **Catches false positives** before they harm customers
2. **Generates labeled data** from production traffic (no manual annotation needed)
3. **Feeds into retraining** — wrong predictions become training signal for the next model

### 6.2 Dashboard Flow

```
SSE Server broadcasts prediction ──► Dashboard renders transaction card
                                         │
                                    ┌────┴────┐
                                    │         │
                              "Correct"   "Not Correct"
                                    │         │
                              Card dismissed  Payload + actual label
                                              written to wrong_predictions
                                              table in PostgreSQL
```

### 6.3 Feedback Data Utilization

When the `wrong_predictions` table is joined with the `transactions` table (via `transaction_id`), we get a dataset of all model mistakes with corrected labels — a high-signal training set for the next retraining cycle.

---

## 7. Drift Detection and Auto-Retraining

### 7.1 Why Drift Detection?

Fraud patterns evolve. A model trained on historical data degrades as:
- **Data drift** — input feature distributions shift (e.g., transaction amounts change seasonally)
- **Concept drift** — the relationship between features and fraud changes (e.g., new fraud tactics)

Without monitoring, model performance silently degrades until a business metric (e.g., fraud loss) spikes.

### 7.2 Implementation

**Airflow DAG runs daily:**

```python
# Pseudocode
training_data = load_from_s3("training.csv")
production_data = load_from_postgresql("transactions")

drift_report = evidently.DataDriftPreset(training_data, production_data)

if drift_report.share_of_drifted_columns > 0.50:  # 50% threshold
    trigger_ecs_pipeline_task()
else:
    no_action()
```

**DAG task flow:**
```
start → check_data_drift → proceed_only_if_drift (short-circuit) → trigger_ecs_pipeline → end
```

### 7.3 Retraining Trigger Chain

```
Drift detected → Airflow triggers ECS Fargate task
              → Pipeline ingests latest data
              → Runs full 9-stage pipeline
              → New model compared against production
              → Promoted only if PR-AUC improves
              → API picks up new model on next deployment
```

This creates a **closed-loop system**: predictions generate production data → drift is monitored → retraining is triggered → better model is deployed → cycle repeats.

---

## 8. Software Engineering Practices

### 8.1 Protocol-Based Architecture

The pipeline uses Python protocols (structural subtyping) for dependency inversion:

```
StorageProtocol          → implemented by S3Storage
ModelVersioningProtocol  → implemented by WandbRepository
BaseModelStrategy (ABC)  → implemented by XGBoostStrategy, LightGBMStrategy
```

**Benefits:**
- Swap S3 for local filesystem or GCS without changing pipeline code
- Swap W&B for MLflow without touching training logic
- Add new model types by implementing one class — zero changes to `MasterTuner` or `PipelineRunner`
- Test with fakes/mocks (see `tests/fakes/fake_storage.py`)

### 8.2 Exception Hierarchy

```
PipelineError (base)
├── IngestionError      # S3 data download failures
├── TransformationError # Train/test split failures
├── PreprocessingError  # Scaling/imputation failures
├── TuningError         # Optuna optimization failures
├── TrainingError       # Model fitting failures
├── EvaluationError     # Metric calculation failures
├── PromotionError      # Model promotion failures
├── StorageError        # S3 operations failures
└── ArtifactError       # W&B operations failures
```

Each exception type enables targeted error handling — a storage failure is handled differently from a tuning failure.

### 8.3 Testing Strategy

| Layer | What's Tested | Count |
|---|---|---|
| **Unit** | Schema validation, SNS publish, model strategies, preprocessing, transformations | 39 |
| **Integration** | Full API endpoints (predict, batch, health), startup sequence, error handling | 45 |
| **Load** | Concurrent user simulation via Locust | Config |
| **Total** | | **84** |

Tests use mock models, fake storage implementations, and mocked AWS clients to run without cloud dependencies.

### 8.4 CI/CD Pipeline

```
GitHub Actions (.github/workflows/deploy-to-ecr.yml)

Push/PR to main
├── test-api (pytest -v)
│   └── ✓ pass → build-and-push-api (Docker → ECR)
└── test-pipeline (pytest -v)
    └── ✓ pass → build-and-push-pipeline (Docker → ECR)

Push gate: images pushed to ECR only on merge to main
Image tags: git SHA + latest
```

### 8.5 Containerization

All services use `python:3.12-slim` base images with multi-stage-aware layer caching:

```dockerfile
# Example: Pipeline Dockerfile
COPY uv.lock pyproject.toml ./     # Dependency cache layer
RUN uv sync --no-install-project   # Install deps (cached if lockfile unchanged)
COPY . .                           # Copy source code
RUN uv sync                        # Install project
```

---

## 9. AWS Infrastructure Map

```
┌─────────────────────────────────────────────────────────┐
│                        AWS Cloud                         │
│                                                         │
│  ┌──────────────┐   ┌──────────────┐   ┌────────────┐ │
│  │  ECS Fargate  │   │  ECS Fargate  │   │ ECS Fargate│ │
│  │  (API)        │   │  (SSE Server) │   │ (Pipeline) │ │
│  └──────┬───────┘   └──────┬───────┘   └────────────┘ │
│         │                   │                           │
│  ┌──────▼───────────────────▼──────┐                   │
│  │          SNS Topic              │                   │
│  └──────┬──────────────────────────┘                   │
│         │                                               │
│  ┌──────▼───────┐                                      │
│  │  SQS Queue   │                                      │
│  └──────┬───────┘                                      │
│         │                                               │
│  ┌──────▼───────┐   ┌──────────────┐   ┌────────────┐ │
│  │  Lambda       │   │  RDS         │   │    S3      │ │
│  │  (DB Writer)  │──►│  PostgreSQL  │   │  (Storage) │ │
│  └──────────────┘   └──────────────┘   └────────────┘ │
│                                                         │
│  ┌──────────────┐   ┌──────────────┐   ┌────────────┐ │
│  │  Secrets Mgr  │   │     ALB      │   │    ECR     │ │
│  └──────────────┘   └──────────────┘   └────────────┘ │
└─────────────────────────────────────────────────────────┘
```

| Service | Purpose |
|---|---|
| **ECS Fargate** | Runs API, SSE server, and pipeline as serverless containers |
| **S3** | Stores raw data, preprocessed splits, models, evaluation reports |
| **SNS** | Fan-out predictions to SQS and SSE server |
| **SQS** | Buffers messages between SNS and Lambda with retry support |
| **Lambda** | Writes predictions to PostgreSQL (SQS-triggered) |
| **RDS PostgreSQL** | Stores production predictions and human feedback |
| **Secrets Manager** | Stores W&B API keys and artifact configuration |
| **ALB** | Routes traffic to ECS services |
| **ECR** | Hosts Docker images built by CI/CD |

---

## 10. Key Design Trade-offs

| Trade-off | Decision | Reasoning |
|---|---|---|
| **PR-AUC vs F1** | PR-AUC as primary metric | With 0.17% fraud rate, ROC-AUC is misleadingly high. PR-AUC better reflects performance on the minority class. |
| **XGBoost vs Neural Network** | Tree-based models | Tabular PCA features with 29 columns. Tree models outperform neural networks on structured tabular data this size. Faster training, better interpretability. |
| **SNS+SQS vs direct DB write** | Event-driven fan-out | Decouples API from persistence. API response time unaffected by DB latency. New consumers added without API changes. |
| **Daily drift vs real-time** | Daily Airflow DAG | Drift is a gradual phenomenon. Daily checks are sufficient and far simpler than streaming drift detection. |
| **W&B vs MLflow** | Weights & Biases | Cloud-hosted, no infrastructure to manage. Built-in artifact versioning with aliases (staging/production) maps directly to promotion workflow. |
| **Optuna vs Grid Search** | Optuna (TPE sampler) | 200 trials with Bayesian optimization explores the hyperparameter space far more efficiently than grid search. |

---

## 11. Full Tech Stack Reference

| Layer | Technology | Version |
|---|---|---|
| Language | Python | 3.12 |
| ML Framework | XGBoost, LightGBM | 3.2+, 4.6+ |
| ML Utilities | scikit-learn | 1.8+ |
| HPO | Optuna | 4.7+ |
| API | FastAPI, Pydantic, uvicorn | Latest |
| Experiment Tracking | Weights & Biases | 0.25+ |
| Drift Detection | Evidently | 0.7+ |
| Database | PostgreSQL (via SQLAlchemy) | — |
| Orchestration | Apache Airflow | — |
| Containerization | Docker | python:3.12-slim |
| Package Manager | uv | Latest |
| CI/CD | GitHub Actions | — |
| Load Testing | Locust | 2.43+ |
| Cloud | AWS (ECS, Lambda, S3, SNS, SQS, RDS, Secrets Manager, ALB, ECR) | — |

---

*Built as a portfolio project demonstrating end-to-end MLOps engineering — from raw data to production deployment with automated monitoring and retraining.*
