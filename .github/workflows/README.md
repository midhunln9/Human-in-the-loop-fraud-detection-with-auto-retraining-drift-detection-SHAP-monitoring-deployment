# CI/CD Setup Guide

This GitHub Actions workflow automatically tests, builds, and pushes Docker images for the API and Pipeline services to Amazon ECR.

## Workflow Overview

The workflow consists of four jobs that run in sequence:

1. **test-api** - Runs unit and integration tests for the API service
2. **test-pipeline** - Runs tests for the Pipeline service
3. **build-and-push-api** - Builds and pushes the API Docker image (depends on test-api)
4. **build-and-push-pipeline** - Builds and pushes the Pipeline Docker image (depends on test-pipeline)

## GitHub Secrets Required

Configure the following secrets in your GitHub repository settings (`Settings > Secrets and variables > Actions`):

| Secret Name | Description | Example Value |
|-------------|-------------|---------------|
| `AWS_ACCESS_KEY_ID` | AWS IAM user access key ID | `AKIAIOSFODNN7EXAMPLE` |
| `AWS_SECRET_ACCESS_KEY` | AWS IAM user secret access key | `wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY` |
| `AWS_REGION` | AWS region where ECR repositories are located | `us-east-1` |
| `ECR_REPOSITORY_API` | Name of the ECR repository for the API service | `fraud-detection-api` |
| `ECR_REPOSITORY_PIPELINE` | Name of the ECR repository for the Pipeline service | `fraud-detection-pipeline` |

## AWS IAM Permissions

The IAM user associated with the access keys needs the following permissions to push images to ECR:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "ecr:GetAuthorizationToken",
                "ecr:BatchCheckLayerAvailability",
                "ecr:GetDownloadUrlForLayer",
                "ecr:BatchGetImage",
                "ecr:InitiateLayerUpload",
                "ecr:UploadLayerPart",
                "ecr:CompleteLayerUpload",
                "ecr:PutImage",
                "ecr:CreateRepository",
                "ecr:DescribeRepositories"
            ],
            "Resource": "*"
        }
    ]
}
```

### Setting Up ECR Repositories

The workflow will automatically create the ECR repositories if they don't exist. However, if you prefer to create them manually beforehand, you can use:

```bash
aws ecr create-repository --repository-name fraud-detection-api --region us-east-1
aws ecr create-repository --repository-name fraud-detection-pipeline --region us-east-1
```

## Test Jobs

### API Service Tests
- **Location**: `services/api/`
- **Python Version**: 3.12
- **Command**: `pytest -v`
- **Dependencies**: Installed via `pip install -e .`

### Pipeline Service Tests
- **Location**: `services/pipeline/`
- **Python Version**: 3.12
- **Command**: `pytest -v`
- **Dependencies**: Installed via `pip install -e .`

## Docker Image Naming

| Service | ECR Repository | Tags |
|---------|---------------|------|
| API | `fraud-detection-api` | `latest`, `<git-sha>` |
| Pipeline | `fraud-detection-pipeline` | `latest`, `<git-sha>` |

## Workflow Behavior

| Event | Tests | Push to ECR |
|-------|-------|-------------|
| **Pull Request to main** | Runs | No (builds only) |
| **Push to main** | Runs | Yes (pushes both tags) |

- Tests always run on both pull requests and pushes to main
- Docker images are only pushed to ECR when code is merged/pushed to the `main` branch
- On pull requests, images are built but not pushed (verifies build succeeds)

## Job Dependencies

```
test-api ---------> build-and-push-api
                          |
                          v
                    (pushes to ECR)

test-pipeline -----> build-and-push-pipeline
                          |
                          v
                    (pushes to ECR)
```

Build jobs only run if their corresponding test jobs pass successfully.

## Manual Trigger

To trigger the workflow manually from the GitHub UI:
1. Go to `Actions` tab in your repository
2. Select `Build and Push to ECR` workflow
3. Click `Run workflow`
