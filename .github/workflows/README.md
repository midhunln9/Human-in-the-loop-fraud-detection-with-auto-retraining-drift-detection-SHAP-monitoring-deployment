# CI/CD Setup Guide

This GitHub Actions workflow automatically builds and pushes Docker images for the API and Pipeline services to Amazon ECR.

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
                "ecr:PutImage"
            ],
            "Resource": "*"
        }
    ]
}
```

### Setting Up ECR Repositories

Before running the workflow, create the ECR repositories in AWS:

```bash
aws ecr create-repository --repository-name fraud-detection-api --region us-east-1
aws ecr create-repository --repository-name fraud-detection-pipeline --region us-east-1
```

## Docker Image Naming

| Service | ECR Repository | Tags |
|---------|---------------|------|
| API | `fraud-detection-api` | `latest`, `<git-sha>` |
| Pipeline | `fraud-detection-pipeline` | `latest`, `<git-sha>` |

## Workflow Behavior

- **Pull Requests**: Builds the Docker images but does NOT push to ECR
- **Push to main**: Builds and pushes images to ECR with both `latest` and commit SHA tags

## Manual Trigger

To trigger the workflow manually from the GitHub UI:
1. Go to `Actions` tab in your repository
2. Select `Build and Push to ECR` workflow
3. Click `Run workflow`
