import wandb
from mlops_pipeline.configs.wandb_config import WandbConfig
from mlops_pipeline.protocols.model_versioning_protocol import ModelVersioningProtocol
from sklearn.base import BaseEstimator
import tempfile
import os
import joblib
import logging
from typing import Tuple
from mlops_pipeline.exceptions import ArtifactError

logger = logging.getLogger(__name__)

class WandbRepository(ModelVersioningProtocol):
    def __init__(self, config: WandbConfig):
        self.config = config

    def create_and_log_model_artifact_to_run(self, model: BaseEstimator, alias: str, metric: float) -> None:
        try : 
            with wandb.init(entity=self.config.entity, project=self.config.project) as run:
                with tempfile.TemporaryDirectory() as temp_dir:
                    model_path = os.path.join(temp_dir, self.config.model_file_name)
                    joblib.dump(model, model_path)
                    artifact = wandb.Artifact(self.config.model_artifact_name, type=self.config.model_artifact_type,
                    metadata={"pr_auc": metric})
                    artifact.add_file(model_path)
                    run.log_artifact(artifact, aliases=[alias])
            logger.info(f"Model artifact created and logged successfully")
        except Exception as e:
            logger.error(f"Error creating and logging model artifact: {e}")
            raise ArtifactError(f"Error creating and logging model artifact") from e
    
    def create_and_log_preprocessor_artifact_to_run(self, preprocessor: BaseEstimator, alias: str) -> None:
        try : 
            with wandb.init(entity=self.config.entity, project=self.config.project) as run:
                with tempfile.TemporaryDirectory() as temp_dir:
                    preprocessor_path = os.path.join(temp_dir, self.config.preprocessor_file_name)
                    joblib.dump(preprocessor, preprocessor_path)
                    artifact = wandb.Artifact(self.config.preprocessor_artifact_name, type=self.config.preprocessor_artifact_type)
                    artifact.add_file(preprocessor_path)
                    run.log_artifact(artifact, aliases=[alias])
            logger.info(f"Preprocessor artifact created and logged successfully")
        except Exception as e:
            logger.error(f"Error creating and logging preprocessor artifact: {e}")
            raise ArtifactError(f"Error creating and logging preprocessor artifact") from e
        
    def stream_load_from_alias(self, artifact_name: str, alias: str) -> Tuple[BaseEstimator, float]:
        try : 
            with wandb.init(entity=self.config.entity, project=self.config.project) as run:
                with tempfile.TemporaryDirectory() as temp_dir:
                    try:
                        artifact = run.use_artifact(f"{artifact_name}:{alias}")
                        with tempfile.TemporaryDirectory() as temp_dir:
                            artifact.download(root = temp_dir)
                            return joblib.load(os.path.join(temp_dir, self.config.model_file_name)), artifact.metadata.get("pr_auc", None)
                    except wandb.errors.CommError:
                        logger.warning(f"Artifact {artifact_name}:{alias} not found. Returning None.")
                        return None, None
        except Exception as e:
            logger.error(f"Error downloading artifact: {e}")
            raise ArtifactError(f"Error downloading artifact") from e
    
    def promote_artifact(self, artifact_name: str, from_alias: str, to_alias: str) -> None:
        try : 
            with wandb.init(entity=self.config.entity, project=self.config.project) as run:
                artifact = run.use_artifact(f"{artifact_name}:{from_alias}")
                artifact.aliases.append(to_alias)
                if from_alias in artifact.aliases:
                    artifact.aliases.remove(from_alias)
                artifact.save()
            logger.info(f"Artifact promoted successfully")
        except Exception as e:
            logger.error(f"Error promoting artifact: {e}")
            raise ArtifactError(f"Error promoting artifact") from e