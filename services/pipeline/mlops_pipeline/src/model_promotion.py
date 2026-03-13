from mlops_pipeline.protocols.model_versioning_protocol import ModelVersioningProtocol
from mlops_pipeline.configs.wandb_config import WandbConfig
import logging
from mlops_pipeline.schemas.hyperparameter_tuning import HyperparameterTuningResult
from sklearn.base import BaseEstimator
from mlops_pipeline.exceptions import PromotionError
logger = logging.getLogger(__name__)

class ModelPromotion:
    def __init__(self, pr_auc_test : float, model_versioning_repository: ModelVersioningProtocol,
    wandb_config : WandbConfig) -> None:
        self.pr_auc_test = pr_auc_test
        self.model_versioning_repository = model_versioning_repository
        self.wandb_config = wandb_config
    
    def promote_model(self):
        try :
            production_model = None
            pr_auc_production = None
            try : 
                production_model, pr_auc_production = self.model_versioning_repository.stream_load_from_alias(artifact_name = self.wandb_config.model_artifact_name, 
                alias = "production")
                logger.info(f"Production model loaded successfully and start testing it now")
            except Exception as e:
                logger.info(f"No model in production")

            if production_model is None or self.pr_auc_test > pr_auc_production:
                self.model_versioning_repository.promote_artifact(self.wandb_config.model_artifact_name, 
                "staging","production")
                logger.info(f"Model promoted to production successfully")
            else:
                logger.info(f"Best staging model is not better than production model")
        except Exception as e:
            logger.error(f"Error promoting model: {e}")
            raise PromotionError(f"Error promoting model") from e