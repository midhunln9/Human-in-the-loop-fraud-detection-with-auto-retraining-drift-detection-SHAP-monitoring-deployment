from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

def test_if_sklearn_model_returned(master_tuner):
    model = master_tuner.start_hyperparameter_tuning()
    assert isinstance(model, XGBClassifier) or isinstance(model, LGBMClassifier)

def test_metadata_of_model(master_tuner):
    model = master_tuner.start_hyperparameter_tuning()
    assert model.get_params() is not None
    assert model.get_params() is not None