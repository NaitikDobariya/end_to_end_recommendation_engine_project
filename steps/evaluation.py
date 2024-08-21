import logging
import pandas as pd

from zenml import step
from src.evaluation import RMSE
from collaborative_filtering import CollaborativeFiltering

@step
def evaluate_model(model: CollaborativeFiltering) -> float:
    try:
        error_log = model.error_log

        rmse_class = RMSE()
        rmse = rmse_class.calculate_scores(error_log = error_log)

        return rmse

    except Exception as e:
        logging.error("Error in evaluating the model: {}".format(e))
        raise e