import logging 

import numpy as np
from zenml import step
from typing import Tuple

from src.model_develpoment import CollaborativeFilteringModel, CollaborativeFiltering

@step
def train_model(X: np.ndarray, 
                num_users: int, 
                num_items: int, 
                num_latent_factors: int,
                num_iterations: int, 
                reg_param:float) -> Tuple[CollaborativeFiltering, CollaborativeFilteringModel]:

    try:
        CFM = CollaborativeFilteringModel()
        model = CFM.train(X_train = X,
                          num_users = num_users,
                          num_items = num_items,
                          num_latent_factors = num_latent_factors,
                          num_iterations = num_iterations,
                          reg_param = reg_param)
        logging.info("Model training succesful")

        return model, CFM
    except Exception as e:
        logging.error("Error in training the model: {}".format(e))

@step
def save_model(model: CollaborativeFilteringModel, model_name: str) -> None:
    try:
        model.save_model(model_name = model_name)
        logging.info("Model matrices saved successfully")

    except Exception as e:
        logging.error("Error in saving the model matrices : {}".format(e))
        raise e