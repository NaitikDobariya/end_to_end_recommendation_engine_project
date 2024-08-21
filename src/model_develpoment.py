import logging
from abc import ABC, abstractmethod
import numpy as np
import os

from collaborative_filtering import CollaborativeFiltering

class Model(ABC):

    @abstractmethod
    def train(self, X_train: np.ndarray):
        pass

class CollaborativeFilteringModel(Model):

    def train(self, 
              X_train: np.ndarray, 
              num_users: int, 
              num_items: int, 
              num_latent_factors: int,
              num_iterations: int, 
              reg_param:float) -> CollaborativeFiltering:
        
        try:
            self.CFM = CollaborativeFiltering(num_users = num_users,
                                         num_items = num_items,
                                         num_latent_factors = num_latent_factors,
                                         reg_param = reg_param)
            self.CFM.train(X_train, num_iterations, verbose = False)
            logging.info("Model training completed")

            return self.CFM

        except Exception as e:
            logging.error("Error in training model: {}".format(e))
            raise e
        
    def save_model(self, model_name: str) -> None:
        try:
            model_dir = f"./models/{model_name}"
            os.makedirs(model_dir, exist_ok=True)

            # Save the matrices in the created directory
            np.save(f"{model_dir}/U_matrix.npy", self.CFM.U)
            np.save(f"{model_dir}/V_matrix.npy", self.CFM.V)


            logging.info("Model matrices saved successfully")

        except Exception as e:
            logging.error("Error in saving the model matrices : {}".format(e))
            raise e