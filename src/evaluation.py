import logging
from abc import ABC, abstractmethod

import numpy as np

class Evaluation(ABC):

    @abstractmethod
    def calculate_scores(self, error_log: np.ndarray):
        pass

class RMSE(Evaluation):

    def calculate_scores(self, error_log: np.ndarray) -> float:
        
        try:
            logging.info("Fetching RMSE")
            rmse = error_log[-1]
            logging.info("RMSE: {}".format(rmse))

            return rmse
        
        except Exception as e:
            logging.error("Error in fetching RMSE score {}".format(e))
            raise e