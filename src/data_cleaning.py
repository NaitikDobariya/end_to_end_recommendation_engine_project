import logging
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class DataStrategy(ABC):

    @abstractmethod
    def handle_data(self, data:np.ndarray) -> np.ndarray:
        pass

class DataPreProcessStrategy(DataStrategy):

    def handle_data(self, data: np.ndarray) -> np.ndarray:
        try:
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    if type(data[i][j]) == float:
                        data[i][j] = None
            
            return data

        except Exception as e:
            logging.error("Error in processing data: {}".format(e))
            raise e

class DataTrimmingStrategy(DataStrategy):

    def handle_data(self, data: np.ndarray, num_users: int, num_items: int) -> np.ndarray:
        try:
            data = data[:num_users]
            data = data.T[:num_items].T

            return data

        except Exception as e:
            logging.error("Error in trimming the data {}".format(e))
            raise e