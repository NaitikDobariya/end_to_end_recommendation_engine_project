import logging
from typing import Tuple
from typing_extensions import Annotated

import pandas as pd
import numpy as np

from zenml import step
from src.data_cleaning import DataPreProcessStrategy, DataTrimmingStrategy

@step
def combine_dataframe(ratings_df: pd.DataFrame, movies_df: pd.DataFrame) -> Tuple[Annotated[pd.DataFrame, 'data_df'], Annotated[np.ndarray, 'data']]:
    try:
        user_IDs = sorted(ratings_df["userId"].unique().tolist())
        num_users = ratings_df.nunique().iloc[0]

        data_df = pd.DataFrame(None, index = user_IDs, columns = movies_df["movieId"])

        for row in ratings_df.values:
            data_df.loc[row[0], row[1]] = row[2] 

        return data_df, data_df.values

    except Exception as e:
        logging.error("Error in combining dataframes: {}".format(e))
        raise e
    
@step
def clean_data(data: np.ndarray, num_users: int, num_items: int) -> np.ndarray:
    try:
        DataTrimming = DataTrimmingStrategy()
        trimmed_data = DataTrimming.handle_data(data = data, num_users = num_users, num_items = num_items)

        PreProcessStrategy = DataPreProcessStrategy()
        processed_data = PreProcessStrategy.handle_data(trimmed_data)

        logging.info("Data cleaning done")

        return processed_data
    
    except Exception as e:
        logging.error("Error in data cleaning: {}".format(e))
        raise e