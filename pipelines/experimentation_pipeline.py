from zenml import pipeline
import numpy as np

from steps.clean_data import combine_dataframe, clean_data
from steps.ingest_data import ingest_data
from steps.model_experimentation import model_experiment


@pipeline(enable_cache = False)
def experimentation_pipeline(ratings_path: str, 
                            movies_path:str, 
                            num_users: int, 
                            num_items: int, 
                            num_latent_factors_array: list,
                            num_iterations_array: list, 
                            reg_param_array: list,
                            project_name: str) -> None:
    
    ratings_df = ingest_data(ratings_path)
    movies_df = ingest_data(movies_path)

    _, data = combine_dataframe(ratings_df, movies_df)
    data = clean_data(data, num_users, num_items)

    model = model_experiment(data, 
                        num_users = num_users, 
                        num_items = num_items, 
                        num_latent_factors_array = num_latent_factors_array,
                        num_iterations_array = num_iterations_array,
                        reg_param_array = reg_param_array,
                        project_name = project_name)