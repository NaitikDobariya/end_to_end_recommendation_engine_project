from zenml import pipeline

from steps.clean_data import combine_dataframe, clean_data
from steps.ingest_data import ingest_data
from steps.model_train import train_model, save_model
from steps.evaluation import evaluate_model


@pipeline(enable_cache = False)
def training_pipeline(ratings_path: str, 
                      movies_path:str, 
                      num_users: int, 
                      num_items: int, 
                      num_latent_factors: int,
                      num_iterations: int, 
                      reg_param: float,
                      saved_model_name: str = None) -> None:
    
    ratings_df = ingest_data(ratings_path)
    movies_df = ingest_data(movies_path)

    _, data = combine_dataframe(ratings_df, movies_df)
    data = clean_data(data, num_users, num_items)

    model, model_CFM = train_model(data, 
                        num_users = num_users, 
                        num_items = num_items, 
                        num_latent_factors = num_latent_factors,
                        num_iterations = num_iterations,
                        reg_param = reg_param
                        )
    
    if saved_model_name:
        save_model(model_CFM, saved_model_name)

    rmse = evaluate_model(model)

