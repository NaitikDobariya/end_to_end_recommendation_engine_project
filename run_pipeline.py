from pipelines.training_pipeline import training_pipeline
import wandb

if __name__ == "__main__":

    wandb.init(project='18-8-2024 experimentation run')
    runs = wandb.Api().runs('ureaguanidine-bits-pilani/18-8-2024 experimentation run')

    best_run = -float('inf')
    Hyperparameters = None
    for run in runs:
        config = run.config  
        summary = run.summary  
        error_metric = summary.get('error')

        if config and error_metric is not None:
            if best_run < error_metric:
                Hyperparameters = config

    training_pipeline(
        ratings_path = r"data\ratings.csv",
        movies_path = r"data\movies.csv",
        num_users = Hyperparameters['num_users'],
        num_items = Hyperparameters['num_items'],
        num_latent_factors = Hyperparameters['num_latent_factors'],
        num_iterations = Hyperparameters['num_iterations'],
        reg_param = Hyperparameters['reg_param'],
        saved_model_name = "18-8-2024_model"
    )