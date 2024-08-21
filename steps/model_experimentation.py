import logging
import time

import numpy as np
import itertools
import wandb
from zenml import step

from src.model_develpoment import CollaborativeFilteringModel

@step
def model_experiment(X: np.ndarray, 
                    num_users: int, 
                    num_items: int, 
                    num_latent_factors_array: list,
                    num_iterations_array: list, 
                    reg_param_array: list,
                    project_name: str) -> None: 
    
    try:
        num_total_runs = len(num_latent_factors_array) * len(num_iterations_array) * len(reg_param_array)
        
        runs_completed = 0
        for num_latent_factors, num_iterations, reg_param in itertools.product(num_latent_factors_array, num_iterations_array, reg_param_array):
            
            current_time = time.localtime()
            run_name = time.strftime("RunID: %H:%M:%S %Y-%m-%d", current_time)

            wandb.init(project = project_name, reinit = True, name = run_name)
            wandb.config.update({
                "num_users": num_users,
                "num_items": num_items,
                "num_latent_factors": num_latent_factors,
                "reg_param": reg_param,
                "num_iterations": num_iterations
            })

            CFM = CollaborativeFilteringModel()
            model = CFM.train(X_train = X,
                            num_users = num_users,
                            num_items = num_items,
                            num_latent_factors = num_latent_factors,
                            num_iterations = num_iterations,
                            reg_param = reg_param)
            
            logging.info(f"Model training succesful {runs_completed + 1}/{num_total_runs}")
            runs_completed += 1

            for iter in range(num_iterations):
                wandb.log({"iteration": iter + 1, "error": model.error_log[iter]})
        
            wandb.finish()
            logging.info("Model info uploaded")

    except Exception as e:
        logging.error("Error in training the model: {}".format(e))