from pipelines.experimentation_pipeline import experimentation_pipeline

if __name__ == "__main__":
    experimentation_pipeline(
        ratings_path = r"data\ratings.csv",
        movies_path = r"data\movies.csv",
        num_users = 100,
        num_items = 500,
        num_latent_factors_array = [15, 20, 25],
        num_iterations_array = [10],
        reg_param_array = [0.1, 0.05],
        project_name = "18-8-2024 experimentation run"
    )