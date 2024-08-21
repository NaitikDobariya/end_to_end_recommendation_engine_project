import numpy as np
import polars as pd
import os

class model:
    def __init__(self, model_name: str) -> None:
        model_dir = f"./models/{model_name}"
        
        # Load the matrices from the specified directory
        self.U_matrix = np.load(os.path.join(model_dir, "U_matrix.npy"))
        self.V_matrix = np.load(os.path.join(model_dir, "V_matrix.npy"))

        # Made by Naitik Dobariya
        movies = pd.read_csv(r"data\movies.csv")
        self.id_to_title = dict(zip(movies['movieId'],  movies['title']))

    def recommend(self, user_ID, num_recommendations):
        recommendations = self.U_matrix[user_ID] @ self.V_matrix.T

        top_indices = np.argpartition(-recommendations, 10)[:num_recommendations]
        top_indices_sorted = top_indices[np.argsort(-recommendations[top_indices])]

        return np.array([self.id_to_title.get(mid, "Unknown") for mid in top_indices_sorted])

# if __name__ == "__main__":
#     model = model("First_try")

#     print(model.recommend(5, 10))