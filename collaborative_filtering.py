import numpy as np

class CollaborativeFiltering:
    def __init__(self, num_users: int, num_items: int, num_latent_factors: int, reg_param: float):
        self.num_users = num_users
        self.num_items = num_items
        self.num_latent_factors = num_latent_factors
        self.reg_param = reg_param
        self.U = np.random.normal(scale=1.0/num_latent_factors, size=(num_users, num_latent_factors))
        self.V = np.random.normal(scale=1.0/num_latent_factors, size=(num_items, num_latent_factors))
        self.W = np.eye(num_latent_factors, dtype = np.float64)

    def train(self, X: np.ndarray, num_iterations: int, verbose:int = True) -> None:

        self.error_log = np.zeros(num_iterations)

        for iter in range(num_iterations):
            
            if verbose: print(f"Current iteration {iter + 1} |", end = "")
            for i, Xi in enumerate(X):
                
                mask_column = np.array([[0 if x is None else 1 for x in Xi]]) 
                mask = mask_column.T @ mask_column

                Ai = self.V.T @ mask @ self.V + self.reg_param * self.W
                Xi = np.where(Xi == None, 0, Xi)
                Bi = self.V.T @ mask @ Xi
                self.U[i] = np.linalg.solve(Ai, Bi.astype(np.float64))

            if verbose: print(" U done |", end = "")
            for i, Xi in enumerate(X.T):
                
                mask_column = np.array([[0 if x is None else 1 for x in Xi]]) 
                mask = mask_column.T @ mask_column

                Ai = self.U.T @ mask @ self.U + self.reg_param * self.W
                Xi = np.where(Xi == None, 0, Xi)
                Bi = self.U.T @ mask @ Xi
                self.V[i] = np.linalg.solve(Ai, Bi.astype(np.float64))

            self.error_log[iter] = np.sqrt(np.mean((X[X != None] - self.reconstructed_matrix()[X != None])**2))

            # Made by Naitik Dobariya
            if verbose: print(" V done |", end = "")
            if verbose: print(f" The error is {self.error_log[iter]}\n")
            

    def predict(self, user_id: int, item_id: np.ndarray) -> np.ndarray:
        prediction = np.ceil(self.U @ self.V.T)
        return prediction[user_id][item_id]
    
    def reconstructed_matrix(self) -> np.ndarray:
        return self.U @ self.V.T
    


if __name__ == "__main__":
    num_users = 10
    num_items = 20
    num_latent_factors = 150
    num_iterations = 20
    reg_param = 0.01

    X = np.array([
        [5, 3, 0, 1, 4, 0, 0, 2, 1, 5, 0, 4, 3, 0, 2, 5, 1, 0, 3, 4],
        [4, 0, 0, 1, 0, 2, 3, 0, 0, 4, 5, 1, 0, 2, 0, 3, 4, 1, 0, 0],
        [1, 1, 0, 5, 2, 4, 0, 0, 3, 2, 0, 1, 4, 3, 0, 0, 5, 2, 1, 4],
        [1, 0, 0, 4, 0, 3, 5, 2, 0, 1, 2, 0, 4, 0, 3, 1, 5, 0, 2, 4],
        [0, 1, 5, 4, 0, 0, 2, 1, 3, 0, 1, 4, 0, 5, 3, 0, 1, 4, 2, 0],
        [3, 4, 2, 0, 1, 5, 0, 3, 0, 0, 2, 0, 1, 5, 4, 0, 3, 2, 0, 1],
        [0, 2, 3, 0, 5, 1, 4, 0, 0, 2, 0, 3, 1, 0, 5, 4, 2, 0, 0, 3],
        [5, 0, 0, 3, 4, 0, 1, 5, 2, 0, 0, 2, 3, 1, 0, 4, 5, 0, 3, 0],
        [4, 3, 1, 0, 0, 2, 0, 4, 5, 1, 2, 0, 0, 3, 1, 5, 0, 2, 4, 3],
        [0, 5, 4, 1, 3, 0, 0, 2, 0, 0, 3, 1, 0, 4, 5, 2, 0, 1, 0, 5]
    ])

    X = np.where(X == 0, None, X)

    # # Create an instance of the ALS class
    als = CollaborativeFiltering(num_users, num_items, num_latent_factors, reg_param)

    # Train the ALS model
    als.train(X, num_iterations)