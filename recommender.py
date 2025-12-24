import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class UserBasedRecommender:
    def __init__(self, data: pd.DataFrame):
        """
        Initializes the recommender with data.
        Builds user-item matrix and user similarity matrix.
        """
        self.data = data
        self.user_item = self.data.pivot_table(index="user_id", columns="movie", values="rating")
        
        # Fill NaN with 0 for similarity calculation
        self.user_item_filled = self.user_item.fillna(0)
        
        # Calculate cosine similarity
        self.sim_matrix = cosine_similarity(self.user_item_filled)
        self.user_similarity = pd.DataFrame(
            self.sim_matrix, 
            index=self.user_item.index, 
            columns=self.user_item.index
        )

    def get_neighbors(self, target_user: str, k: int) -> pd.Series:
        """
        Returns the top k similar users for the target_user.
        """
        if target_user not in self.user_similarity.index:
            raise ValueError(f"User {target_user} not found.")
            
        return (
            self.user_similarity.loc[target_user]
            .drop(index=target_user)          # Exclude self
            .sort_values(ascending=False)     # Sort descending
            .head(k)                          # Top k
        )

    def predict_rating(self, movie: str, neighbor_sims: pd.Series) -> float | None:
        """
        Predicts the rating for a movie based on neighbors' ratings.
        """
        # Ratings of neighbors for this movie
        r = self.user_item.loc[neighbor_sims.index, movie]

        # Filter only neighbors who rated the movie
        mask = r.notna()
        if mask.sum() == 0:
            return None  # Cannot predict if no neighbor rated it

        r = r[mask]
        s = neighbor_sims[mask]

        # Weighted average
        if np.abs(s).sum() == 0:
            return None
            
        return float((r * s).sum() / (np.abs(s).sum()))

    def recommend(self, target_user: str, k_neighbors: int = 3, top_n: int = 5) -> pd.DataFrame:
        """
        Generates top_n movie recommendations for target_user.
        """
        if target_user not in self.user_item.index:
             raise ValueError(f"User {target_user} not found.")

        neighbors = self.get_neighbors(target_user, k_neighbors)
        
        # Find unseen movies (NaN entries for the user)
        user_ratings = self.user_item.loc[target_user]
        unseen_movies = user_ratings[user_ratings.isna()].index.tolist()

        predictions = []
        for movie in unseen_movies:
            pred = self.predict_rating(movie, neighbors)
            if pred is not None:
                predictions.append((movie, pred))

        pred_df = pd.DataFrame(predictions, columns=["movie", "pred_rating"])
        pred_df = pred_df.sort_values("pred_rating", ascending=False)
        
        return pred_df.head(top_n).reset_index(drop=True)
