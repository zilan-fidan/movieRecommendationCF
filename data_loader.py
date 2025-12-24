import pandas as pd

def load_sample_data() -> pd.DataFrame:
    """
    Creates and returns a sample dataframe for movie ratings.
    """
    ratings = pd.DataFrame(
        [
            ("U1", "Inception",     5),
            ("U1", "Interstellar",  4),
            ("U1", "The Matrix",    5),

            ("U2", "Inception",     4),
            ("U2", "The Matrix",    5),
            ("U2", "Joker",         2),

            ("U3", "Interstellar",  5),
            ("U3", "The Matrix",    4),
            ("U3", "Joker",         1),
            ("U3", "Toy Story",     3),

            ("U4", "Inception",     2),
            ("U4", "Interstellar",  2),
            ("U4", "Toy Story",     5),

            ("U5", "The Matrix",    5),
            ("U5", "Toy Story",     4),
            ("U5", "Joker",         1),
        ],
        columns=["user_id", "movie", "rating"]
    )
    return ratings
