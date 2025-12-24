import os
import zipfile
import requests
import pandas as pd
import io

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
DATA_DIR = "data"

def download_and_extract_data():
    """
    Downloads the MovieLens dataset if not already present.
    """
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    # Check if files already exist
    movies_path = os.path.join(DATA_DIR, "ml-latest-small", "movies.csv")
    ratings_path = os.path.join(DATA_DIR, "ml-latest-small", "ratings.csv")
    
    if os.path.exists(movies_path) and os.path.exists(ratings_path):
        print("Veri dosyalari zaten mevcut.")
        return

    print("Veri seti indiriliyor... (MovieLens Latest Small)")
    try:
        r = requests.get(MOVIELENS_URL)
        r.raise_for_status()
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(DATA_DIR)
        print("İndirme ve cikarma tamamlandi.")
    except Exception as e:
        print(f"Veri indirilirken hata olustu: {e}")
        raise

def load_movielens_data() -> pd.DataFrame:
    """
    Loads ratings and merges with movie titles.
    Returns a dataframe with columns: [user_id, movie, rating]
    """
    download_and_extract_data()
    
    movies_path = os.path.join(DATA_DIR, "ml-latest-small", "movies.csv")
    ratings_path = os.path.join(DATA_DIR, "ml-latest-small", "ratings.csv")
    
    # Load CSVs
    movies_df = pd.read_csv(movies_path)
    ratings_df = pd.read_csv(ratings_path)
    
    # Merge to get movie titles
    # movies.csv: movieId, title, genres
    # ratings.csv: userId, movieId, rating, timestamp
    
    merged_df = pd.merge(ratings_df, movies_df, on="movieId")
    
    # Select and rename columns to match our existing logic
    # We use 'title' as 'movie' identifier for simplicity in this project
    # Note: Using titles can be risky if duplicates exist, but okay for this scale.
    final_df = merged_df[["userId", "title", "rating"]]
    final_df.columns = ["user_id", "movie", "rating"]
    
    return final_df

# For backward compatibility if needed, but we plan to switch logic
def load_sample_data() -> pd.DataFrame:
    # ... (Keep existing simple data if necessary, or just alias)
    # For now, let's keep the old one distinct just in case
    return pd.DataFrame([
        ("U1", "Inception", 5), ("U1", "Interstellar", 4) # ... shortened
    ], columns=["user_id", "movie", "rating"])
