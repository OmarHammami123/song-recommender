"""
Configuration settings for the song recommender system.
"""

# Path configurations
DATA_PATH = {
    'RAW': 'data/raw/dataset.csv',
    'PROCESSED': 'data/processed/feature_matrix.csv',
    'MODELS': 'models/'
}

# Feature configurations
AUDIO_FEATURES = [
    'acousticness', 
    'danceability', 
    'energy', 
    'instrumentalness', 
    'liveness', 
    'loudness', 
    'speechiness', 
    'tempo', 
    'valence'
]

# Model parameters
MODEL_PARAMS = {
    'similarity_metric': 'cosine',
    'default_recommendations': 5,
    'batch_size': 1000  # For memory-efficient processing
}

# UI configurations
UI_CONFIG = {
    'primary_color': '#1DB954',  # Spotify-like green
    'secondary_color': '#191414',  # Spotify-like dark
    'page_title': 'Song Recommender',
    'page_icon': '🎵'
}

# Example songs for the main page
EXAMPLE_SONGS = [
    "Shape of You",
    "Billie Jean",
    "Bohemian Rhapsody",
    "Bad Guy",
    "Dance Monkey"
]
