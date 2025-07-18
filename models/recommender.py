import pandas as pd 
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os 

class SongRecommender:
    """
    memory-efficient song recommender system that calculates similarities on-demand
    rather than storing a large similarity matrix in memory.
    """
    
    def __init__(self, feature_matrix_path=None):
        """initialize the recommender with a path to the feature matrix"""
        self.feature_matrix=None
        self.audio_features= ['acousticness', 'danceability', 'energy', 
                               'instrumentalness', 'liveness', 'loudness', 
                               'speechiness', 'tempo', 'valence']
        if feature_matrix_path and os.path.exists(feature_matrix_path):
            self.load_data(feature_matrix_path)
    def load_data(self, feature_matrix_path):
        """load the feature matrix from a CSV file"""
        self.feature_matrix = pd.read_csv(feature_matrix_path)
        if not set(self.audio_features).issubset(self.feature_matrix.columns):
            raise ValueError("Feature matrix must contain the following audio features: " + ", ".join(self.audio_features))
        print(f"Feature matrix loaded with shape: {self.feature_matrix.shape}")
        return self
    
    def search_songs(self , query , limit=10):
        if self.feature_matrix is None:
            raise ValueError("Feature matrix is not loaded")
        
        #combining song name and artist for search
        self.feature_matrix['search_field'] =  (
            self.feature_matrix['track_name'].str.lower() + ' ' + 
            self.feature_matrix['artists'].str.lower()
        )
        
        #filter words that match the query 
        query = query.lower()
        matches = self.feature_matrix[self.feature_matrix['search_field'].str.contains(query , na=False)]
        
        return matches.head(limit)[['track_name', 'artists']].to_dict(orient='records')
       
    def get_song_by_name_and_artist(self , track_name, artist_name):
        """get a song by its name and artist"""
        if self.feature_matrix is None:
            raise ValueError("Feature matrix is not loaded")
        
        track_name = track_name.lower()
        artist_name = artist_name.lower()
        
        song = self.feature_matrix[
            (self.feature_matrix['track_name'].str.lower() == track_name) & 
            (self.feature_matrix['artists'].str.lower() == artist_name)
        ]
        
        if song.empty:
            return None
        
        return song.iloc[0].to_dict()
    
    def get_song_features(self, song_idx):
        """Get audio features for a specific song"""
        if self.feature_matrix is None or song_idx >= len(self.feature_matrix):
            return None
            
        song = self.feature_matrix.iloc[song_idx]
        features = {feature: song[feature] for feature in self.audio_features if feature in song.index}
        return features

    def _get_similar_songs(self, song_idx, n=5):
        """get similar songs to a given song """
        if self.feature_matrix is None or song_idx >= len(self.feature_matrix):
            raise ValueError("Feature matrix is not loaded or index is out of bounds")
        
        song_features = self.feature_matrix.iloc[song_idx][self.audio_features].values.reshape(1, -1)
        
        #calculate cosine similarity with all the songs (mermory efficient approach)
        batch_size = 1000
        all_similarities = []
        
        for i in range (0, len(self.feature_matrix), batch_size):
            batch = self.feature_matrix.iloc[i:i + batch_size]
            batch_features = batch[self.audio_features].values
            
            similarities = cosine_similarity(song_features, batch_features)[0]
            
            for j, sim in enumerate(similarities):
                all_similarities.append((i + j, sim))
                
        all_similarities.sort(key=lambda x: x[1], reverse=True)   
        similar_indices = [idx for idx,_ in all_similarities[1:n+1]]  
        similar_songs = []
        for idx in similar_indices:
            song = self.feature_matrix.iloc[idx]
            sim_score = next(score for song_idx, score in all_similarities if song_idx == idx)

            similar_songs.append({
                'track_name': song['track_name'],
                'artists': song['artists'],
                'similarity_score': sim_score
            })   
            
            
        return similar_songs
    
    
    def recommend_by_song_name(self , track_name , artist_name=None, n=5):
        """recommend songs similar to a given song name and artist"""
        if self.feature_matrix is None:
            raise ValueError("Feature matrix is not loaded")
        
        matches= self.feature_matrix[
            self.feature_matrix['track_name'].str.lower() == track_name.lower()
        ]
        if artist_name and not matches.empty:
            artist_matches= matches[
                matches['artists'].str.lower() == artist_name.lower()
            ]
        if matches.empty:
            raise ValueError(f"Song '{track_name}' not found")    
        song_idx = matches.index[0]
        return self._get_similar_songs(song_idx, n)
    
    
    
if __name__ == "__main__":
    # test the recommender system
    recommender = SongRecommender (feature_matrix_path='data/processed/feature_matrix.csv')
    #search for a song
    query = "Shape of You"
    results = recommender.search_songs(query)
    print("\n search results for query :")
    for song in results:
        print(f"{song['track_name']} by {song['artists']}")
        
    if results:
        similar = recommender.recommend_by_song_name(results[0]['track_name'], results[0]['artists'])
        print("\nSimilar songs:")
        for song in similar:
            print(f"{song['track_name']} by {song['artists']} (Similarity: {song['similarity_score']:.4f})")

