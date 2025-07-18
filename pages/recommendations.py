"""
Song Recommender - Recommendations Page

This page allows users to get personalized recommendations
based on their music preferences and listening history.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

# Add the root directory to the path to import from parent directory
sys.path.append(str(Path(__file__).parent.parent))
from config import DATA_PATH, AUDIO_FEATURES, MODEL_PARAMS
from models.recommender import SongRecommender

# Page configuration
st.set_page_config(
    page_title="Recommendations - Song Recommender",
    page_icon="🎵",
    layout="wide"
)

# Load the recommender
@st.cache_resource
def load_recommender():
    feature_matrix_path = DATA_PATH['PROCESSED']
    if not os.path.exists(feature_matrix_path):
        st.error(f"Feature matrix not found at {feature_matrix_path}. Please run the data processing first.")
        return None
    return SongRecommender(feature_matrix_path=feature_matrix_path)

recommender = load_recommender()

# Main function
def main():
    # Header
    st.title("🎵 Advanced Song Recommendations")
    st.write("Get personalized song recommendations based on your preferences.")
    
    if recommender is None:
        st.error("Recommender system could not be initialized. Please check if the data is processed correctly.")
        return
    
    # Check if we have a selected song from the main page
    song_from_main = None
    artist_from_main = None
    
    if 'selected_song' in st.session_state and 'selected_artist' in st.session_state:
        song_from_main = st.session_state.selected_song
        artist_from_main = st.session_state.selected_artist
        # Clear after use
        st.session_state.pop('selected_song', None)
        st.session_state.pop('selected_artist', None)
    
    # Create tabs for different recommendation methods
    tab1, tab2, tab3 = st.tabs(["By Song", "By Audio Features", "Playlist Generator"])
    
    # Tab 1: Recommendations by Song
    with tab1:
        st.subheader("Find songs similar to one you love")
        
        # If we have a song from the main page, use it automatically
        if song_from_main and artist_from_main:
            st.info(f"Finding recommendations for: **{song_from_main}** by {artist_from_main}")
            search_query = song_from_main
            auto_select_song = True
        else:
            # Search for song
            search_query = st.text_input("Search for a song or artist", key="song_search")
            auto_select_song = False
        
        if search_query:
            with st.spinner("Searching..."):
                results = recommender.search_songs(search_query, limit=10)
            
            if results:
                # Display results as selectable options
                selected_song = st.selectbox(
                    "Select a song to get recommendations",
                    options=range(len(results)),
                    format_func=lambda i: f"{results[i]['track_name']} by {results[i]['artists']}"
                )
                
                # If auto-select from main page or button pressed
                if auto_select_song or st.button("Get Similar Songs"):
                    # If auto-select, find the matching song
                    if auto_select_song:
                        matching_songs = [i for i, song in enumerate(results) 
                                       if song['track_name'] == song_from_main and song['artists'] == artist_from_main]
                        if matching_songs:
                            selected = results[matching_songs[0]]
                        else:
                            selected = results[0]  # Fallback to first result
                    else:
                        selected = results[selected_song]
                    
                    with st.spinner("Finding similar songs..."):
                        similar_songs = recommender.recommend_by_song_name(selected['track_name'], selected['artists'], n=10)
                        
                        if similar_songs:
                            # Show recommendations
                            st.subheader("Similar Songs:")
                            
                            # Create columns for recommendations
                            for i, song in enumerate(similar_songs):
                                col1, col2, col3 = st.columns([3, 2, 1])
                                with col1:
                                    st.write(f"**{song['track_name']}**")
                                with col2:
                                    st.write(f"by {song['artists']}")
                                with col3:
                                    st.write(f"Match: {song['similarity_score']:.2f}")
                                st.divider()
                        else:
                            st.info("No similar songs found.")
            else:
                st.info("No songs found matching your search. Try another query.")
    
    # Tab 2: Recommendations by Audio Features
    with tab2:
        st.subheader("Find songs with specific audio features")
        
        # Feature sliders
        st.write("Adjust the sliders to find songs with your preferred audio characteristics:")
        
        col1, col2 = st.columns(2)
        
        # Define feature ranges
        feature_values = {}
        
        with col1:
            feature_values['danceability'] = st.slider("Danceability", 0.0, 1.0, 0.5)
            feature_values['energy'] = st.slider("Energy", 0.0, 1.0, 0.5)
            feature_values['acousticness'] = st.slider("Acousticness", 0.0, 1.0, 0.5)
            feature_values['valence'] = st.slider("Positivity", 0.0, 1.0, 0.5)
        
        with col2:
            feature_values['instrumentalness'] = st.slider("Instrumentalness", 0.0, 1.0, 0.2)
            feature_values['liveness'] = st.slider("Liveness", 0.0, 1.0, 0.2)
            feature_values['speechiness'] = st.slider("Speechiness", 0.0, 1.0, 0.1)
            feature_values['tempo'] = st.slider("Tempo (Normalized)", 0.0, 1.0, 0.5)
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        features = list(feature_values.keys())
        values = list(feature_values.values())
        
        # Number of variables
        N = len(features)
        
        # Angle of each axis
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Values for each axis
        values = np.array(values)
        values = np.append(values, values[0])  # Close the loop
        
        # Draw the chart
        ax.plot(angles, values, linewidth=2, linestyle='solid')
        ax.fill(angles, values, alpha=0.25)
        
        # Set labels
        plt.xticks(angles[:-1], [f.title() for f in features], size=8)
        
        # Display the chart
        st.pyplot(fig)
        
        # Get recommendations based on features
        if st.button("Find Songs With These Features"):
            # Create feature vector
            # Make sure the order matches what the recommender expects
            feature_vector = np.array([
                feature_values[feature] for feature in recommender.audio_features
            ]).reshape(1, -1)
            
            with st.spinner("Finding songs with these characteristics..."):
                try:
                    # Find similar songs using batch processing
                    batch_size = MODEL_PARAMS['batch_size']
                    similarities = []
                    
                    for i in range(0, len(recommender.feature_matrix), batch_size):
                        batch = recommender.feature_matrix.iloc[i:i+batch_size]
                        batch_features = batch[recommender.audio_features].values
                        
                        batch_similarities = cosine_similarity(feature_vector, batch_features)[0]
                        
                        for j, sim in enumerate(batch_similarities):
                            similarities.append((i+j, sim))
                    
                    # Sort by similarity
                    similarities.sort(key=lambda x: x[1], reverse=True)
                    
                    # Get top matches
                    st.subheader("Songs matching your preferences:")
                    
                    col1, col2, col3 = st.columns([3, 2, 1])
                    with col1:
                        st.write("**Song**")
                    with col2:
                        st.write("**Artist**")
                    with col3:
                        st.write("**Match**")
                    st.divider()
                    
                    for idx, sim in similarities[:10]:
                        song = recommender.feature_matrix.iloc[idx]
                        
                        col1, col2, col3 = st.columns([3, 2, 1])
                        with col1:
                            st.write(f"{song['track_name']}")
                        with col2:
                            st.write(f"{song['artists']}")
                        with col3:
                            st.write(f"{sim:.2f}")
                        st.divider()
                except Exception as e:
                    st.error(f"Error finding songs: {str(e)}")
    
    # Tab 3: Playlist Generator
    with tab3:
        st.subheader("Create a custom playlist")
        st.write("Start with a song you like and we'll build a playlist for you")
        
        # Search for seed song
        seed_query = st.text_input("Start with this song or artist:", key="playlist_search")
        playlist_length = st.slider("Playlist length:", 5, 25, 10)
        
        # Diversity control
        st.write("Playlist diversity:")
        diversity = st.slider("Low (similar songs) to High (more variety)", 0.1, 1.0, 0.5)
        
        if seed_query and st.button("Generate Playlist"):
            with st.spinner("Searching for seed song..."):
                seed_results = recommender.search_songs(seed_query, limit=5)
            
            if seed_results:
                seed_song = seed_results[0]  # Use the top match
                
                st.write(f"Starting with: **{seed_song['track_name']}** by {seed_song['artists']}")
                
                with st.spinner("Generating playlist..."):
                    try:
                        # Get initial recommendations
                        similar_songs = recommender.recommend_by_song_name(
                            seed_song['track_name'], 
                            seed_song['artists'], 
                            n=min(50, playlist_length * 3)  # Get more than needed for diversity
                        )
                        
                        if similar_songs:
                            # Add diversity by adjusting the selection
                            playlist = []
                            
                            # Always include the seed song
                            playlist.append({
                                'track_name': seed_song['track_name'],
                                'artists': seed_song['artists'],
                                'is_seed': True
                            })
                            
                            # Use diversity parameter to select songs
                            # Higher diversity = more random selection from recommendations
                            if diversity < 0.3:
                                # Low diversity - take top matches
                                selected_songs = similar_songs[:playlist_length-1]
                            else:
                                # Higher diversity - mix in some variety
                                top_picks = max(2, int((1-diversity) * playlist_length))
                                random_picks = playlist_length - 1 - top_picks
                                
                                selected_songs = similar_songs[:top_picks]
                                
                                if random_picks > 0 and len(similar_songs) > top_picks:
                                    import random
                                    random_indices = random.sample(
                                        range(top_picks, len(similar_songs)), 
                                        min(random_picks, len(similar_songs) - top_picks)
                                    )
                                    selected_songs.extend([similar_songs[i] for i in random_indices])
                            
                            # Add to playlist
                            for song in selected_songs:
                                playlist.append({
                                    'track_name': song['track_name'],
                                    'artists': song['artists'],
                                    'similarity_score': song['similarity_score'],
                                    'is_seed': False
                                })
                            
                            # Display playlist
                            st.subheader("Your Custom Playlist:")
                            
                            for i, song in enumerate(playlist):
                                col1, col2, col3 = st.columns([1, 3, 2])
                                with col1:
                                    st.write(f"**{i+1}.**")
                                with col2:
                                    if song.get('is_seed', False):
                                        st.write(f"**{song['track_name']}** (Seed Song)")
                                    else:
                                        st.write(f"**{song['track_name']}**")
                                with col3:
                                    st.write(f"{song['artists']}")
                                st.divider()
                        else:
                            st.info("Could not generate recommendations for this seed song.")
                    except Exception as e:
                        st.error(f"Error generating playlist: {str(e)}")
            else:
                st.info("No songs found matching your search. Try another query.")

if __name__ == "__main__":
    main()
