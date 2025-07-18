"""
Song Recommender - Data Exploration Page

This page allows users to explore the dataset and understand 
the audio features that power the recommendation system.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os
import sys
from pathlib import Path

# Add the root directory to the path to import from parent directory
sys.path.append(str(Path(__file__).parent.parent))
from config import DATA_PATH, AUDIO_FEATURES

# Page configuration
st.set_page_config(
    page_title="Explore Data - Song Recommender",
    page_icon="🔍",
    layout="wide"
)

# Load the dataset
@st.cache_data
def load_data():
    """Load the processed dataset"""
    try:
        df = pd.read_csv(DATA_PATH['PROCESSED'])
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Main function
def main():
    # Header
    st.title("🔍 Explore Music Dataset")
    st.write("Explore the audio features that power our recommendation system.")
    
    # Load data
    df = load_data()
    
    if df is None:
        st.error("Could not load the dataset. Please make sure the data processing has been completed.")
        return
    
    # Overview stats
    st.header("Dataset Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Songs", f"{len(df):,}")
    
    # Check if certain columns exist
    if 'artists' in df.columns:
        col2.metric("Unique Artists", f"{df['artists'].nunique():,}")
    
    if 'track_genre' in df.columns:
        col3.metric("Genres", f"{df['track_genre'].nunique():,}")
    
    # Show a sample of the data
    with st.expander("Show sample data"):
        st.dataframe(df.head(10))
    
    # Feature distributions
    st.header("Audio Feature Distributions")
    
    # Select features
    available_features = [feat for feat in AUDIO_FEATURES if feat in df.columns]
    
    if not available_features:
        st.warning("No audio features found in the dataset.")
        return
    
    # Feature selection
    selected_feature = st.selectbox(
        "Select a feature to explore", 
        options=available_features,
        format_func=lambda x: x.title()
    )
    
    # Visualization tabs
    tab1, tab2, tab3 = st.tabs(["Distribution", "Correlations", "Top Songs"])
    
    with tab1:
        st.subheader(f"Distribution of {selected_feature.title()}")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df[selected_feature], kde=True, ax=ax)
        plt.title(f"Distribution of {selected_feature.title()}")
        plt.xlabel(selected_feature.title())
        plt.ylabel("Count")
        st.pyplot(fig)
        
        st.write(f"**Description:** {get_feature_description(selected_feature)}")
        
        # Feature stats
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Average", f"{df[selected_feature].mean():.2f}")
        col2.metric("Median", f"{df[selected_feature].median():.2f}")
        col3.metric("Min", f"{df[selected_feature].min():.2f}")
        col4.metric("Max", f"{df[selected_feature].max():.2f}")
    
    with tab2:
        st.subheader("Feature Correlations")
        
        # Create correlation matrix
        corr = df[available_features].corr()
        
        # Plot heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        plt.title("Feature Correlations")
        st.pyplot(fig)
        
        # Plot scatter against selected feature
        if len(available_features) > 1:
            second_feature = st.selectbox(
                f"Compare {selected_feature.title()} with", 
                options=[f for f in available_features if f != selected_feature],
                format_func=lambda x: x.title()
            )
            
            fig = px.scatter(
                df.sample(min(1000, len(df))),  # Sample for better performance
                x=selected_feature,
                y=second_feature,
                hover_data=['track_name', 'artists'] if 'track_name' in df.columns else None,
                opacity=0.7,
                title=f"{selected_feature.title()} vs {second_feature.title()}"
            )
            st.plotly_chart(fig)
    
    with tab3:
        st.subheader(f"Top Songs by {selected_feature.title()}")
        
        # Choose highest or lowest
        sort_order = st.radio(
            "Sort order",
            options=["Highest", "Lowest"],
            horizontal=True
        )
        
        # Filter columns to display
        display_cols = ['track_name', 'artists'] if all(col in df.columns for col in ['track_name', 'artists']) else df.columns[:2].tolist()
        display_cols.append(selected_feature)
        
        # Sort and display
        if sort_order == "Highest":
            top_songs = df.sort_values(by=selected_feature, ascending=False).head(10)
        else:
            top_songs = df.sort_values(by=selected_feature, ascending=True).head(10)
        
        st.dataframe(top_songs[display_cols])

# Helper functions
def get_feature_description(feature):
    """Return description for audio features"""
    descriptions = {
        'acousticness': 'A confidence measure from 0.0 to 1.0 of whether the track is acoustic.',
        'danceability': 'How suitable a track is for dancing based on tempo, rhythm stability, beat strength, and overall regularity.',
        'energy': 'A measure from 0.0 to 1.0 representing intensity and activity. Energetic tracks feel fast, loud, and noisy.',
        'instrumentalness': 'Predicts whether a track contains no vocals. Values above 0.5 represent instrumental tracks.',
        'liveness': 'Detects the presence of an audience in the recording. Higher values mean higher probability the track was performed live.',
        'loudness': 'The overall loudness of a track in decibels (dB). Values typically range between -60 and 0 db.',
        'speechiness': 'Detects spoken words in a track. Values above 0.66 indicate speech-only tracks.',
        'tempo': 'The overall estimated tempo of a track in beats per minute (BPM).',
        'valence': 'A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track.'
    }
    
    return descriptions.get(feature, "No description available for this feature.")

if __name__ == "__main__":
    main()
