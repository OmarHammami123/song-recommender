import streamlit as st 
import pandas as pd
import os
from models.recommender import SongRecommender
from config import DATA_PATH, UI_CONFIG, EXAMPLE_SONGS

# Add custom CSS
try:
    with open("assets/styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except Exception:
    st.write("Note: Custom styling not loaded")

# Page configuration
st.set_page_config(
    page_title = UI_CONFIG['page_title'],
    page_icon = UI_CONFIG['page_icon'],
    layout = "wide",
)

# Load the recommender
@st.cache_resource
def load_recommender():
    feature_matrix_path = DATA_PATH['PROCESSED']
    if not os.path.exists(feature_matrix_path):
        st.error("Feature matrix file not found. Please ensure the data is processed correctly.")
        st.info("Try running the data processing notebook: data/explore.ipynb")
        return None
    return SongRecommender(feature_matrix_path)

recommender = load_recommender()

# App header
st.title("Song Recommender 🎵")

# App introduction
st.markdown("""
Welcome to the Song Recommender app! This tool helps you discover new music 
based on songs you already enjoy or specific audio characteristics you prefer.

### What can you do here?
""")

# Main page navigation cards
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🔍 Quick Search")
    st.markdown("Find songs and get instant recommendations")
    if st.button("Start Here", key="quick_search", use_container_width=True):
        st.session_state.show_search = True

with col2:
    st.markdown("### 📊 Explore Data")
    st.markdown("Discover patterns in music features")
    if st.button("Explore Data", key="explore_data", use_container_width=True):
        # Navigate to explore page
        st.switch_page("pages/explore.py")

with col3:
    st.markdown("### 🎵 Advanced Recommendations")
    st.markdown("Create playlists and fine-tune your search")
    if st.button("Advanced Features", key="advanced_features", use_container_width=True):
        # Navigate to recommendations page
        st.switch_page("pages/recommendations.py")

# Quick search section (shown by default or when button clicked)
if 'show_search' not in st.session_state:
    st.session_state.show_search = True
    
if st.session_state.show_search:
    st.markdown("---")
    st.header("Quick Song Search")
    st.markdown("Search for a song and get instant recommendations")
    
    # Search box
    col1, col2 = st.columns([3, 1])
    with col1:
        search_query = st.text_input("Enter a song name or artist", placeholder="Example: Shape of You")
    with col2:
        st.write("")
        st.write("")
        search_button = st.button("Search", use_container_width=True)
    
    # Example songs for quick access
    st.markdown("#### Popular examples:")
    example_cols = st.columns(len(EXAMPLE_SONGS))
    for i, example in enumerate(EXAMPLE_SONGS):
        with example_cols[i]:
            if st.button(example, key=f"example_{i}"):
                st.session_state.search_query = example
                st.experimental_rerun()
    
    # Process search
    if search_query or ('search_query' in st.session_state and st.session_state.search_query):
        # Use session state if available
        if 'search_query' in st.session_state:
            search_query = st.session_state.search_query
            # Clear after use
            st.session_state.search_query = ""
            
        with st.spinner("Searching..."):
            if recommender:
                results = recommender.search_songs(search_query, limit=10)
                if results:
                    st.success(f"Found {len(results)} songs matching '{search_query}'")
                    
                    # Display results
                    st.markdown("### Select a song to get recommendations:")
                    selected_song = st.selectbox(
                        "Select a song",
                        options=range(len(results)),
                        format_func=lambda i: f"{results[i]['track_name']} by {results[i]['artists']}"
                    )
                    
                    if st.button("Get Recommendations"):
                        # Redirect to recommendations page with parameters
                        st.session_state.selected_song = results[selected_song]['track_name']
                        st.session_state.selected_artist = results[selected_song]['artists']
                        st.switch_page("pages/recommendations.py")
                else:
                    st.error(f"No songs found matching '{search_query}'. Try another search.")
            else:
                st.error("Recommender system not available. Please check if data processing is complete.")

# Add "what's possible" section
st.markdown("---")
st.header("Features")

feature_col1, feature_col2, feature_col3 = st.columns(3)

with feature_col1:
    st.markdown("### 🎵 Content-Based Recommendations")
    st.markdown("""
    * Get song recommendations based on audio features
    * Discover music with similar acoustic qualities
    * Find hidden gems that match your taste
    """)

with feature_col2:
    st.markdown("### 📊 Audio Feature Analysis")
    st.markdown("""
    * Visualize song characteristics
    * Compare features like energy and danceability
    * Understand what makes songs similar
    """)

with feature_col3:
    st.markdown("### 🎯 Custom Discovery")
    st.markdown("""
    * Define your preferred audio qualities
    * Generate playlists based on your criteria
    * Explore music across different styles
    """)

# Footer
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.caption("This recommender uses audio features from songs to find similar music using cosine similarity.")
    st.caption("Data processed from music dataset with features like tempo, energy, and danceability.")

with col2:
    st.caption("**Navigation:**")
    st.caption("• [Home](/) • [Explore Data](/explore) • [Advanced Recommendations](/recommendations)")

# Check if we need to show a data processing warning
if recommender is None:
    st.warning("""
    ## Data Processing Required
    
    It looks like you need to process your music dataset before using the recommender.
    
    Please run the data exploration notebook first:
    ```
    jupyter notebook data/explore.ipynb
    ```
    
    Then come back to this app!
    """)
