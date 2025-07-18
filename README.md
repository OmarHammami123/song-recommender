# Song Recommender System 🎵

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.30.0+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-brightgreen.svg)

A sophisticated music recommendation system built with Streamlit and machine learning that suggests songs based on audio features and similarity analysis.

## 🎯 Features

- **🎵 Content-Based Recommendations**: Discover songs with similar audio characteristics
- **📊 Interactive Visualizations**: Explore audio features with radar charts and comparisons
- **🎯 Custom Feature Matching**: Find songs matching your preferred audio qualities
- **🎪 Multi-Page Interface**: Clean, intuitive navigation between different features
- **⚡ Memory-Efficient Design**: Handles large datasets without performance issues
- **🎨 Professional UI**: Custom styling with Spotify-inspired design

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Jupyter Notebook (for data processing)
- Kaggle account (for dataset access)

### 1. Clone & Setup

```bash
# Clone the repository
git clone https://github.com/OmarHammami123/song-recommender.git
cd song-recommender

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Acquisition

1. Visit [Spotify Dataset on Kaggle](https://www.kaggle.com/datasets/yamaerenay/spotify-dataset-19212020-160k-tracks)
2. Download and extract to `data/raw/`
3. Rename main file to `dataset.csv`

### 3. Data Processing

```bash
# Launch Jupyter notebook
jupyter notebook data/explore.ipynb

# Run all cells to process the data
# This creates: data/processed/feature_matrix.csv
```

### 4. Launch Application

```bash
# Start the Streamlit app
streamlit run app.py
```

## 📊 How It Works

The system uses **cosine similarity** to find songs with similar audio characteristics:

1. **Feature Extraction**: Normalize audio features (0-1 scale)
2. **Similarity Calculation**: Compute cosine similarity between feature vectors
3. **Memory-Efficient Processing**: Process datasets without loading full similarity matrices
4. **Ranking**: Sort by similarity score and return top matches

## 🛠️ Project Structure

```
song-recommender/
├── app.py                    # Main Streamlit application
├── config.py                 # Configuration settings
├── requirements.txt          # Python dependencies
├── pages/                    # Multi-page components
│   ├── explore.py           # Data exploration
│   └── recommendations.py   # Recommendation engine
├── models/
│   └── recommender.py       # Core ML logic
├── assets/
│   └── styles.css           # Custom styling
└── data/
    ├── explore.ipynb        # Data processing notebook
    ├── raw/                 # Raw dataset storage
    └── processed/           # Processed data
```

## 🚀 Deployment

### Streamlit Cloud
1. Push to GitHub
2. Connect at [share.streamlit.io](https://share.streamlit.io)
3. Deploy directly from repository

## 📧 Contact

- **Author**: OmarHammami123
- **Email**: [omarhammami090@gmail.com](mailto:omarhammami090@gmail.com)
- **Repository**: [https://github.com/OmarHammami123/song-recommender](https://github.com/OmarHammami123/song-recommender)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Spotify Web API** for audio feature definitions
- **Kaggle Community** for providing music datasets
- **Streamlit Team** for the amazing web framework
- **scikit-learn** for machine learning utilities

⭐ **Star this repository** if you find it helpful!
