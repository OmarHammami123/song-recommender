# Data Directory

This directory contains the datasets used by the Song Recommender system.

## Structure

```bash
data/
├── raw/
│   └── dataset.csv         # Original song dataset with audio features
├── processed/
│   └── feature_matrix.csv  # Processed features for recommendations
└── explore.ipynb          # Data exploration and processing notebook
```

## Data Acquisition

### Prerequisites

1. **Kaggle Account**: Sign up at [kaggle.com](https://www.kaggle.com/)
2. **Kaggle API Setup**: Install the Kaggle API client
   ```bash
   pip install kaggle
   ```

### Step 1: Configure Kaggle API

1. Go to your Kaggle account settings: [https://www.kaggle.com/settings/account](https://www.kaggle.com/settings/account)
2. Scroll down to the "API" section
3. Click "Create New API Token" - this downloads `kaggle.json`
4. Place the file in the appropriate location:
   - **Windows**: `C:\Users\<username>\.kaggle\kaggle.json`
   - **Linux/macOS**: `~/.kaggle/kaggle.json`
5. Set file permissions (Linux/macOS only):
   ```bash
   chmod 600 ~/.kaggle/kaggle.json
   ```

### Step 2: Environment Configuration

1. Copy the environment template:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` file with your Kaggle credentials:
   ```bash
   KAGGLE_USERNAME=your_kaggle_username
   KAGGLE_KEY=your_kaggle_api_key
   ```

### Step 3: Download Recommended Dataset

We recommend using the **Spotify Dataset 1921-2020** which contains 160k+ tracks with comprehensive audio features:

```bash
# Download the dataset
kaggle datasets download -d yamaerenay/spotify-dataset-19212020-160k-tracks

# Extract to the raw directory
unzip spotify-dataset-19212020-160k-tracks.zip -d data/raw/

# Rename the main file to match our expected format
mv data/raw/data.csv data/raw/dataset.csv
```

### Alternative: Manual Download

If you prefer manual download:

1. Visit: [https://www.kaggle.com/datasets/yamaerenay/spotify-dataset-19212020-160k-tracks](https://www.kaggle.com/datasets/yamaerenay/spotify-dataset-19212020-160k-tracks)
2. Click "Download" button
3. Extract the ZIP file
4. Move `data.csv` to `data/raw/dataset.csv`

### Step 4: Verify Data Format

The dataset should have these essential columns:
- `track_name`: Song title
- `artists`: Artist name(s)
- `danceability`: How suitable for dancing (0.0-1.0)
- `energy`: Intensity and activity level (0.0-1.0)
- `acousticness`: Acoustic quality (0.0-1.0)
- `instrumentalness`: Instrumental content (0.0-1.0)
- `liveness`: Live performance likelihood (0.0-1.0)
- `loudness`: Overall loudness in dB
- `speechiness`: Spoken word content (0.0-1.0)
- `tempo`: Beats per minute
- `valence`: Musical positiveness (0.0-1.0)

### Step 5: Process the Data

Run the data processing notebook:
```bash
jupyter notebook data/explore.ipynb
```

Or use the Python script:
```bash
python -c "
import pandas as pd
from models.recommender import SongRecommender

# Load and process data
df = pd.read_csv('data/raw/dataset.csv')
print(f'Loaded {len(df)} songs')

# The notebook handles the rest of the processing
print('Run the explore.ipynb notebook to complete data processing')
"
```

## Alternative Datasets

### Option 1: Million Song Dataset (Subset)
```bash
# Download subset (10k songs)
kaggle datasets download -d notshrirang/million-song-dataset

# Extract and rename
unzip million-song-dataset.zip -d data/raw/
mv data/raw/song_data.csv data/raw/dataset.csv
```

### Option 2: Last.fm Dataset
```bash
# Download Last.fm dataset
kaggle datasets download -d pcbreviglieri/lastfm-music-artist-song-dataset

# Extract and rename
unzip lastfm-music-artist-song-dataset.zip -d data/raw/
mv data/raw/lastfm_dataset.csv data/raw/dataset.csv
```

## Data Schema Requirements

Your CSV file must include these columns (case-sensitive):

| Column | Type | Description | Required |
|--------|------|-------------|----------|
| `track_name` | string | Song title | ✅ |
| `artists` | string | Artist name(s) | ✅ |
| `danceability` | float | 0.0-1.0 | ✅ |
| `energy` | float | 0.0-1.0 | ✅ |
| `acousticness` | float | 0.0-1.0 | ✅ |
| `instrumentalness` | float | 0.0-1.0 | ✅ |
| `liveness` | float | 0.0-1.0 | ✅ |
| `loudness` | float | dB values | ✅ |
| `speechiness` | float | 0.0-1.0 | ✅ |
| `tempo` | float | BPM | ✅ |
| `valence` | float | 0.0-1.0 | ✅ |
| `popularity` | int | 0-100 | ❌ |
| `track_genre` | string | Genre labels | ❌ |

## Troubleshooting

### Common Issues

1. **Kaggle API not found**:
   ```bash
   pip install --upgrade kaggle
   ```

2. **Permission denied**:
   ```bash
   # Windows (run as administrator)
   icacls C:\Users\<username>\.kaggle\kaggle.json /inheritance:r /grant:r "%USERNAME%":F

   # Linux/macOS
   chmod 600 ~/.kaggle/kaggle.json
   ```

3. **Dataset not found**:
   - Verify the dataset URL is correct
   - Check if the dataset is still available
   - Ensure your Kaggle account has access

4. **Large file handling**:
   ```bash
   # For datasets larger than 100MB, use streaming
   python -c "
   import pandas as pd
   chunks = pd.read_csv('data/raw/dataset.csv', chunksize=10000)
   for chunk in chunks:
       print(f'Processing chunk with {len(chunk)} rows')
       break
   "
   ```

### Data Validation

After downloading, validate your dataset:

```bash
python -c "
import pandas as pd
import numpy as np

# Load and validate
df = pd.read_csv('data/raw/dataset.csv')
print(f'Dataset shape: {df.shape}')
print(f'Missing values: {df.isnull().sum().sum()}')

# Check required columns
required_cols = ['track_name', 'artists', 'danceability', 'energy', 'acousticness']
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    print(f'Missing required columns: {missing_cols}')
else:
    print('✅ All required columns present')

# Check data types
print(f'Data types: {df.dtypes}')
"
```

## Security Notes

- Never commit your `.env` file to version control
- Keep your Kaggle API key secure
- The `.gitignore` file already excludes sensitive files
- Consider using environment variables in production

## Next Steps

Once you have your data:
1. Run `jupyter notebook data/explore.ipynb`
2. Execute all cells to process the data
3. Launch the app: `streamlit run app.py`
4. Start exploring music recommendations!
