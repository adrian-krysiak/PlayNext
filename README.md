# PlayNext

PlayNext is a recommendation system designed to help users discover their next favorite game. It leverages data processing, machine learning, and a user-friendly Streamlit interface to provide personalized game recommendations.

## Project Structure

```
PlayNext/
├── data/                          # Data-related files
│   ├── games_march2025_cleaned.csv
│   └── processed/                 # Processed data files
│       ├── games_march2025_cleaned_ready_ml_mx.npz
│       ├── games_march2025_cleaned_ready_ml.pkl
│       └── games_march2025_cleaned_ready_ui.pkl
├── notebooks/                     # Jupyter notebooks for analysis or experimentation
│   └── notebook.ipynb
├── src/                           # Source code for the application
│   ├── __init__.py
│   ├── app/                       # Streamlit app code
│   │   ├── __init__.py
│   │   └── streamlit.py
│   ├── core/                      # Core logic and utilities
│   │   ├── __init__.py
│   │   ├── data_downloader.py
│   │   ├── data_process.py
│   └── runner.py                  # Entry point for running the app
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/adrian-krysiak/PlayNext.git
   cd PlayNext
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Run the Streamlit app**:
   ```bash
   python src/runner.py
   ```

2. Open the app in your browser at `http://localhost:8501`.

## Demo

Here are a few screenshots showcasing the Streamlit interface:

### 🔹 Start Page
![Start Page](https://github.com/user-attachments/assets/9a685526-3c31-4de0-89fa-2427d6fd64f2)


### 🔹 Filters Chosen
![Filters Chosen](https://github.com/user-attachments/assets/8fd06475-2b23-426e-a491-26ddd133a4d9)


### 🔹 Recommendations Received
![Recommendations](https://github.com/user-attachments/assets/c0f40a80-8685-4e7a-889a-29a42835d2d4)


## Features

- **Data Processing**: Cleans, preprocesses, and weights game data for analysis and recommendations.
- **Recommendation System**: Suggests games based on user preferences using nearest neighbors and weighted attributes.
- **Streamlit Interface**: Provides an interactive and user-friendly web application.

## Methodology

The recommendation system in PlayNext utilizes a combination of data processing techniques and machine learning methodologies to suggest relevant games.

### Data Processing
- The raw game data is cleaned and preprocessed to ensure consistency and usability.
- **TF-IDF (Term Frequency-Inverse Document Frequency)** is applied to vectorize textual features such as game descriptions, tags, and genres. This technique assigns weights to words based on their importance within a document relative to the entire dataset, downplaying common words and highlighting unique ones.
- Numerical features, specifically 'game_age', are also considered.
- Both textual and numerical features are then **weighted** to control their influence on the similarity calculations. Default weights are applied as follows:
    - "tags": √0.45 ≈ 0.67
    - "desc": √0.45 ≈ 0.67
    - "game_age": √0.1 ≈ 0.32
- The weighted textual and numerical features are then combined into a comprehensive feature matrix in Compressed Sparse Row (CSR) format for efficient computation.

### Recommendation Engine
- The recommendation engine employs a **Nearest Neighbors** model with the **cosine distance** metric to find games similar to a given set of "root games" based on their combined features. Cosine distance measures the dissimilarity between two vectors, and the similarity is derived as 1 - cosine distance.
- The `get_recommendations` function takes a set of "root games" and "input games" to find recommendations for the latter based on the neighbors of the former.
- **Similarity Boosting**: After identifying the nearest neighbors, the similarity scores are further adjusted based on:
    - **User Attractiveness**: A weight (`user_attractiveness`, default 0.25) is added to the base similarity score based on the 'user_attractiveness' of the recommended games.
    - **Platform Popularity**: A weight (`platforms`, default 0.1) is applied multiplicatively if a recommended game is available on the **most popular platform** among the initially selected "input games".
- Finally, the resulting similarity scores are scaled to a range between 0 and 1.

### Key Steps
1. **TF-IDF Vectorization**: Textual features ('detailed_description' and 'tags_combined') are transformed into numerical vectors using TF-IDF.
2. **Feature Weighting**: Weights are applied to the TF-IDF vectors and the numerical 'game_age' feature.
3. **Feature Combination**: The weighted textual and numerical features are combined into a single sparse matrix.
4. **Nearest Neighbors Search**: A Nearest Neighbors model using cosine distance is fitted on a matrix potentially including both "root games" and "input games" to find similar games.
5. **Similarity Calculation**: Cosine similarity scores are calculated from the distances obtained from the Nearest Neighbors model.
6. **Similarity Boosting**: The initial similarity scores are boosted based on 'user_attractiveness' and platform popularity.
7. **Ranking and Filtering**: The games are ranked based on their boosted similarity scores, and the top N recommendations (default 10) are selected, excluding the input games themselves.
8. **Score Scaling**: The final similarity scores are scaled to be between 0 and 1.
9. **Output**: A DataFrame containing the recommended games and their final similarity scores is returned.

## Dependencies

The project uses the following Python libraries:
- `jupyter`
- `kagglehub`
- `pandas`
- `scikit-learn`
- `scipy`
- `streamlit`

Refer to `requirements.txt` for the full list of dependencies.
**Python version**: 3.13.2

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

- Data originally sourced from Kaggle: [Steam Games Dataset by artermiloff](https://www.kaggle.com/datasets/artermiloff/steam-games-dataset)
