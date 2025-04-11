import ast
import re
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, vstack, save_npz, load_npz
from sklearn.neighbors import NearestNeighbors


class DataProcessor:
    """
    DataProcessor is a class designed for processing and preparing game-related data for analysis and recommendations. 
    It provides methods for loading, cleaning, and transforming datasets, as well as generating recommendation matrices 
    and retrieving game recommendations.

    Attributes:
        data (pd.DataFrame): The loaded and processed dataset.
        combined_matrix (csr_matrix): The combined matrix used for recommendations.

    Methods:
        __init__(data_path: str, data_path_old: str | None = None) -> None:
            Initializes the DataProcessor class by loading and optionally cleaning the dataset.
        load_data(data_path: str) -> None:
            Loads data from a specified file path into the `self.data` attribute. Supports `.parquet`, `.csv`, and `.pkl` formats.
        create_cleaned_data(data_path: str) -> None:
            Cleans and processes the dataset to prepare it for further analysis or modeling. Saves the cleaned data to a specified file path.
        to_midpoint(range_str: str) -> float | None:
            Converts a range string in the format 'start - end' into its midpoint.
        clean_text_column(df: pd.DataFrame, columns: list) -> pd.DataFrame:
            Cleans and processes specified text columns in a DataFrame.
        combine_features(row: dict, features_to_combine: list) -> str:
            Combines multiple text-based features from a row into a single string of unique, sorted tags.
        calculate_user_attractiveness(df: pd.DataFrame, columns_to_combine: list | None = None, weights: dict | None = None) -> pd.DataFrame:
            Calculates user attractiveness based on specified columns and weights.
        calculate_game_age(df: pd.DataFrame) -> pd.DataFrame:
            Calculates the age of games in days and scales it to a range of 0-1.
        prepare_matrix_for_recommendations(data_path: str, data_path_old: str | None = None, weights: dict | None = None) -> None:
            Prepares a combined matrix for recommendations by processing and weighting textual and numerical features.
        get_recommendations(root_indices: np.ndarray, game_indicies: np.ndarray, n_recommendations: int = 10, n_neighbors: int = 50, weights: dict[str, float] = None) -> pd.DataFrame:
            Generates game recommendations based on a combination of nearest neighbors and weighted attributes.
    """
    
    def __init__(self, data_path: str, data_path_old: str | None = None) -> None:
        """
        Initializes the data processing class.

        Args:
            data_path (str): The path to the current data file.
            data_path_old (str | None, optional): The path to the old data file. 
                If provided, the old data is loaded first, cleaned, and then the 
                current data is loaded. Defaults to None.

        """
        if not data_path_old:
            self.load_data(data_path)
        else:
            self.load_data(data_path_old)
            self.create_cleaned_data(data_path)
            self.load_data(data_path)
            
    def load_data(self, data_path: str) -> None:
        """
        Load data from a specified file path into the `self.data` attribute.

        This method supports loading data from files with the following extensions:
        - `.parquet`: Loads data using `pandas.read_parquet`.
        - `.csv`: Loads data using `pandas.read_csv`.
        - `.pkl`: Loads data using `pandas.read_pickle`.

        Args:
            data_path (str): The file path to the data file.

        Raises:
            FileNotFoundError: If the specified file does not exist.
        """
        # Check if the file exists
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"The file {data_path} does not exist.")
        # Load data based on the file extension
        if re.match(r'.*\.parquet$', data_path):
            self.data = pd.read_parquet(data_path)
        elif re.match(r'.*\.csv$', data_path):
            self.data = pd.read_csv(data_path)
        elif re.match(r'.*\.pkl$', data_path):
            self.data = pd.read_pickle(data_path)
            
    def create_cleaned_data(self, data_path: str):
        """
        Cleans and processes the dataset to prepare it for further analysis or modeling.
        Args:
            data_path (str): The file path where the cleaned DataFrame will be saved as a pickle file.
        Returns:
            None
        Steps:
            1. Selects and copies relevant columns from the original dataset.
            2. Converts the 'release_date' column to datetime format.
            3. Converts the 'estimated_owners' column to numeric format using a midpoint calculation.
            4. Cleans text columns, excluding 'name' and 'header_image'.
            5. Combines 'categories', 'genres', 'tags', and 'developers' into a single 'tags_combined' column.
            6. Removes rows with empty 'tags_combined' values.
            7. Removes duplicate rows based on the 'name' column, keeping the row with the highest 'num_reviews_total'.
            8. Calculates a 'user_attractiveness' column as a weighted average of popularity and positive reviews.
            9. Calculates the age of each game and adds it as a new column.
            10. Combines 'windows', 'mac', and 'linux' columns into a single 'platforms' column.
            11. Drops unnecessary columns, including 'header_image', 'estimated_owners', 'pct_pos_recent', and 'num_reviews_recent'.
            12. Resets the DataFrame index and sorts it by 'user_attractiveness' and 'name'.
            13. Saves the cleaned DataFrame to the specified file path as a pickle file.
        """
        df = self.data.copy(deep=True)[[
            'name',
            'release_date',
            'detailed_description',
            'header_image',
            'windows',
            'mac',
            'linux',
            'supported_languages',
            'developers',
            'categories',
            'genres',
            'estimated_owners',
            'tags',
            'pct_pos_total',
            'num_reviews_total',
            'pct_pos_recent',
            'num_reviews_recent'
            ]]
        
        # Convert 'release_date' to datetime
        df['release_date'] = pd.to_datetime(df['release_date'], errors='raise', format='%Y-%m-%d')
        
        # Convert 'estimated_owners' to numeric
        df['estimated_owners'] = pd.to_numeric(df['estimated_owners'].apply(self.to_midpoint), errors='raise').astype('int64')
        
        # Clean text columns
        text_columns = [i for i in df.columns if df[i].dtype == 'object' if i not in ('name', 'header_image')]
        df = self.clean_text_column(df, text_columns)
        
        # Combine 'categories', 'genres', 'tags', 'developers' into one column
        features_to_combine = ('categories', 'genres', 'tags', 'developers')
        df['tags_combined'] = df.apply(lambda x: self.combine_features(row=x, features_to_combine=features_to_combine), axis=1)
        
        # Remove rows with empty 'tags_combined'
        df = df.loc[df['tags_combined'] != '']
        df = df[[i for i in df.columns if i not in features_to_combine]]
        
        # Remove duplicates
        df = df.sort_values(by='num_reviews_total', ascending=False).drop_duplicates(subset=['name'], keep='first')
        
        # Get users_attractiveness column (weighted average of popularity and positive reviews)
        df = self.calculate_user_attractiveness(df, columns_to_combine=['num_reviews_total', 'pct_pos_total'])
        
        # Calculate game_age
        df = self.calculate_game_age(df)
        
        # Convert 'windows', 'mac', 'linux' to a single 'platforms' column
        df['platforms'] = df[['windows', 'mac', 'linux']].apply(lambda row: np.array([int(i) for i in row]), axis=1)
        df.drop(columns=['windows', 'mac', 'linux'], inplace=True, axis=1)
        
        # Remove 'header_image' and other unnecessary columns
        df.drop(columns=['header_image', 'estimated_owners', 'pct_pos_recent', 'num_reviews_recent'], inplace=True, axis=1)
        
        # Reset index
        df.reset_index(drop=True, inplace=True)
        df.sort_values(by=['user_attractiveness', 'name'], ascending=False, inplace=True)
        
        df.to_pickle(data_path)
        
    @staticmethod
    def to_midpoint(range_str) -> (float | None):
        """
        Converts a range string in the format 'start - end' into its midpoint.

        Args:
            range_str (str): A string representing a range in the format 'start - end'.
                             If the input is null (NaN), it will be handled accordingly.

        Returns:
            float | None: The midpoint of the range as a float, or NaN if the input is invalid
                          or cannot be processed.
        """
        if pd.isnull(range_str):
            return np.nan
        try:
            start, end = map(int, range_str.split(' - '))
            return (start + end) / 2
        except ValueError:
            return np.nan
        
    @staticmethod
    def clean_text_column(df, columns) -> pd.DataFrame:
        """
        Cleans and processes specified text columns in a DataFrame.
        Args:
            df (pd.DataFrame): The input DataFrame containing the columns to be cleaned.
            columns (list): A list of column names to be processed.
        Returns:
            pd.DataFrame: The DataFrame with the specified columns cleaned and processed.
        Processing Details:
            - For the 'detailed_description' column:
                - Converts all text to lowercase.
            - For the 'tags' column:
                - Converts stringified dictionaries to lists of keys if the value is a valid string representation of a dictionary.
                - If the value is not a valid string or is an empty list, it assigns an empty list.
            - For other columns:
                - Converts stringified lists to actual lists if the value is a valid string representation of a list.
                - If the value is not a valid string or does not start with '[', it assigns an empty list.
            - For all processed columns:
                - Normalizes text by converting to lowercase, replacing spaces with underscores, and joining list elements into a single string.
        Notes:
            - This function assumes that the input DataFrame uses stringified lists or dictionaries in certain columns.
            - The function modifies the input DataFrame in place.
        """
        for column in columns:
            if column == 'detailed_description':
                df[column] = df[column].astype(str).str.lower()
                continue

            if column == 'tags':
                df[column] = df[column].apply(
                    lambda x: list(ast.literal_eval(x).keys()) if isinstance(x, str) and x != '[]' else []
                )
            else:
                # Convert stringified lists to actual lists
                df[column] = df[column].apply(
                    lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else []
                )

            # Normalize text: lowercase, replace spaces, join into one string
            df[column] = df[column].apply(
                lambda lst: ' '.join(i.lower().replace(' ', '_') for i in lst) if isinstance(lst, list) else ''
            )

        return df
    
    @staticmethod
    def combine_features(row, features_to_combine) -> str:
        """
        Combines multiple text-based features from a row into a single string of unique, sorted tags.

        Args:
            row (dict or pandas.Series): A dictionary-like object representing a row of data, 
                                         where keys are column names and values are strings.
            features_to_combine (list of str): A list of column names whose values will be combined.

        Returns:
            str: A single string containing unique, sorted tags separated by spaces.
        """
        all_tags = set()
        for col in features_to_combine:
            all_tags.update(map(str.strip, row[col].split(' ')))
        return ' '.join(sorted(all_tags))
    
    @staticmethod
    def calculate_user_attractiveness(df, columns_to_combine=None, weights=None) -> pd.DataFrame:
        """
        Calculate user attractiveness based on specified columns and weights.
        This function scales the specified columns in the input DataFrame using MinMaxScaler,
        combines them using the provided weights, and calculates a new column called 
        'user_attractiveness'. The original columns used for the calculation are dropped 
        from the DataFrame, and any missing values in the 'user_attractiveness' column 
        are filled with 0.
        Args:
            df (pd.DataFrame): The input DataFrame containing the data.
            columns_to_combine (list, optional): List of column names to be combined for 
                calculating user attractiveness. Defaults to ['num_reviews_total', 'pct_pos_total'].
            weights (dict, optional): Dictionary specifying the weights for each column 
                in `columns_to_combine`. Defaults to {'num_reviews_total': 0.5, 'pct_pos_total': 0.5}.
        Returns:
            pd.DataFrame: The modified DataFrame with the 'user_attractiveness' column added 
            and the original columns used for the calculation removed.
        """
        # Get weights for each column
        if weights is None:
            weights = {
                'num_reviews_total': 0.5,
                'pct_pos_total': 0.5,
            }
            
        # Get the columns to combine
        if columns_to_combine is None:
            columns_to_combine = ['num_reviews_total', 'pct_pos_total']
        
        # Scale the columns to be combined
        scaler = MinMaxScaler()
        scaled_values = scaler.fit_transform(df[columns_to_combine])
        scaled_df = pd.DataFrame(scaled_values, columns=[f'{col}_scaled' for col in columns_to_combine])
        
        # Calculate user attractiveness
        scaled_df['users_attractiveness'] = ((weights['num_reviews_total'] * scaled_df['num_reviews_total_scaled'] + weights['pct_pos_total'] * scaled_df['pct_pos_total_scaled'])
                                             / (weights['num_reviews_total'] + weights['pct_pos_total']))
        
        # Combine the scaled values with the original DataFrame and drop the scaled columns
        df['user_attractiveness'] = scaled_df['users_attractiveness']
        df.drop(columns=columns_to_combine, inplace=True, axis=1)
        df.fillna({'user_attractiveness': 0}, inplace=True)
        return df
    
    @staticmethod
    def calculate_game_age(df) -> pd.DataFrame:
        """
        Calculate the age of games in days and scale it to a range of 0-1.
        This method computes the age of each game based on its release date 
        and the current date. The calculated age is then scaled to a range 
        between 0 and 1 using MinMaxScaler. The original 'release_date' column 
        is removed from the DataFrame, and a new column 'game_age' is added 
        with the scaled values.
        Args:
            df (pd.DataFrame): A pandas DataFrame containing a 'release_date' 
                column with datetime values representing the release dates of games.
        Returns:
            pd.DataFrame: The modified DataFrame with the 'game_age' column added 
            and the 'release_date' column removed.
        """
        # Calculate the age based on the current date
        current_date = pd.Timestamp("today")
        game_age = (current_date - df['release_date']).dt.days
        
        # Scale the game age to a range of 0-1
        scaler = MinMaxScaler()
        game_age_scaled = scaler.fit_transform(game_age.values.reshape(-1, 1))
        df['game_age'] = game_age_scaled
        df.drop(columns=['release_date'], inplace=True, axis=1)
        return df
    
    def prepare_matrix_for_recommendations(self, data_path: str, data_path_old: str | None = None, weights=None) -> None:
        """
        Prepares a combined matrix for recommendations by processing and weighting
        textual and numerical features, and optionally loading from a pre-saved file.
        Args:
            data_path (str): The file path to save the combined matrix or load it if `data_path_old` is not provided.
            data_path_old (str | None, optional): The file path to an existing matrix to load. Defaults to None.
            weights (dict | None, optional): A dictionary of weights for different features. If not provided, default
                weights are used:
                - "tags": sqrt(0.45)
                - "desc": sqrt(0.45)
                - "game_age": sqrt(0.1)
        Returns:
            None: The combined matrix is saved to `data_path` and stored in `self.combined_matrix`.
        Notes:
            - If `data_path_old` is provided, the method loads the matrix from `data_path` and skips processing.
            - Textual features (`detailed_description` and `tags_combined`) are vectorized using TF-IDF.
            - Numerical features (`game_age`) are weighted and combined with textual features.
            - The resulting matrix is saved in Compressed Sparse Row (CSR) format.
        """
        if not data_path_old:
            self.combined_matrix = load_npz(data_path)
            return
        
        if weights is None:
            weights = {
                # Similarity
                "tags": np.sqrt(0.45),
                "desc": np.sqrt(0.45),
                "game_age": np.sqrt(0.1),
            }
        df = self.data
        # Vectorize textual features with TF-IDF
        tfidf_desc = TfidfVectorizer(stop_words='english')
        tfidf_tags = TfidfVectorizer()

        desc_matrix = tfidf_desc.fit_transform(df['detailed_description'])
        tags_matrix = tfidf_tags.fit_transform(df['tags_combined'])
        
        # Get age matrix
        game_age_matrix = df[['game_age']].values
        
        # Apply weights
        desc_matrix = desc_matrix.multiply(weights['desc'])
        tags_matrix = tags_matrix.multiply(weights['tags'])
        game_age_matrix_weighted = game_age_matrix * weights['game_age']
        
        # Combine all into one matrix
        combined_matrix = hstack([
            tags_matrix,
            desc_matrix,
            game_age_matrix_weighted,
        ])
        combined_matrix = combined_matrix.tocsr()
        save_npz(data_path, combined_matrix)
        self.combined_matrix = combined_matrix
    
    def get_recommendations(
        self,
        root_indices: np.ndarray[np.int64],
        game_indicies: np.ndarray[np.int64], 
        n_recommendations: int = 10,
        n_neighbors: int = 50,
        weights: dict[str, float] = None
    ) -> pd.DataFrame:
        """
        Generates game recommendations based on a combination of nearest neighbors and weighted attributes.

        Args:
            root_indices (np.ndarray): Indices of the root games to base the recommendations on.
            game_indicies (np.ndarray): Indices of the games for which recommendations are sought.
            n_recommendations (int, optional): The number of recommendations to return. Defaults to 10.
            n_neighbors (int, optional): The number of nearest neighbors to consider for each input game. Defaults to 50.
            weights (dict, optional): A dictionary containing the weights for different attributes used to boost similarity scores. 
                                    Defaults to None, in which case predefined weights are used.
                                    - 'user_attractiveness' (float): The weight for boosting recommendations based on user attractiveness.
                                    - 'platforms' (float): The weight for boosting recommendations based on platform popularity.

        Returns:
            pd.DataFrame: A DataFrame containing the top game recommendations and their corresponding similarity scores.
            
        Raises:
            ValueError: If no valid game indices are provided that match the dataset.

        The function performs the following steps:
        1. Validates the provided game indices to ensure they are present in the dataset.
        2. Initializes default weights if none are provided.
        3. Extracts details for the selected games and determines the most common platform.
        4. Filters the combined matrix based on the root games and input game indices.
        5. Uses a nearest neighbor model to compute the similarity between input games and root games.
        6. Applies cosine distance to compute similarity scores, adjusting them based on 'user_attractiveness' and platform similarity.
        7. Sorts the games by similarity score and limits the results to the specified number of recommendations.
        8. Scales the final similarity scores to a range between 0 and 1.
        9. Returns a DataFrame with the recommended games and their similarity scores.
        
        Note:
            - The 'user_attractiveness' and 'platforms' weights can be adjusted to customize how these factors influence the final recommendations.
            - The nearest neighbors model uses the cosine distance metric for measuring similarities between games.
        """
        if len(game_indicies) == 0:
            raise ValueError("None of the provided game names match the dataset.")
        
        df = self.data
        combined_matrix = self.combined_matrix
        # Default weights using square root normalization
        if weights is None:
            weights = {
                # After nearest neighbors
                "user_attractiveness": 0.25,
                'platforms': 0.1,
            }
        
        # Get details about the selected games
        games = df.loc[game_indicies]
        
        # Extract the most common plaform
        game_most_popular_platform = pd.DataFrame(games['platforms'].tolist()).sum().idxmax()
        
        # Get the right matrix for the selected games
        combined_matrix_filtered = combined_matrix[root_indices]

        # Get input vectors to get recommendations for
        input_vectors = combined_matrix[game_indicies]

        # Check if all input vectors are in filtered matrix
        mask_matrix = ~np.isin(game_indicies, root_indices)
        missing_indices = game_indicies[mask_matrix]
        if missing_indices.size > 0:
            fit_matrix = vstack([combined_matrix_filtered, combined_matrix[missing_indices]])
            mapping_indices = np.concatenate([root_indices, missing_indices])
        else:
            fit_matrix = combined_matrix_filtered
            mapping_indices = root_indices

        # Fit Nearest Neighbors
        nn_model = NearestNeighbors(metric='cosine')
        nn_model.fit(fit_matrix)

        # Compute neighbors for multiple input vectors directly
        distances, indices = nn_model.kneighbors(input_vectors, n_neighbors=min(len(mapping_indices), n_neighbors + len(game_indicies)))
        indices = mapping_indices[indices[0]] # Get the original indices of the neighbors
        distances = distances[0]
        
        # Remove the games itself from the results
        mask_result = ~np.isin(indices, game_indicies)
        filtered_indices = indices[mask_result]
        filtered_distances = distances[mask_result]
        
        # Compute cosine similarity scores (Cosine similarity: 1 - distance)
        base_similarities = 1 - filtered_distances
        
        # Boost similarities based on user_attractiveness
        user_attractiveness = df.loc[filtered_indices, 'user_attractiveness'].values
        base_similarities += user_attractiveness * weights['user_attractiveness']
        
        # Boost similarities based on platforms
        platforms = df.loc[filtered_indices, 'platforms'].values
        base_similarities *= np.array([1 + weights['platforms'] if i[game_most_popular_platform] else 1. for i in platforms])
        
        # Sort by similarity score
        indices_scores = np.array(sorted(zip(filtered_indices, base_similarities), key=lambda x: x[1], reverse=True))

        # Limit to the top n recommendations
        top_indices = indices_scores[:n_recommendations]
        
        # Scale the final similarities to be between 0 and 1
        top_indices[:, 1] = np.clip(top_indices[:, 1], 0, 1)
        
        # Retrieve the recommendations from the DataFrame and add similarity scores
        results = df.loc[top_indices[:, 0]].copy()
        results['similarity'] = top_indices[:, 1]

        return results[['name', 'similarity']].reset_index(drop=True)
    