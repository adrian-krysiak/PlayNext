import os
import ast
import streamlit as st
import pandas as pd
import sys

# Set PlayNext as root for imports
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from core.data_process import DataProcessor

DATA_MAIN_PATH = os.path.join('data', 'games_march2025_cleaned.csv')
DATA_UI_PATH = os.path.join('data', 'processed', 'games_march2025_cleaned_ready_ui.pkl')
DATA_ML_PATH = os.path.join('data', 'processed', 'games_march2025_cleaned_ready_ml.pkl')
DATA_ML_MX_PATH = os.path.join('data', 'processed', 'games_march2025_cleaned_ready_ml_mx.npz')


st.set_page_config(
    page_title="PlayNext",
    page_icon=":game_die:",
    layout="wide",
)

def process_data(ml_data: pd.DataFrame, main_data_path: str) -> pd.DataFrame:
    """
    Processes and filters game data from a main dataset and aligns it with machine learning data.
    Args:
        ml_data (pd.DataFrame): A DataFrame containing machine learning data with a 'name' column.
        main_data_path (str): The file path to the main dataset CSV file.
    Returns:
        pd.DataFrame: A processed DataFrame containing relevant game data aligned with the machine learning data.
    The function performs the following steps:
    1. Loads the main dataset from the specified CSV file.
    2. Converts the 'release_date' column to datetime and sorts the data by 'num_reviews_total' in descending order.
    3. Removes duplicate entries based on the 'name' column, keeping only the first occurrence.
    4. Filters the dataset to include only relevant columns and games present in the `ml_data` DataFrame.
    5. Replaces invalid or missing values in specific columns with `pd.NA`.
    6. Processes the 'tags' column to extract the top 3 tags and creates a set of all tags.
    7. Creates a 'review_bucket' column based on the 'num_reviews_total' values and reorganizes columns.
    8. Converts the 'developers' and 'supported_languages' columns to sets if they contain valid string representations.
    9. Aligns the processed data with the `ml_data` DataFrame based on the 'name' column.
    10. Saves the processed data to a pickle file for later use.
    Note:
        - The function assumes the 'tags', 'developers', and 'supported_languages' columns contain stringified lists or dictionaries.
        - The 'bucket_reviews' function is used to categorize the 'num_reviews_total' values into buckets.
        - The processed data is saved to a pickle file at the path specified by `DATA_UI_PATH`.
    """
    # Load main data from the CSV file
    data = pd.read_csv(main_data_path)
    
    # Define relevant column categories
    id_columns = ['name', 'header_image']
    about_game_columns = ['tags', 'release_date', 'metacritic_score']
    platforms = ['windows', 'mac', 'linux']
    other_columns = ['developers', 'num_reviews_total', 'short_description', 'supported_languages']
    
    # Convert release date to datetime and sort by number of reviews (descending)
    data['release_date'] = pd.to_datetime(data['release_date'], errors='raise', format='%Y-%m-%d').dt.date
    data.sort_values(by='num_reviews_total', ascending=False, inplace=True)
    
    # Drop duplicates based on the 'name' column, keeping only the first entry
    data.drop_duplicates(subset=['name'], keep='first', inplace=True)
    
    # Select the relevant columns
    data = data[id_columns + about_game_columns + platforms + other_columns]
    
    # Filter data to include only games that are in the ml_data DataFrame
    data = data[data['name'].isin(ml_data['name'])]
    
    # Replace metacritic score 0 values with missing data (pd.NA)
    data = data.replace({'metacritic_score': {0: pd.NA}})
    
    # Reset index of the DataFrame
    data.reset_index(drop=True, inplace=True)
    
    # Process the 'tags' column to convert it to a list of keys (if it's a valid string)
    data['tags'] = data['tags'].apply(
        lambda x: list(ast.literal_eval(x).keys()) if isinstance(x, str) and x != '[]' else pd.NA
    )
    
    # Create a 'top_tags' column with the top 3 tags from the list or dictionary
    data['top_tags'] = data['tags'].apply(
        lambda x: x[:3] if isinstance(x, list) else [tag for tag, _ in sorted(x.items(), key=lambda item: item[1], reverse=True)[:3]]
        if isinstance(x, dict) else pd.NA
    )
    
    # Create a 'all_tags' column as a set of tags
    data['all_tags'] = data['tags'].apply(lambda x: set(x) if isinstance(x, list) else set())
    
    # Set 'tags' to only contain the top 3 tags and drop 'top_tags'
    data['tags'] = data['top_tags']
    data.drop(columns=['top_tags'], inplace=True)
    
    # Replace -1 in 'num_reviews_total' with missing data (pd.NA)
    data.replace({'num_reviews_total': {-1: pd.NA}}, inplace=True)
    
    # Create a new column 'review_bucket' based on the 'num_reviews_total' values
    data['review_bucket'] = data['num_reviews_total'].apply(bucket_reviews)
    
    # Reorganize columns: move 'review_bucket' to the position of 'num_reviews_total' and drop 'num_reviews_total'
    cols = data.columns.tolist()
    idx = cols.index('num_reviews_total')
    data.drop(columns=['num_reviews_total'], inplace=True)
    cols.remove('review_bucket')
    cols[idx] = 'review_bucket'
    data = data[cols]
    
    # Process the 'developers' column to convert it to a set (if it's a valid string)
    data['developers'] = data['developers'].apply(
        lambda x: set(ast.literal_eval(x)) if isinstance(x, str) and x != '[]' else pd.NA
    )
    
    # Process the 'supported_languages' column to convert it to a set (if it's a valid string)
    data['supported_languages'] = data['supported_languages'].apply(
        lambda x: set(ast.literal_eval(x)) if isinstance(x, str) and x != '[]' else pd.NA
    )
    
    # Set 'name' as the index in both dataframes to align them
    data = data.set_index('name')
    ml_data = ml_data.set_index('name')

    # Reindex 'data' to match 'ml_data' and reset both indices
    data = data.reindex(ml_data.index).reset_index()
    ml_data = ml_data.reset_index()
    
    # Save the processed data to a pickle file for later use
    data.to_pickle(DATA_UI_PATH)
    
    # Return the processed data
    return data

def bucket_reviews(val: int) -> str:
    """
    Categorizes a numerical value into buckets based on its range.
    Parameters:
        val (int): The numerical value to be categorized. Can be NaN.
    Returns:
        str: A category label as a string:
            - Returns `pd.NA` if the input is NaN.
            - Returns '1' if the value is less than 100.
            - Returns '2' if the value is between 100 (inclusive) and 1,000 (exclusive).
            - Returns '3' if the value is between 1,000 (inclusive) and 10,000 (exclusive).
            - Returns '4' if the value is between 10,000 (inclusive) and 100,000 (exclusive).
            - Returns '5' if the value is between 100,000 (inclusive) and 1,000,000 (exclusive).
            - Returns '6' if the value is 1,000,000 or greater.
    """

    if pd.isna(val):
        return pd.NA
    elif val < 100:
        return 1
    elif val < 1_000:
        return 2
    elif val < 10_000:
        return 3
    elif val < 100_000:
        return 4
    elif val < 1_000_000:
        return 5
    else:
        return 6
    
def extract_unique_values_from_column(series: pd.Series) -> set:
    """
    Extract unique values from a pandas Series containing lists or strings.

    This function processes a pandas Series where each element may be a list or a string.
    It explodes the Series to flatten any lists, removes NaN values, strips whitespace
    from strings, and returns a set of unique values.

    Args:
        series (pd.Series): A pandas Series containing lists, strings, or NaN values.

    Returns:
        set: A set of unique values extracted from the Series.
    """
    return set(series.explode().dropna().apply(lambda x: x.strip()))

@st.dialog("Recommended Games", width='large')
def show_recommended_games(game_include: list[str], game_exclude: list[str], data: pd.DataFrame) -> None:
    """
    Displays a dialog with recommended games based on user-selected inclusion and exclusion criteria.
    Parameters:
        game_include (list[str]): A list of game names to include in the recommendation process.
        game_exclude (list[str]): A list of game names to exclude from the recommendation process.
        data (pd.DataFrame): A DataFrame containing game data, including names and other attributes.
    Returns:
        None: This function does not return a value. It displays a dialog with the recommended games.
    Behavior:
        - Filters the main dataset based on the inclusion and exclusion criteria.
        - Computes the indices of the games to include and exclude.
        - If no games match the criteria, displays a message indicating no games were found.
        - Otherwise, uses a data processor to find recommended games based on the filtered indices.
        - Displays the recommended games in a styled Streamlit DataFrame, including additional game details such as name, cover image, tags, and platform compatibility.
    """
    include_set = set(game_include)
    exclude_set = set(game_exclude)
    main_data = st.session_state.data
    
    game_names = include_set - exclude_set
    game_indicies = main_data[main_data['name'].isin(game_names)].index.to_numpy()
    
    root_indices = data[~data['name'].isin(game_exclude)].index.to_numpy()
    
    if game_indicies.size == 0 or root_indices.size == 0:
        st.write("No games found based on your selections.")
    else:
        with st.spinner("Finding recommendations...", show_time=True):
            recommended_games = st.session_state.data_processor.get_recommendations(
                root_indices=root_indices,
                game_indicies=game_indicies,
            )
            recommended_games_full = pd.merge(recommended_games, main_data[['name', 'header_image', 'tags', 'windows', 'mac', 'linux']], on='name', how='left')
            # Index starts from 1
            recommended_games_full.index = recommended_games_full.index + 1
        st.dataframe(recommended_games_full,
                     use_container_width=True,
                     hide_index=False,
                     column_config= {
                         'name': st.column_config.TextColumn("Game Name"),
                         'similarity': st.column_config.NumberColumn("Similarity Score", format='percent'),
                         'header_image': st.column_config.ImageColumn("Game Cover", width="medium"),
                         'tags': st.column_config.ListColumn("Tags"),
                         'windows': st.column_config.CheckboxColumn("Windows"),
                         'mac': st.column_config.CheckboxColumn("Mac"),
                         'linux': st.column_config.CheckboxColumn("Linux"),
                         },
                     row_height=80,
                     height=700
                     )
        
# Initialize data
if 'data_processor' not in st.session_state:
    with st.spinner("Loading data...", show_time=True):
        if not os.path.exists(DATA_MAIN_PATH):
            print('Main data file not found. Downloading...')
            from core.data_downloader import download_prepare_source_data
            download_prepare_source_data()
            
        if os.path.exists(DATA_ML_PATH):
            print('ML data file found. Loading...')
            st.session_state.data_processor = DataProcessor(data_path=DATA_ML_PATH)
        else:
            print('ML data file not found. Creating new one...')
            st.session_state.data_processor = DataProcessor(data_path=DATA_ML_PATH, data_path_old=DATA_MAIN_PATH)
            
        if os.path.exists(DATA_UI_PATH):
            print('UI data file found. Loading...')
            st.session_state.data = pd.read_pickle(DATA_UI_PATH)
        else:
            print('UI data file not found. Creating new one...')
            st.session_state.data = process_data(ml_data=st.session_state.data_processor.data, main_data_path=DATA_MAIN_PATH)
            
        if os.path.exists(DATA_ML_MX_PATH):
            print('ML matrix data file found. Loading...')
            st.session_state.data_processor.prepare_matrix_for_recommendations(data_path=DATA_ML_MX_PATH)
        else:
            print('ML matrix data file not found. Creating new one...')
            st.session_state.data_processor.prepare_matrix_for_recommendations(data_path=DATA_ML_MX_PATH, data_path_old=DATA_ML_PATH)

# Initialize session_state for game selections if they don't exist
if 'included_games' not in st.session_state:
    print('Initializing session state...')
    st.session_state.included_games = []
if 'excluded_games' not in st.session_state:
    st.session_state.excluded_games = []
    
if 'games_names' not in st.session_state:
    st.session_state.games_names = set(st.session_state.data['name'])
    
if 'data_range' not in st.session_state:
    st.session_state.data_range = {
        'min_date': st.session_state.data['release_date'].min(),
        'max_date': st.session_state.data['release_date'].max(),
        'date_range_daily': pd.date_range(
            start=st.session_state.data['release_date'].min(), 
            end=st.session_state.data['release_date'].max(), 
            freq='D').date
    }
    
if 'tags' not in st.session_state:
    st.session_state.tags = extract_unique_values_from_column(st.session_state.data['tags'])
    
if 'developers' not in st.session_state:
    st.session_state.developers = extract_unique_values_from_column(st.session_state.data['developers'])
    
if 'languages' not in st.session_state:
    st.session_state.languages = sorted(extract_unique_values_from_column(st.session_state.data['supported_languages']))
    
if 'min_popularity' not in st.session_state and 'max_popularity' not in st.session_state:
    st.session_state.min_popularity, st.session_state.max_popularity = st.session_state.data['review_bucket'].min(), st.session_state.data['review_bucket'].max()

data = st.session_state.data.copy()

st.title('PlayNext')

with st.sidebar:
    st.header("Filters & Selections")
    
    st.markdown("### Filter by Platform")
    col1, col2, col3 = st.columns(3)
    with col1:
        filter_windows = st.checkbox("Windows", value=False)
    with col2:
        filter_linux = st.checkbox("Linux", value=False)
    with col3:
        filter_mac = st.checkbox("Mac", value=False)
        
    st.markdown("### Filter by Date")
    start_date, end_date = st.select_slider(
        "Select release date range",
        options=st.session_state.data_range['date_range_daily'],
        value=(st.session_state.data_range['min_date'], st.session_state.data_range['max_date']),
        label_visibility="collapsed",
    )
    
    st.markdown('### Filter by Tag')
    selected_tags = st.multiselect(
        "Select tags",
        options=st.session_state.tags,
        label_visibility="collapsed",
    )
    
    st.markdown('### Filter by Developer')
    selected_developers = st.multiselect(
        "Select developers",
        options=st.session_state.developers,
        label_visibility="collapsed",
    )
    
    st.markdown("### Filter by Language")
    selected_languages = st.multiselect(
        "Select languages",
        options=st.session_state.languages,
        label_visibility="collapsed",
    )
    
    st.markdown("### Filter by Popularity")
    col1, col2 = st.columns(2)
    with col1:
        selected_min_popularity = st.number_input(
            "Select minimum number of reviews",
            min_value=st.session_state.min_popularity,
            max_value=st.session_state.max_popularity,
            value=None,
            label_visibility="collapsed",
        )
    with col2:
        selected_max_popularity = st.number_input(
            "Select maximum number of reviews",
            min_value=st.session_state.min_popularity,
            max_value=st.session_state.max_popularity,
            value=st.session_state.max_popularity,
            label_visibility="collapsed",
        )
    
    st.markdown("### Filter by Game Name (for reference only)")
    game_search = st.text_input("Search game by name (for reference)", value="", label_visibility="collapsed", placeholder="e.g. Dota 2")
    
    
    # Apply filters based on user selections
    if filter_windows:
        data = data[data['windows'] == True]
    if filter_mac:
        data = data[data['mac'] == True]
    if filter_linux:
        data = data[data['linux'] == True]
        
    data = data[(data['release_date'] >= start_date) & (data['release_date'] <= end_date)]

    if selected_tags:
        data = data.loc[data['all_tags'].apply(lambda x: not x.isdisjoint(selected_tags))]
        
    if selected_developers:
        data = data.dropna(subset=['developers'])
        data = data.loc[data['developers'].apply(lambda x: not x.isdisjoint(selected_developers))]
        
    if selected_languages:
        data = data.dropna(subset=['supported_languages'])
        data = data.loc[data['supported_languages'].apply(lambda x: not x.isdisjoint(selected_languages))]

    if selected_min_popularity:
        data = data[data['review_bucket'] >= selected_min_popularity]

    if selected_max_popularity != 6:
        data = data[data['review_bucket'] <= selected_max_popularity]

    data_to_lookup = data

    if game_search:
        data_to_lookup = data[data['name'].str.contains(game_search, case=False)]
        
    st.markdown("### Selected Games")
    st.multiselect("Include", options=st.session_state.games_names, key="included_games")
    st.multiselect("Exclude", options=st.session_state.games_names, key="excluded_games")
    
    if st.button('Recommend', key='recommend_button', help="Click to get game recommendations based on your selections.", type='primary', icon="🔍", use_container_width=True):
        if st.session_state.included_games:
            show_recommended_games(st.session_state.included_games, st.session_state.excluded_games, data)
        else:
            st.write('No games selected for recommendation.')
            
    if st.toggle('Limit games shown to 1K', value=True, key='limit_games'):
        data_to_lookup = data_to_lookup.head(1000)


st.subheader(f'Games on Steam (March 2025 - {len(data)} games)')


with st.container():
    st.dataframe(
        data_to_lookup,
        column_config={
            'name': st.column_config.TextColumn("Game Name", pinned=True),
            "header_image": st.column_config.ImageColumn("Game Cover", pinned=True, width="medium"),
            'tags': st.column_config.ListColumn("Tags"),
            'release_date': st.column_config.DateColumn("Release Date"),
            'metacritic_score': st.column_config.NumberColumn("Metacritic Score"),
            'windows': st.column_config.CheckboxColumn("Windows"),
            'mac': st.column_config.CheckboxColumn("Mac"),
            'linux': st.column_config.CheckboxColumn("Linux"),
            'developers': st.column_config.ListColumn("Developers"),
            'review_bucket': st.column_config.ProgressColumn(
                "Popularity",
                min_value=0,
                max_value=6,
                format='plain',
                help="Reviews: 1:<100, 2:<1K, 3:<10K, 4:<100K, 5:<1M, 6:≥1M | Empty = N/A"),
            'short_description': st.column_config.TextColumn("Short Description", width='large'),
            'supported_languages': None,
            'all_tags': None
        },
        use_container_width=True,
        hide_index=True,
        row_height=80,
        height=700
    )

with st.expander("About this app"):
    st.write("""
        This app is designed to help you find your next favorite game on Steam. 
        It uses a dataset of games available on Steam and provides various features to help you explore and discover new games.
        You can filter games based on their availability on different platforms, release dates, and more.
        \nData source: [Steam Games Dataset on Kaggle](https://www.kaggle.com/datasets/artermiloff/steam-games-dataset/data)
    """)
if 'app_loaded' not in st.session_state: st.session_state.app_loaded = True; print("App loaded")
