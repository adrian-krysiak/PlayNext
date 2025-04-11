import os
import kagglehub
import shutil

def download_prepare_source_data():
    """
    Prepares the source data for processing by ensuring the necessary directories 
    and files are in place. If the required data file is not found, it downloads 
    the dataset from KaggleHub, extracts the relevant file, and organizes the 
    directory structure.
    Steps performed:
    1. Ensures the existence of the 'data' directory.
    2. Checks for the presence of the required data file ('games_march2025_cleaned.csv').
       - If not found, downloads the dataset from KaggleHub and moves the required file 
         to the 'data' directory.
       - Cleans up unnecessary directories created during the download process.
    3. Ensures the existence of the 'processed' directory for further processing.
    Note:
    - Requires the `kagglehub` library for dataset downloading.
    - Assumes the dataset contains a file named 'games_march2025_cleaned.csv'.
    Raises:
    - Any exceptions related to file operations or directory creation will propagate 
      to the caller.
    Prints:
    - Status messages indicating the progress of directory creation, file downloading, 
      and cleanup operations.
    """
    # Define paths
    data_dir = 'data'
    data_file = 'games_march2025_cleaned.csv'
    processed_dir = os.path.join(data_dir, 'processed')

    # Create data directory if it doesn't exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created directory: {data_dir}")
    else:
        print(f"Directory already exists: {data_dir}")

    # Check if data.csv exists
    data_path = os.path.join(data_dir, data_file)
    if not os.path.exists(data_path):
        print(f"{data_file} not found. Downloading from KaggleHub...")

        # Download using kagglehub
        path = kagglehub.dataset_download("artermiloff/steam-games-dataset")
        print("Path to dataset files:", path)

        # Move a specific file to the data directory as data_file
        source_file = os.path.join(path, data_file)
        if os.path.exists(source_file):
            os.rename(source_file, data_path)
            print(f"Moved {source_file} to {data_path}")
        else:
            print("Expected source file not found in the downloaded dataset.")

        # Delete the artermiloff directory and its contents
        artermiloff_dir = os.path.join(path, "..", "..", "..")
        artermiloff_dir = os.path.abspath(artermiloff_dir)  # Normalize the path
        if os.path.exists(artermiloff_dir):
            shutil.rmtree(artermiloff_dir)
            print(f"Deleted directory and its contents: {artermiloff_dir}")
        else:
            print(f"Directory not found: {artermiloff_dir}")
    else:
        print(f"{data_file} already exists.")

    # Create processed directory if it doesn't exist
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
        print(f"Created directory: {processed_dir}")
    else:
        print(f"Directory already exists: {processed_dir}")