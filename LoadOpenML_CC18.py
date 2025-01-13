from openml import datasets, tasks, runs, flows, config, study
from openml.datasets import edit_dataset, fork_dataset, get_dataset
from openml.tasks import TaskType
import openml
import os
from tqdm import tqdm
import pandas as pd

# Load OpenML API key from a file
key_file = os.path.expanduser('~/openml_key.txt')  # Path to the key file
try:
    with open(key_file, 'r') as file:
        openml.config.apikey = file.read().strip()  # Read API key
except FileNotFoundError:
    raise FileNotFoundError(f"No API key found at {key_file}.")


openml.config.server = 'https://www.openml.org/api/v1'

# Cache directory
openml.config.cache_directory = os.path.expanduser('~/openml/cache')

# Select the correct suite (OpenML-CC18 has ID = 99 )
benchmark_suite = openml.study.get_suite(suite_id=99)

# Extract dataset IDs from OpenML-CC18
dataset_ids = benchmark_suite.data

# Define a path to save the datasets
save_path = os.path.expanduser('~/openml_benchmark_data_classification99')
os.makedirs(save_path, exist_ok=True)

# Download and save all datasets in the suite
for did in tqdm(dataset_ids):
    try:
        # Get the dataset by its ID
        dataset = openml.datasets.get_dataset(did)
        X, y, _, _ = dataset.get_data(
            target=dataset.default_target_attribute
        )
        
        # Create folder for this specific dataset
        dataset_path = os.path.join(save_path, f'dataset_{did}_{dataset.name}')
        os.makedirs(dataset_path, exist_ok=True)
        
        # Save features target as CSV file
        X.to_csv(os.path.join(dataset_path, 'features.csv'), index=False)
        y.to_csv(os.path.join(dataset_path, 'target.csv'), index=False)
        
        print(f"Successfully saved dataset {dataset.name} (ID: {did})")

    except Exception as e:
        print(f"Error occured while loading dataset with ID: {did}: {e}")