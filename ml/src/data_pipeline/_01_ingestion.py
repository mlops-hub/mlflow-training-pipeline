import pandas as pd
import re
import os

PROJECT_ROOT = os.getcwd()

DATASET_PATH = os.path.join(PROJECT_ROOT, "ml/data/ingestion/merged_df.csv")

def ingestion():
    # collect data from central system
    get_dataset = pd.read_csv(DATASET_PATH)
    return get_dataset
