import pandas as pd
import re
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ZOO_DATASET_PATH = os.path.join(PROJECT_ROOT, "datasets/raw/zoo.csv")
CLASS_DATASET_PATH = os.path.join(PROJECT_ROOT, "datasets/raw/class.csv")
MERGED_DATASET_PATH = os.path.join(PROJECT_ROOT, "datasets/ingestion")

def ingestion():
    # 1. load datasets
    zoo_df = pd.read_csv(ZOO_DATASET_PATH)
    class_df = pd.read_csv(CLASS_DATASET_PATH)

    # 2. split 'class_df' => 'animal_names' column in separate rows and create new column 'animal_name'
    class_df['animal_name'] = class_df['Animal_Names'].apply(lambda x: re.split(r",\s*", x))

    class_df = class_df.explode('animal_name')

    # remove unnecessary columns in 'class_dataset'
    class_df = class_df.drop(columns=['Animal_Names', 'Number_Of_Animal_Species_In_Class', 'Class_Number'])

    # rename 'Class_Type' to 'class_name'
    class_df = class_df.rename(columns={'Class_Type': 'class_name'})
    # 3. merge two datasets
    merged_df = pd.merge(zoo_df, class_df, on='animal_name', how='left')

    # 4. Create directory if it doesn't exist and save the file
    os.makedirs(MERGED_DATASET_PATH, exist_ok=True)  # This creates the directory if it doesn't exist

    merged_df.to_csv(f"{MERGED_DATASET_PATH}/merged_df.csv", index=False)
    print(f"File saved successfully to: {MERGED_DATASET_PATH}/merged_df.csv")
    return merged_df