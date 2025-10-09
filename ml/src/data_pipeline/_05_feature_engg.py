import os
import pandas as pd
import datetime
from feast import FeatureStore
from feature_repo.features import animal_feature_fv
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = os.getcwd()
SAVE_FEATURE_DF = os.path.join(PROJECT_ROOT, "ml/data/feature_engg")
os.makedirs(SAVE_FEATURE_DF, exist_ok=True)
    
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "ml/data/prepared/prepared_df.csv")
os.makedirs(os.path.dirname(OUTPUT_DIR), exist_ok=True)    

OUTPUT_PARQUET_PATH = os.path.join(PROJECT_ROOT, 'feature_repo/data/preprocessed_data.parquet')
os.makedirs(os.path.dirname(OUTPUT_PARQUET_PATH), exist_ok=True)    


def feature_engg(df):
    df['can_fly'] = (df['airborne'] == 1) & (df['feathers'] == 1).astype(int)
    df['can_swim'] = (df['aquatic'] == 1) & (df['fins'] == 1).astype(int)
    df['is_domestic_pet'] = (df['domestic'] == 1) & (df['catsize'] == 1).astype(int)

    # save feature_engg dataset
    df.to_csv(f"{SAVE_FEATURE_DF}/feature_df.csv", index_label=False)
    print(df.head(3))
    return df

# save to feast registry
def prepare_data_for_feast(df):
    final_df = df.copy()
    final_df['event_timestamp'] = pd.to_datetime(datetime.datetime.now()) - pd.to_timedelta(final_df.index, unit='D')
    print('final-dataset: ', final_df)
    # ensure output_data directory exists !    
    final_df.to_parquet(OUTPUT_PARQUET_PATH, index=False)
    # save in local file as well
    final_df.to_csv(OUTPUT_DIR, index=False)
    print("Data preparation complete and saved successfully.")
    print(f"Final data columns: {final_df.columns.tolist()}")
    print("Column names: ", {final_df.shape})
    # test the parquet file
    df = pd.read_parquet(OUTPUT_PARQUET_PATH)
    # Check the row for 'bear'
    bear_data = df[df['animal_name'] == 'bear']
    print('bear-data: ', bear_data.head())


# get features from feast
def get_data_from_feast():
    feast_repo_path = os.path.join(PROJECT_ROOT, "feature_repo")
    # import feast feature
    MODEL_INPUT_FEATURE_ORDER = sorted([
        field.name for field in animal_feature_fv.schema
        if field.name not in ["animal_name", "event_timestamp", "created_timestamp"]
    ])

    fs = FeatureStore(repo_path=feast_repo_path)
    preprocessed_df_path = os.path.join(PROJECT_ROOT, 'feature_repo/data/preprocessed_data.parquet')

    if not os.path.exists(preprocessed_df_path):
        print(f"Preprocessed data not found at: {preprocessed_df_path}")
        return None
    
    entity_df = pd.read_parquet(preprocessed_df_path, columns=['animal_name', 'event_timestamp'])
    all_features_to_fetch_from_feast = [
        f"animal_feature_fv:{feature}" 
        for feature in MODEL_INPUT_FEATURE_ORDER
    ]

    training_df = fs.get_historical_features(
        entity_df=entity_df,
        features=all_features_to_fetch_from_feast
    ).to_df()
    print("training_df: ", training_df)

    return training_df