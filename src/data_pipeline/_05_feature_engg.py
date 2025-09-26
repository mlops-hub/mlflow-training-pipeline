import pandas as pd
import datetime
import os
from feast import FeatureStore
from feature_store.features import animal_features_fv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def feature_engg(df):
    df['can_fly'] = (df['airborne'] == 1) & (df['feathers'] == 1).astype(int)
    df['can_swim'] = (df['aquatic'] == 1) & (df['fins'] == 1).astype(int)
    df['is_domestic_pet'] = (df['domestic'] == 1) & (df['catsize'] == 1).astype(int)

    # save feature_engg dataset
    output_dir = os.path.join(PROJECT_ROOT, "datasets/feature_engg")
    os.makedirs(output_dir, exist_ok=True)

    df.to_csv(f"{output_dir}/feature_df.csv", index_label=False)
    print(df.head(3))
    return df


def prepare_data_for_feast(df):
    # Create unique employee_id and timestamp for Feast
    final_df = df.copy()

    final_df['event_timestamp'] = pd.to_datetime(datetime.datetime.now()) - pd.to_timedelta(final_df.index, unit='D')
    print("Added 'event_timestamp' for feast")

    # ensoure output_data directory exists !
    output_parquet_path = os.path.join(PROJECT_ROOT, 'feature_store/data/preprocessed_data.parquet')
    os.makedirs(os.path.dirname(output_parquet_path), exist_ok=True)    
    
    # save to parquet
    final_df.to_parquet(output_parquet_path, index=False)

    print("Data preparation complete and saved successfully.")
    print(f"Final data columns: {final_df.columns.tolist()}")
    print("Column names: ", {final_df.shape})



# Added 'event_timestamp' for feast
# Data preparation complete and saved successfully.
# Final data columns: ['hair', 'eggs', 'milk', 'predator', 'toothed', 'backbone', 'breathes', 'venomous', 'legs', 'tail', 'can_fly', 'can_swim', 'is_domestic_pet', 'animal_name', 'event_timestamp']
# Column names:  {(119, 15)}


def get_data_from_feast():
    feast_repo_path = os.path.join(PROJECT_ROOT, "feature_store")

    # import feast features
    MODEL_INPUT_FEATURE_ORDER = sorted([
        field.name for field in animal_features_fv.schema
        if field.name not in ["animal_name", "event_timestamp", "created_timestamp"]
    ])

    fs = FeatureStore(repo_path=feast_repo_path)
    preprocessed_df_path = os.path.join(PROJECT_ROOT, 'feature_store/data/preprocessed_data.parquet')
    if not os.path.exists(preprocessed_df_path):
        print(f"Preprocessed data not found at: {preprocessed_df_path}")
        return None
    
    entity_df = pd.read_parquet(preprocessed_df_path, columns=['animal_name', 'event_timestamp'])
    all_features_to_fetch_from_feast = [f"animal_preprocessed_features:{feature}" for feature in MODEL_INPUT_FEATURE_ORDER]

    training_df = fs.get_historical_features(
        entity_df=entity_df,
        features=all_features_to_fetch_from_feast
    ).to_df()
    print("training_df: ", training_df)

    columns_to_drop_from_training_df = [
        col for col in training_df.columns
        if col.startswith(('anima_name', 'event_timestamp', 'created_timestamp'))
    ]
    model_features = training_df.drop(columns_to_drop_from_training_df, axis=1)
    print(f"Training data columns: {training_df.columns.tolist()}")
    print(f"Model Features columns: {model_features.columns.tolist()}")

    return model_features

# Training data columns: ['animal_name', 'event_timestamp', 'backbone', 'breathes', 'can_fly', 'can_swim', 'eggs', 'hair', 'is_domestic_pet', 'legs', 'milk', 'predator', 'tail', 'toothed', 'venomous']
# Model Features columns: ['animal_name', 'backbone', 'breathes', 'can_fly', 'can_swim', 'eggs', 'hair', 'is_domestic_pet', 'legs', 'milk', 'predator', 'tail', 'toothed', 'venomous']