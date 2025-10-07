from feast import FeatureStore
import pandas as pd 
import os
from dotenv import load_dotenv
from feature_store.features import animal_features_fv

load_dotenv()

PROJECT_ROOT = os.getcwd()
feast_repo_path = os.path.join(PROJECT_ROOT, "feature_store")
print('repo-path: ', feast_repo_path)

preprocessed_df_path = os.path.join(PROJECT_ROOT, 'feature_store/data/preprocessed_data.parquet')

# import feast feature
MODEL_INPUT_FEATURE_ORDER = sorted([
        field.name for field in animal_features_fv.schema
        if field.name not in ["animal_name", "event_timestamp", "created_timestamp"]
])

store = FeatureStore(repo_path=feast_repo_path)

entity_df = pd.DataFrame({
    "animal_name": ["bear"],
    "event_timestamp": [pd.Timestamp.now()]
})
all_features_to_fetch_from_feast = [f"animal_preprocessed_features:{feature}" for feature in MODEL_INPUT_FEATURE_ORDER]


df = store.get_historical_features(
    entity_df=entity_df,
    features=all_features_to_fetch_from_feast
).to_df()

print(df)


