import joblib
import pandas as pd
import requests
import json
import os
from pathlib import Path

FEAST_SERVER_URL = "http://4.246.120.68:30800"
PROJECT_ROOT = Path(os.getcwd())

# 
import pandas as pd

data_path = os.path.join(PROJECT_ROOT, "feature_store/data/preprocessed_data.parquet")
print('data-path: ', data_path)

df = pd.read_parquet(data_path)
print(df['event_timestamp'].min(), df['event_timestamp'].max())
print(df['animal_name'].unique())


# 
from feast import FeatureStore
fs = FeatureStore(repo_path="feature_store")

online_features = fs.get_online_features(
    features=["animal_preprocessed_features:backbone", "animal_preprocessed_features:milk"],
    entity_rows=[{"animal_name": "bear"}]  # or "Bear"
).to_df()

print(online_features)

# 
payload = {
    "feature_service": "animal_features_service",
    "entities": {
        "animal_name": ['bear']
    }
}
headers = {"Content-Type": "application/json"}
   
try:
    response = requests.post(
        f"{FEAST_SERVER_URL}/get-online-features",
        data=json.dumps(payload),
        headers=headers
    )

    feature_names = response.json()['metadata']['feature_names']
    results = response.json()['results']

    values = [r['values'][0] for r in results]

    df = pd.DataFrame([values], columns=feature_names)
    print(df)
            
    print(f"Feature_name: {feature_names}")

except Exception as e:
    print(f"Error communicating with Feast server: {e}")
 

