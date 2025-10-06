import joblib
import pandas as pd
import requests
import json
import os
from feast import FeatureStore

FEAST_SERVER_URL = "http://localhost:5050"

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

    
    # Filter out employee_id as it's an entity key, not a feature for the model
    filtered_feature_names = [name for name in feature_names if name != 'animal_name']
        
    print(f"Feature_name: {filtered_feature_names}")

except Exception as e:
    print(f"Error communicating with Feast server: {e}")
 

