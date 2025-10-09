import joblib
import pandas as pd
import requests
import json
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(os.getcwd()).parent.parent
print(PROJECT_ROOT)

FEAST_SERVER_URL = os.environ.get("FEAST_SERVER_URL", "http://localhost:5050")
print(FEAST_SERVER_URL)

OUTPUT_PARQUET_PATH = os.path.join(PROJECT_ROOT, 'feature_repo/data/preprocessed_data.parquet')
os.makedirs(os.path.dirname(OUTPUT_PARQUET_PATH), exist_ok=True)  

df = pd.read_parquet(OUTPUT_PARQUET_PATH)
# Check the row for 'bear'
bear_data = df[df['animal_name'] == 'bear']
print('bear-data: ', bear_data.head())


payload = {
    "feature_service": "animal_feature_service",
    "entities": {
        "animal_name": ["kangaroo"]
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
    print(results)

    values = [r['values'][0] for r in results]

    df = pd.DataFrame([values], columns=feature_names)
    print(df)
            
    print(f"Feature_name: {feature_names}")

except Exception as e:
    print(f"Error communicating with Feast server: {e}")
 

