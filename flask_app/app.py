import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import requests
import pandas as pd
import mlflow
import joblib
import json


load_dotenv()

FEAST_SERVER_URL = os.environ.get("FEAST_SERVER_URL", "http://localhost:5050") # Or the load balancer URL if on K8s
KSERVE_URL = os.environ.get("KSERVE_URL", "http://localhost:7070/v1/models/mlops_animal_classifer:predict")

MLFLOW_URL = os.environ.get("MLFLOW_URL", "http://localhost:5000") # Or the load balancer URL if on K8s
MLFLOW_RUN_ID = os.environ.get("MLFLOW_RUN_ID", "185c5c005a2b4d32a3d6cbc281ec7add") # Or the load balancer URL if on K8s
print(MLFLOW_RUN_ID)
print('feast-url', FEAST_SERVER_URL)

# mlflow
mlflow.set_tracking_uri(MLFLOW_URL)

# download artifacts
# scaler_path = mlflow.artifacts.download_artifacts(artifact_uri=f"runs:/{MLFLOW_RUN_ID}/preprocessor/scaler.pkl")
# scaler = joblib.load(scaler_path)

# print("Scaler features:", scaler.feature_names_in_)
# scaler_features = scaler.feature_names_in_

# features_path = mlflow.artifacts.download_artifacts(artifact_uri=f"runs:/{MLFLOW_RUN_ID}/preprocessor/features_names.pkl")
# feature_names = joblib.load(features_path)
# print('feature-names: ', feature_names)


# features from feast
def get_features_from_feast(animal_name):
    payload = {
        "feature_service": "animal_feature_service",
        "entities": {
            "animal_name": [animal_name]
        }
    }
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(
            f"{FEAST_SERVER_URL}/get-online-features",
            data=json.dumps(payload),
            headers=headers
        )
        resp = response.json()
        features_from_feast = resp['metadata']['feature_names']  # keep original order!
        results = resp['results']

        print('feast-results: ', results)

        values = [r['values'][0] for r in results]
        df = pd.DataFrame([values], columns=features_from_feast)
        print(df)
        return df, features_from_feast
    
    except Exception as e:
        print(f"Error communicating with Feast server: {e}")
        return None
    

# start app
app = Flask(__name__)

CORS(app)


# home page
@app.route("/", methods=['GET'])
def index():
    return render_template('index.html')


# predict with animal name
@app.route("/predict", methods=['POST'])
def predict():
    user_data = request.get_json()
    print('data: ', user_data)

    try:
        df, feature_names = get_features_from_feast(user_data['userInput'])
        feature_names = sorted(feature_names)

        # validation: check for any NaNs introduced by reindexing
        if df.isnull().values.any():
            print("This animal is not found in dataset.")
            return {"prediction": "Find With Features"}
        else:
            print('found df')
    
        input_df = df.reindex(columns=feature_names)
        print('idf: ', input_df)

        # Drop unused columns if still present
        if "animal_name" in input_df.columns:
            input_df = input_df.drop(columns=["animal_name"])
        if "class_name" in input_df.columns:
            input_df = input_df.drop(columns=["class_name"])

        # Send to KServe model running locally
        print(KSERVE_URL)
        response = requests.post(KSERVE_URL, json={"instances": input_df.to_dict(orient="records")})        
        print('✅😷 result: ', response.json())

        prediction_result = response.json()["prediction"][0]

        payload = { "prediction": prediction_result }
        return payload
    
    except requests.exceptions.RequestException as e:
        print(f"Error communicating with KServe: {e}")

    except Exception as e:
        return {"error": str(e)}


# predict with animal features
@app.route("/predict_features", methods=['POST'])
def predict_features():
    user_data = request.get_json()
    print('data: ', user_data)

    try:
        _, feature_names = get_features_from_feast(user_data['animal_name'])
        feature_names = sorted(feature_names)
        
        df = pd.DataFrame([user_data])
        print('df: ', df)

        input_df = df.reindex(columns=feature_names)
        print('idf: ', input_df)

        # Drop unused columns if still present
        if "animal_name" in input_df.columns:
            input_df = input_df.drop(columns=["animal_name"])
        if "class_name" in input_df.columns:
            input_df = input_df.drop(columns=["class_name"])

        print(KSERVE_URL)
        response = requests.post(KSERVE_URL, json={"instances": input_df.to_dict(orient="records")})        
        print('✅😷 result: ', response.json())

        prediction_result = response.json()["prediction"][0]

        payload = { "prediction": prediction_result }
        return payload

    except requests.exceptions.RequestException as e:
        print(f"Error communicating with KServe: {e}")

    except Exception as e:
        return {"error": str(e)}



if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=3000)