import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import requests
import pandas as pd
import mlflow
import json
# 
from monitoring.scripts.save_logs import init_db, log_prediction
from monitoring.scripts.save_live_data import init_live_db, log_live_data


load_dotenv()

FEAST_SERVER_URL = os.environ.get("FEAST_SERVER_URL", "http://localhost:5050") 
KSERVE_URL = os.environ.get("KSERVE_URL", "http://localhost:7070/v1/models/mlops_animal_classifer:predict")
MLFLOW_URL = os.environ.get("MLFLOW_URL", "http://localhost:5000")
MLFLOW_RUN_ID = os.environ.get("MLFLOW_RUN_ID", "53485df8413c4d2b8eafb31e97c904ba")

# mlflow
mlflow.set_tracking_uri(MLFLOW_URL)

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

# init db
init_db()
init_live_db()



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
    
        original_df = df.reindex(columns=feature_names)
        print('og: ', original_df)

        # Drop unused columns if still present
        input_df = original_df.drop(columns=[c for c in ["animal_name", "class_name"] if c in original_df.columns])
        print('input-df: ', input_df)
         
        # Send to KServe model running locally
        response = requests.post(KSERVE_URL, json={"instances": input_df.to_dict(orient="records")})        
        print('✅😷 result: ', response.json())

        prediction_result = response.json()["prediction"][0]
        confidence = response.json()["confidence"][0]

        # log in db
        log_prediction(
            input_data=user_data['userInput'],
            prediction=prediction_result,
            confidence=confidence
        )
        feature_row = original_df.iloc[0].to_dict()
        print('feature_row): ', feature_row)
        log_live_data(user_data['userInput'], feature_row, prediction_result)

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

    try:
        _, feature_names = get_features_from_feast(user_data['animal_name'])
        feature_names = sorted(feature_names)
        print('_df: ', _)
        
        df = pd.DataFrame([user_data])
        print('df: ', df)

        original_df = df.reindex(columns=feature_names)
        print('idf: ', original_df)

        # Drop unused columns if still present
        input_df = original_df.drop(columns=[c for c in ["animal_name", "class_name"] if c in original_df.columns])
        print('input-df: ', input_df)

        response = requests.post(KSERVE_URL, json={"instances": input_df.to_dict(orient="records")})        
        print('✅😷 result: ', response.json())

        prediction_result = response.json()["prediction"][0]

        # log in db
        log_prediction(
            input_data=user_data['animal_name'],
            prediction=prediction_result,
            confidence=None
        )
        feature_row = original_df.iloc[0].to_dict()
        log_live_data(user_data['animal_name'], feature_row, prediction_result)

        payload = { "prediction": prediction_result }
        return payload

    except requests.exceptions.RequestException as e:
        print(f"Error communicating with KServe: {e}")

    except Exception as e:
        return {"error": str(e)}



if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=3000)