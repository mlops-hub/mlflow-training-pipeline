import pandas as pd
import requests
import json
import mlflow

FEAST_SERVER_URL = "http://localhost:5050"
KSERVE_URL = "http://localhost:7070/v1/models/mlops_animal_classifer:predict"
MLFLOW_URL = "http://localhost:5000"
MLFLOW_RUN_ID = "185c5c005a2b4d32a3d6cbc281ec7add"

# mlflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")

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


# Get the features for prediction
def create_animal(scaler_features):
    print("Enter features here: ")
    features_dict = {
        "hair": int(input("hair (0 or 1): ")),
        "eggs": int(input("eggs (0 or 1): ")),
        "milk": int(input("milk (0 or 1): ")),
        "predator": int(input("predator (0 or 1): ")),
        "toothed": int(input("toothed (0 or 1): ")),
        "backbone": int(input("backbone (0 or 1): ")),
        "breathes": int(input("breathes (0 or 1): ")),
        "venomous": int(input("venomous (0 or 1): ")),
        "legs": int(input("legs (0,2,4,5,6,8): ")),
        "tail": int(input("tail (0 or 1): ")),
        "can_fly": int(input("can_fly (0 or 1): ")),
        "can_swim": int(input("can_swim (0 or 1): ")),
        "is_domestic_pet": int(input("is_domestic_pet (0 or 1): ")),
    }

    df = pd.DataFrame([features_dict]).reindex(columns=scaler_features)
    print('df: ', df)
    return df


def predict_animal(animal_name):
    df, feature_names = get_features_from_feast(animal_name)
    print(df.isnull().values.any())

    if df.isnull().values.any():
        print("This animal is not found in dataset.")
        df = create_animal(scaler_features)
    else:
        print('found df')
    
    input_df = df.reindex(columns=feature_names)
    print('idf: ', input_df)

    # Drop unused columns if still present
    if "animal_name" in input_df.columns:
        input_df = input_df.drop(columns=["animal_name"])
    if "class_name" in input_df.columns:
        input_df = input_df.drop(columns=["class_name"])

    y_result = model.predict(input_df)[0]
    print(f"\nAnimal Type for {animal_name}: {y_result}\n")


if __name__ == "__main__":
    while True:
        animal_name = input("Enter animal name: ")
        if animal_name.lower() in ['exit', 'quit']:
            print("👋🏻 Bye")
            break
        predict_animal(animal_name)
