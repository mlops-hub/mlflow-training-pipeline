import joblib
import pandas as pd
import requests
import json
import mlflow
import pickle

# FEATURE_PATH = '../../feature_store/feature_names.pkl'
# SCALER = "../../utility/scaler.pkl"
# PREPROCESSED_DATASET = '../../datasets/preprocess/preprocessing_df.csv'

FEAST_SERVER_URL = "http://localhost:5050"

# mlflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")

model_name = "Animal Classifier Model"
model_version = "1"
run_id = "5a69d14e57ef42f3820ec18e756a11b5"
model = mlflow.pyfunc.load_model(f"models:/{model_name}/{model_version}")


# download artifacts
scaler_path = mlflow.artifacts.download_artifacts(artifact_uri=f"runs:/{run_id}/preprocessor/scaler.pkl")
scaler = joblib.load(scaler_path)

print("Scaler features:", scaler.feature_names_in_)
scaler_features = scaler.feature_names_in_

features_path = mlflow.artifacts.download_artifacts(artifact_uri=f"runs:/{run_id}/preprocessor/features_names.pkl")
feature_names = joblib.load(features_path)
print('feature-names: ', feature_names)



# features from feast
def get_features_from_feast(animal_name):
    payload = {
        "feature_service": "animal_features_service",
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

        # features_names = sorted([fn for fn in features_from_feast if fn not in ['animal_name', 'class_name']])
        # print(features_names)

        values = [r['values'][0] for r in results]
        df = pd.DataFrame([values], columns=features_from_feast)

        print(df)

        return df
    
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

    input_df = pd.DataFrame(scaler.transform(df), columns=scaler_features)

    print('sf: ', input_df)
    
    return input_df


def predict_animal(animal_name):
    df = get_features_from_feast(animal_name)

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
