import pandas as pd
import mlflow
from kserve import Model, ModelServer
import os
from dotenv import load_dotenv

load_dotenv()

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "1")
MODEL_NAME = os.environ.get("MODEL_NAME", "Animal Classifier")
KSERVE_PORT = os.environ.get("KSERVE_PORT", 7070)


class AnimalClassPrediction(Model):
    def __init__(self, name, model_uri):
        super().__init__(name)
        self.model_uri = model_uri
        self.ready = False

    def load(self):
        try:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            self.model = mlflow.sklearn.load_model(self.model_uri)
            self.ready = True
        except Exception as e:
            print(f"Error during load: {e}")
            self.ready = False


    def predict(self, payload, headers=None):
        print(f"Recieved payload: {payload}")

        instances = payload.get("instances", [])
        if not instances:
            return {"error": "No instances provided."}

        df = pd.DataFrame(instances)
        
        print(f"Input DataFrame shape: {df.shape}")
        print(f"Input DataFrame columns: {df}")
        
        # Predict using MLflow model
        predictions = self.model.predict(df)
        print(f"prediction: {predictions}")

        # Return results as list
        return {
            "prediction": predictions.tolist(), 
        }


if __name__ == "__main__":
    #  construct model uri from registry
    model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
    print(f"Using Mlflow tracking URI: {MLFLOW_TRACKING_URI}")
    print(f"Using model registry uri: {model_uri}")

    server = ModelServer(http_port=KSERVE_PORT)

    model = AnimalClassPrediction(
        name="mlops_animal_classifer",
        model_uri=model_uri
    )
    
    model.load()
    server.start(models=[model])

