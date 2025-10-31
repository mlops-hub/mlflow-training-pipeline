import pandas as pd
import mlflow
from kserve import Model, ModelServer
import os
from dotenv import load_dotenv
# 
import time
from prometheus_client import Counter, Histogram, start_http_server

load_dotenv()

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "4")
MODEL_NAME = os.environ.get("MODEL_NAME", "Animal Classifier")
KSERVE_PORT = int(os.environ.get("KSERVE_PORT", 7070))
PROMETHEUS_PORT = int(os.environ.get("PROMETHEUS_PORT", 9090))

# promethes metrics

REQUEST_COUNT = Counter(
    "animal_classifier_requests_total",
    "Total number of prediction requests recieved",
    ["model_name", "model_version", "status"]
)

REQUEST_LATENCY = Histogram(
    "animal_classifier_request_latency_second",
    "LLatency of prediction requests in seconds"
)

ERROR_COUNT = Counter(
    "animal_classifier_error_total",
    "Number of failed prediction requests"
)

# ---------------------------------------------------------------

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
        start_time = time.time()
        print(f"Recieved payload: {payload}")
        instances = payload.get("instances", [])

        if not instances:
            REQUEST_COUNT.labels(MODEL_NAME, MODEL_VERSION, "error").inc()
            ERROR_COUNT.inc()
            return {"error": "No instances provided."}
        
        df = pd.DataFrame(instances)
        # Predict using MLflow model
        try:
            predictions = self.model.predict(df)
            if (hasattr(self.model, "predict_proba")):
                probs = self.model.predict_proba(df)
                confidence = probs.max(axis=1).tolist()
                print('probs: ', probs)
                print('confidence: ', confidence)

            # log metrics
            latency = time.time() - start_time
            REQUEST_COUNT.labels(MODEL_NAME, MODEL_VERSION, "success").inc()
            REQUEST_LATENCY.observe(latency)

            print(f"prediction: {predictions}")
            
            return {
                "prediction": predictions.tolist(), 
                "confidence": confidence if confidence else None,
            }
        except Exception as e:
            REQUEST_COUNT.labels(MODEL_NAME, MODEL_VERSION, "error").inc()
            ERROR_COUNT.inc()
            print(f"Error during prediction: {e}")
            return {"error": str(e)}


if __name__ == "__main__":
    # start prometheus metric
    start_http_server(PROMETHEUS_PORT)
    
    model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
    print(f"Using MLflow tracking URI: {MLFLOW_TRACKING_URI}")
    print(f"Using model registry URI: {model_uri}")

    server = ModelServer(http_port=KSERVE_PORT)

    model = AnimalClassPrediction(
        name="mlops_animal_classifer",
        model_uri=model_uri
    )
    
    model.load()
    server.start(models=[model])

