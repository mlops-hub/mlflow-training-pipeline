from dotenv import load_dotenv
import os
import mlflow
import mlflow.sklearn

def model_registry(best_model, accuracy_metric, accuracy_ht, best_params):
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("Animal Classification")

    with mlflow.start_run(run_name="animal_calssification") as run:
        mlflow.log_metric("base_model_accuracy", accuracy_metric)
        mlflow.log_metric("best_model_accuracy", accuracy_ht)

        mlflow.log_param("best_params", str(best_params))
        mlflow.sklearn.log_model(
            sk_model=best_model,
            name="LogisticRegression",
            registered_model_name="Animal Classifier Model", # Register the model in MLflow Model Registry
        )

