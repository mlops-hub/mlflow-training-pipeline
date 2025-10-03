from dotenv import load_dotenv
import joblib
import os
import mlflow
import mlflow.sklearn


def model_registry(best_model, accuracy_metric, accuracy_ht, best_params, feature_names):
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
        # save model artifact
        OUTPUT_DIR = "utility"
        os.makedirs(f"{OUTPUT_DIR}", exist_ok=True)

        mlflow.log_artifact(f"{OUTPUT_DIR}/scaler.pkl", artifact_path="preprocessor")

        joblib.dump(feature_names, f"{OUTPUT_DIR}/features_names.pkl")
        mlflow.log_artifact(f"{OUTPUT_DIR}/features_names.pkl", artifact_path="preprocessor")

