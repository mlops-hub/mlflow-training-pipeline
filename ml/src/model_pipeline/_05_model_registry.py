import os
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature


def model_registry(best_model, X_train): 
    client = MlflowClient()

    # Infer model signature
    signature = infer_signature(X_train, best_model.predict(X_train))

    # Log the model
    model_info = mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            signature=signature,
            input_example=X_train[:5],  # Sample input for documentation
    )

    # DEBUG: Check where artifacts are stored
    print(f"Model URI: {model_info.model_uri}")
    registered_model = mlflow.register_model(
            model_uri=model_info.model_uri,
            name="Animal Classifier Model"
    )

    # Use tags to track model status and metadata
    client.set_model_version_tag(
            name="Animal Classifier Model",
            version=registered_model.version,
            key="production_ready",
            value="approved",
    )
