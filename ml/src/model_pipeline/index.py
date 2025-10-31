from ml.src.data_pipeline._05_feature_engg import get_data_from_feast
from ml.src.model_pipeline._01_training import training
from ml.src.model_pipeline._02_evaluation import evaluation
from ml.src.model_pipeline._03_validation import validation
from ml.src.model_pipeline._04_tuning import tuning
from ml.src.model_pipeline._05_model_registry import model_registry
from sklearn.model_selection import train_test_split
import os
from pathlib import Path
import mlflow
# 
from monitoring.scripts.save_reference_data import save_reference_data

PROJECT_ROOT = Path(os.getcwd())
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT_NAME", "Animal Classifier Dev"))

# Ensure no active runs
while mlflow.active_run() is not None:
    mlflow.end_run()

# Optional: clean up env variable
os.environ.pop("MLFLOW_RUN_ID", None)

def model_pipleine(X_train, X_test, y_train, y_test, df_ref):
    with mlflow.start_run(run_name="animal_classifier_dev") as run:
        print("Run ID:", run.info.run_id)

        base_model = training(X_train, y_train)

        accuracy_metric = evaluation(base_model, X_train, y_train, X_test, y_test)
        mlflow.log_metric("base_model_accuracy", accuracy_metric)

        cv_score = validation(base_model, X_train, y_train)
        mlflow.log_metric("base_model_cv_score", cv_score)

        best_model, best_params, accuracy_ht = tuning(base_model, X_train, X_test, y_train, y_test)
        mlflow.log_metric("best_model_accuracy", accuracy_ht)
        mlflow.log_param("best_params", str(best_params))
        
        print("✅ Register the best model")
        model_registry(best_model, X_train)

        # save the reference dataset
        X_ref = df_ref.drop(columns=['animal_name', 'class_name', 'event_timestamp'])
        ref_pred = best_model.predict(X_ref)
        df_ref['prediction'] = ref_pred
        save_reference_data(df_ref)
        
        print("✅ the end")



if __name__ == '__main__':
    # get training data
    df = get_data_from_feast()
    print(df)
    df_ref = df.copy()

    # split
    X = df.drop(columns=['animal_name', 'class_name', 'event_timestamp'])
    y = df['class_name']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_pipleine(X_train, X_test, y_train, y_test, df_ref)

