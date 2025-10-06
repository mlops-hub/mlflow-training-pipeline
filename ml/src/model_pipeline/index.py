from src.data_pipeline._05_feature_engg import get_data_from_feast
from src.model_pipeline._09_training import training
from src.model_pipeline._10_evaluation import evaluation
from src.model_pipeline._11_validation import validation
from src.model_pipeline._12_tuning import tuning
from src.model_pipeline._13_model_registry import model_registry
from sklearn.model_selection import train_test_split
import os
from pathlib import Path
import mlflow


PROJECT_ROOT = Path(os.getcwd())
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
mlflow.set_experiment("Animal Classification")

def model_pipleine(X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name="animal_calssification") as run:
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
        
        print("✅ the end")



if __name__ == '__main__':
    # get training data
    df, feature_names = get_data_from_feast()
    print(df)

    # split
    X = df.drop(columns=['animal_name', 'class_name', 'event_timestamp'])
    y = df['class_name']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_pipleine(X_train, X_test, y_train, y_test)

