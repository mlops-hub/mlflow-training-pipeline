



from dotenv import load_dotenv
import os
import mlflow
import mlflow.sklearn


load_dotenv()

if __name__ == "__main__":
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("Animal Classification")

    with mlflow.start_run(run_name="animal_calssification") as run:
        data_pie(...)

        # log preprocessing artifacts
        mlflow.log_artifact("../feature_store/feature_names.pkl")
        mlflow.log_artifact("../utility/scaler.pkl")
        mlflow.log_artifact("../datasets/preprocess/preprocessing_df.csv")

        
