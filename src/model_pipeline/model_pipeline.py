from src.model_pipeline._07_training import training
from src.model_pipeline._08_evaluation import evaluation
from src.model_pipeline._09_validation import validation
from src.model_pipeline._10_tuning import tuning
from src.data_pipeline._05_feature_engg import get_data_from_feast
from src.model_pipeline._11_model_registry import model_registry
import os
import pandas as pd
from pathlib import Path

# get training data
# df = get_data_from_feast()
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# build preprocess dataset directory and read files
preprocess_dir = PROJECT_ROOT / "datasets" / "preprocess"

X_train = pd.read_csv(preprocess_dir / "X_train_dataset.csv")
X_test = pd.read_csv(preprocess_dir / "X_test_dataset.csv")
y_train = pd.read_csv(preprocess_dir / "y_train_dataset.csv").values.ravel()
y_test = pd.read_csv(preprocess_dir / "y_test_dataset.csv").values.ravel()


# model pipleine
lr_model = training(X_train, y_train)

# # mlflow.sklearn.log_model(
# #     sk_model=lr_model,
# #     name="animal_classification_base_model",
# #     registered_model_name="Animal Classifier Base Model",
# # )

accuracy_metric = evaluation(lr_model, X_train, y_train, X_test, y_test)
# # mlflow.log_metric("base_model_accuracy", accuracy_metric)

cv_score = validation(lr_model, X_train, y_train)
# # mlflow.log_metric("base_model_cv_score", cv_score)

best_model, best_params, accuracy_ht = tuning(lr_model, X_train, X_test, y_train, y_test)

print("✅ the end")

model_registry(best_model, accuracy_metric, accuracy_ht, best_params)