from src.data_pipeline._05_feature_engg import get_data_from_feast
from src.model_pipeline._09_training import training
from src.model_pipeline._10_evaluation import evaluation
from src.model_pipeline._11_validation import validation
from src.model_pipeline._12_tuning import tuning
from src.model_pipeline._13_model_registry import model_registry
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(os.getcwd())

def model_pipleine(X_train, X_test, y_train, y_test, feature_names):

    lr_model = training(X_train, y_train)

    accuracy_metric = evaluation(lr_model, X_train, y_train, X_test, y_test)

    validation(lr_model, X_train, y_train)

    best_model, best_params, accuracy_ht = tuning(lr_model, X_train, X_test, y_train, y_test)

    print("✅ the end")

    model_registry(best_model, accuracy_metric, accuracy_ht, best_params, feature_names)


if __name__ == '__main__':
    # get training data
    df, feature_names = get_data_from_feast()
    print(df)

    # split
    X = df.drop(columns=['animal_name', 'class_name', 'event_timestamp'])
    y = df['class_name']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    feature_names = X_train.columns.tolist()
    print(feature_names)

    model_pipleine(X_train, X_test, y_train, y_test, feature_names)


# Final data columns: ['hair', 'eggs', 'milk', 'predator', 'toothed', 'backbone', 'breathes', 'venomous', 'legs', 'tail', 'can_fly', 'can_swim', 'is_domestic_pet', 'animal_name', 'class_name', 'event_timestamp']        
# Column names:  {(119, 16)}
