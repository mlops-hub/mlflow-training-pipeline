from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib
import os

PROJECT_ROOT = os.getcwd()

def training(X_train, y_train):
    model = LogisticRegression()
    
    model.fit(X_train, y_train)
    print(model)

    # save base model
    output_dir = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(output_dir, exist_ok=True)

    joblib.dump(model, f"{output_dir}/base_model.pkl")
    return model
