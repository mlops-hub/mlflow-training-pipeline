from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def training(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    print(model)

    # save base model
    output_dir = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(output_dir, exist_ok=True)

    joblib.dump(model, f"{output_dir}/base_model.pkl")
    return model


# notes
# Warning :
# C:\Users\DELL\OneDrive\Desktop\CrunchOps\mlops-hub\classifier-model\venv\Lib\site-packages\sklearn\utils\validation.py:1406: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
#   y = column_or_1d(y, warn=True)

# solution:
# This warning means your y_train is a column vector (shape (n_samples, 1)) instead of a 1D array (shape (n_samples,)).
# scikit-learn expects a 1D array for the target variable.

# How to fix:
# When loading y_train from CSV, use .values.ravel() or .squeeze() to convert it to 1D:

