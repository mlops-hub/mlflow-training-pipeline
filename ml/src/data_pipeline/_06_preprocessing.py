import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

PROJECT_ROOT = os.getcwd()
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data/preprocess")
UTILITY_DIR = os.path.join(PROJECT_ROOT, "utility")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(UTILITY_DIR, exist_ok=True)


def preprocessing(feature_df):
    # Columns to drop (example of non-relevant columns)
    drop_cols = ['airborne', 'feathers', 'domestic', 'aquatic', 'fins', 'catsize', 'class_type', 'class_name']

    X = feature_df.drop(columns=drop_cols)
    y = feature_df['class_name']

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    x_train_numeric = X_train.drop(columns=['animal_name']).reset_index(drop=True)
    x_test_numeric = X_test.drop(columns=['animal_name']).reset_index(drop=True)

    # scaling
    scaler = StandardScaler()
    x_train_scaled = pd.DataFrame(scaler.fit_transform(x_train_numeric), columns=x_train_numeric.columns)
    x_test_scaled = pd.DataFrame(scaler.transform(x_test_numeric), columns=x_test_numeric.columns) # keep original index
    
    # Reset indexes on categorical parts too
    animal_name_train = X_train['animal_name'].reset_index(drop=True)
    animal_name_test = X_test['animal_name'].reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    # Add them back
    x_train_scaled['animal_name'] = animal_name_train
    x_train_scaled['class_name'] = y_train

    x_test_scaled['animal_name'] = animal_name_test
    x_test_scaled['class_name'] = y_test

    # Save combined dataset
    x_train_scaled.to_csv(f"{OUTPUT_DIR}/train_preprocessed.csv", index=False)
    x_test_scaled.to_csv(f"{OUTPUT_DIR}/test_preprocessed.csv", index=False)

    full_df = pd.concat([x_train_scaled, x_test_scaled], axis=0).reset_index(drop=True)
    full_df.to_csv(os.path.join(OUTPUT_DIR, "full_preprocessed_df.csv"), index=False)

    joblib.dump(scaler, f"{UTILITY_DIR}/scaler.pkl")

    print("Preprocessing complete. Files saved in:", OUTPUT_DIR)
    
    return full_df
