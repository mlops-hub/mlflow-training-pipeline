import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def preprocessing(feature_df):
    # split train/test datasets
    X = feature_df.drop(columns=['airborne', 'feathers', 'domestic', 'aquatic', 'fins', 'catsize', 'class_type', 'class_name'])
    y = feature_df['class_name']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # scaling on numerical data only
    scaler = StandardScaler()

    # save animal_names and class_name separately
    animal_name_train = X_train['animal_name'].reset_index(drop=True)
    animal_name_test = X_test['animal_name'].reset_index(drop=True)

    # remove animal_names for scaling
    X_train_nameless = X_train.drop(columns=['animal_name'])
    X_test_nameless = X_test.drop(columns=['animal_name'])

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_nameless), 
        columns=X_train_nameless.columns
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test_nameless), 
        columns=X_test_nameless.columns
    )

    # save train/test files separetly
    output_dir = os.path.join(PROJECT_ROOT, "datasets/preprocess")
    os.makedirs(output_dir, exist_ok=True)

    # save preprocessed data in csv for later use.
    # Add 'animal_name' back
    X_train_with_name = X_train_scaled.copy()
    X_train_with_name["animal_name"] = animal_name_train
    X_test_with_name = X_test_scaled.copy()
    X_test_with_name["animal_name"] = animal_name_test

    X_train_scaled.to_csv(f"{output_dir}/X_train_dataset.csv", index_label=False)
    X_test_scaled.to_csv(f"{output_dir}/X_test_dataset.csv", index_label=False)
    y_train.to_csv(f"{output_dir}/y_train_dataset.csv", index_label=False)
    y_test.to_csv(f"{output_dir}/y_test_dataset.csv", index_label=False)

    # Save to CSV
    preprocessing_df = pd.concat([X_train_with_name, X_test_with_name], axis=0).reset_index(drop=True)
    preprocessing_df.to_csv(f"{output_dir}/preprocessing_df.csv", index=False)

    print(preprocessing_df.head(3))

    # save feature names
    utility_path = os.path.join(PROJECT_ROOT, "utility")
    os.makedirs(utility_path, exist_ok=True)
    
    joblib.dump(X_train_scaled.columns.to_list(), f'{utility_path}/feature_names.pkl')
    joblib.dump(scaler, f'{utility_path}/scaler.pkl')

    return preprocessing_df, X_train_scaled, X_test_scaled, y_train, y_test

