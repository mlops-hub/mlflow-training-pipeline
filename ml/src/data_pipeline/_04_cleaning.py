import os

PROJECT_ROOT = os.getcwd()

def cleaning(eda_df):
    # 1. Remove duplicate rows
    print(f"Removing {eda_df.duplicated().sum()} duplicates...")
    eda_df = eda_df.drop_duplicates()

    # 2. Handle missing values
    # 2.1 Remove row with missing animal_name (can't impute this)
    eda_df = eda_df.dropna(subset=['animal_name'])

    # Impute 'class_type' based on class_name whre possible
    class_mapping = {
        'Mammal': 1, 'Bird': 2, 'Reptile': 3, 'Fish': 4, 
        'Amphibian': 5, 'Bug': 6, 'Invertebrate': 7
    }
        
    # 2.2 Fill class_type from class_name
    mask = eda_df['class_type'].isnull() & eda_df['class_name'].notnull()
    eda_df.loc[mask, 'class_type'] = eda_df.loc[mask, 'class_name'].map(class_mapping)

    # 2.3 Fill class_name from class_type (reverse mapping)
    reverse_mapping = {v: k for k, v in class_mapping.items()}
    mask = eda_df['class_name'].isnull() & eda_df['class_type'].notnull()
    eda_df.loc[mask, 'class_name'] = eda_df.loc[mask, 'class_type'].map(reverse_mapping)

    # 2.4 Remove rows where both class_type and class_name are missing
    eda_df = eda_df.dropna(subset=['class_type', 'class_name'], how='all')

    # 2.5 Impute numerical missing values with mode (binary features)
    binary_columns = ['hair', 'feathers', 'eggs', 'milk', 'airborne', 'aquatic', 
        'predator', 'toothed', 'backbone', 'breathes', 'venomous', 
        'fins', 'tail', 'domestic', 'catsize']
        
    for col in binary_columns:
        if col in eda_df.columns and eda_df[col].isnull().any():
            mode_val = eda_df[col].mode()[0]
            eda_df[col] = eda_df[col].fillna(mode_val)


    # 2.6 Impute legs with median (non-binary feature)
    if 'legs' in eda_df.columns and eda_df['legs'].isnull().any():
        median_legs = eda_df['legs'].median()
        eda_df['legs'] = eda_df['legs'].fillna(median_legs)


    # 3. Fix data types
    # Convert float columns to int where appropriate
    int_columns = ['hair', 'venomous', 'legs', 'class_type']
    for col in int_columns:
        if col in eda_df.columns:
            eda_df[col] = eda_df[col].astype("int64")

    # 6. Save cleaned dataset
    output_dir = os.path.join(PROJECT_ROOT, "data/cleaned")
    os.makedirs(output_dir, exist_ok=True)

    eda_df.to_csv(f"{output_dir}/zoo_dataset_cleaned.csv", index=False)
    return eda_df

