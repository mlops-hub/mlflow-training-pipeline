from src.data_pipeline._01_ingestion import ingestion
from src.data_pipeline._02_validation import check_schema, check_quality
from src.data_pipeline._04_cleaning import cleaning
from src.data_pipeline._05_feature_engg import feature_engg, prepare_data_for_feast
from src.data_pipeline._06_preprocessing import preprocessing

# 1. load dataset
raw_df = ingestion()


# 2. initial schema validation
initial_schema = check_schema(raw_df)
if initial_schema is None:
    print("Schema validation failed. Check logs.")

    # 3. lets clean dataset
    cleaned_df = cleaning(raw_df)

    # 4. quality check
    validated_df = check_quality(cleaned_df)     
    print('validated-df: ', validated_df)
    if validated_df is None:
        raise ValueError("Quality validation failed. Check logs.")


# 5. Proceed to feature engg 
feature_df = feature_engg(validated_df)

# 6. Preprocess data / data preparation
preprocessing_df = preprocessing(feature_df)
print('columns: ', preprocessing_df.columns.tolist())


# 7. save preprocessd data in feast - feature-store
prepare_data_for_feast(preprocessing_df)

# columns:  ['hair', 'eggs', 'milk', 'predator', 'toothed', 'backbone', 'breathes', 'venomous', 'legs', 'tail', 'can_fly', 'can_swim', 'is_domestic_pet', 'animal_name', 'class_name']

# Data preparation complete and saved successfully.
# Final data columns: ['hair', 'eggs', 'milk', 'predator', 'toothed', 'backbone', 'breathes', 'venomous', 'legs', 'tail', 'can_fly', 'can_swim', 'is_domestic_pet', 'animal_name', 'class_name', 'event_timestamp']
# Column names:  {(119, 16)}