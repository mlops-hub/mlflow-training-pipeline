from src.data_pipeline._01_ingestion import ingestion
from src.data_pipeline._02_validation import check_validation
from src.data_pipeline._03_eda import eda
from src.data_pipeline._04_cleaning import cleaning
from src.data_pipeline._05_feature_engg import feature_engg, prepare_data_for_feast
from src.data_pipeline._06_preprocessing import preprocessing

# load dataset
merged_df = ingestion()
print('1. ingestion: ', merged_df.head(2))
# mlflow.log_param("raw_rows", merged_df.shape[0])

# validated dataset
validated_df = check_validation(merged_df)
while validated_df is None:
    print("Validation is failed")
    # go to eda
    eda_df = eda(merged_df)

    # lets clean dataset
    cleaned_df = cleaning(eda_df)

    # validate dataset
    validated_df = check_validation(cleaned_df)
    print('check error: ', validated_df)
        
    # mlflow.log_param("validated_rows", validated_df.shape[0])

# proceed only when validation passes
if validated_df is not None:
    feature_df = feature_engg(validated_df)
    final_df, X_train, X_test, y_train, y_test = preprocessing(feature_df)

    prepare_data_for_feast(final_df)
