from ml.src.data_pipeline._01_ingestion import ingestion
from ml.src.data_pipeline._02_validation import check_schema, check_quality
from ml.src.data_pipeline._04_cleaning import cleaning
from ml.src.data_pipeline._05_feature_engg import feature_engg, prepare_data_for_feast


def data_pipeline():
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

    drop_cols = ['airborne', 'feathers', 'domestic', 'aquatic', 'fins', 'catsize', 'class_type']
    prepared_df = feature_df.drop(columns=drop_cols)

    # 6. save prepared data in feast - feature-store
    prepare_data_for_feast(prepared_df )


if __name__ == "__main__":
    data_pipeline()
