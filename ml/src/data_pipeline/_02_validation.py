import os
import pandas as pd
import pandera.pandas as pa
from pandera import Column, Check
from pandera.dtypes import Int

PROJECT_ROOT = os.getcwd()
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "ml/logs/validation")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1.Define the schema
class AnimalSchema (pa.DataFrameModel):
    animal_name: str
    hair: int = pa.Field(isin=[0, 1])
    feathers: int = pa.Field(isin=[0, 1])
    eggs: int = pa.Field(isin=[0, 1])
    milk: int = pa.Field(isin=[0, 1])
    airborne: int = pa.Field(isin=[0, 1])
    aquatic: int = pa.Field(isin=[0, 1])
    predator: int = pa.Field(isin=[0, 1])
    toothed: int = pa.Field(isin=[0, 1])
    backbone: int = pa.Field(isin=[0, 1])
    breathes: int = pa.Field(isin=[0, 1])
    venomous: int = pa.Field(isin=[0, 1])
    fins: int = pa.Field(isin=[0, 1])
    legs: int = pa.Field(isin=[0, 2, 3, 4, 5, 6, 7, 8])
    tail: int = pa.Field(isin=[0, 1])
    domestic: int = pa.Field(isin=[0, 1])
    catsize: int = pa.Field(isin=[0, 1])
    class_type: int = pa.Field(ge=1, le=7)
    class_name: str = pa.Field(isin=['Mammal', 'Bird', 'Amphibian', 'Bug', 'Reptile', 'Invertebrate', 'Fish'])


def check_schema(df):
    try:
        validated_df = AnimalSchema.validate(df, lazy=True) # lazy=True to catch all errors
        print(validated_df.head(3))
        print("✅ Data validation successful!")
        return validated_df
    except pa.errors.SchemaErrors as err:
        print("Data validation failed:")
        print(err.failure_cases)
        # log error
        err.failure_cases.to_csv(f"{OUTPUT_DIR}/error_df.csv", index=False)
        return None

# quality / bussiness validation
def check_quality(df: pd.DataFrame) -> pd.DataFrame | None:
    errors = []
    if df.duplicated().sum():
        errors.append("Duplicate rows found")
    if (df["legs"] < 0).any():
        errors.append("Negative values found in 'legs' column.")
    if df["class_type"].nunique() < 2:
        errors.append("Only one class_type present (bad for classification).")
    if df.isnull().sum().sum() > 0:
        errors.append("Null values still present after cleaning.")

    print(f"Final shape: {df.shape}")
    print(f"Missing values:\n{df.isnull().sum()}")
    print(f"Duplicates: {df.duplicated().sum()}")
    print(f"Data types:\n{df.dtypes}")

    if errors:
        error_path = os.path.join(OUTPUT_DIR, "quality_errors.txt")
        with open(error_path, "w") as f:
            f.write("\n".join(errors))
        print(f"⚠️ Errors saved to: {error_path}")
        return None
    else:
        print("✅ Data Quality is validated. Proceed to next steps.")
        return df