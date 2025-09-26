import pandera.pandas as pa
from pandera import Column, Check
from pandera.dtypes import Int
import pandas as pd
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 1.Define the schema
schema = pa.DataFrameSchema({
    "animal_name": Column(str, nullable=False),
    "hair": Column(Int, checks=[Check.isin([0, 1])], nullable=False),
    "feathers": Column(Int, checks=[Check.isin([0, 1])], nullable=False),
    "eggs": Column(Int, checks=[Check.isin([0, 1])], nullable=False),
    "milk": Column(Int, checks=[Check.isin([0, 1])], nullable=False),
    "airborne": Column(Int, checks=[Check.isin([0, 1])], nullable=False),
    "aquatic": Column(Int, checks=[Check.isin([0, 1])], nullable=False), 
    "predator": Column(Int, checks=[Check.isin([0, 1])], nullable=False),
    "toothed": Column(Int, checks=[Check.isin([0, 1])], nullable=False),
    "backbone": Column(Int, checks=[Check.isin([0, 1])], nullable=False),
    "breathes": Column(Int, checks=[Check.isin([0, 1])], nullable=False),
    "venomous": Column(Int, checks=[Check.isin([0, 1])], nullable=False),
    "fins": Column(Int, checks=[Check.isin([0, 1])], nullable=False),  
    "legs": Column(Int, checks=[Check.isin([0, 2, 4, 5, 6, 8])], nullable=False),
    "tail": Column(Int, checks=[Check.isin([0, 1])], nullable=False),
    "domestic": Column(Int, checks=[Check.isin([0, 1])], nullable=False), 
    "catsize": Column(Int, checks=[Check.isin([0, 1])], nullable=False),
    "class_type": Column(Int, [Check.ge(1), Check.le(7)], nullable=False),
    "class_name": Column(str, Check.isin(["Mammal", "Fish", "Amphibian", "Bird", "Invertebrate", "Bug", "Reptile"]), nullable=False),
})


def check_validation(df):
    try:
        validated_df = schema.validate(df, lazy=True) # lazy=True to catch all errors
        print(validated_df.head(3))
        print("✅ Data validation successful!")
        return validated_df
        
    except pa.errors.SchemaErrors as err:
        print("Data validation failed:")
        print(err.failure_cases)
        # log error
        output_dir = os.path.join(PROJECT_ROOT, "logs/validation")
        os.makedirs(output_dir, exist_ok=True)
        
        err.failure_cases.to_csv(f"{output_dir}/error_df.csv", index=False)
        print(f"Error saved to: {output_dir}/error_df.csv")

        return None


