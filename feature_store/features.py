# features.py
from feast import Entity, FeatureView, Field, FileSource, ValueType, FeatureService
from feast.types import Int64, String, Bool
from datetime import timedelta

# Define an entity for 'employee'
animal = Entity(
    name="animal_name",
    value_type=ValueType.STRING,  # since animal names are strings, not integers
    description="Unique name of the animal"
)

animal_preprocessed_source = FileSource(
    path="data/preprocessed_data.parquet", # Path to the preprocessed Parquet file
    timestamp_field="event_timestamp", # Column in your data indicating when the event occurred
)

animal_features_fv = FeatureView(
    name="animal_preprocessed_features",
    entities=[animal],
    ttl=timedelta(days=365), # A long TTL as these are mostly static features
    schema=[
        Field(name="hair", dtype=Int64),
        Field(name="eggs", dtype=Int64),
        Field(name="milk", dtype=Int64),
        Field(name="predator", dtype=Int64),
        Field(name="toothed", dtype=Int64),
        Field(name="backbone", dtype=Int64),
        Field(name="breathes", dtype=Int64),
        Field(name="venomous", dtype=Int64),
        Field(name="legs", dtype=Int64),
        Field(name="tail", dtype=Int64),
        Field(name="can_fly", dtype=Int64),
        Field(name="can_swim", dtype=Int64),
        Field(name="is_domestic_pet", dtype=Int64),
        # Keep entity feature for completeness (but usually entity is separate)
        Field(name="animal_name", dtype=String),
    ],
    source=animal_preprocessed_source,
)

# Define FeatureService
animal_features_service = FeatureService(
    name="animal_features_service",
    features=[animal_features_fv]
)


#  testing or logging the feature values
feature_count = len(animal_features_fv.schema)
print('total features in feast: ', feature_count)

