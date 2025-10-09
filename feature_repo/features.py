# features.py
from feast import Entity, FeatureView, Field, FileSource, ValueType, FeatureService
from feast.types import Int64, String, Bool
from datetime import timedelta

# Define an entity for 'amimal'
animal_entity = Entity(
    name="animal_name",
    value_type=ValueType.STRING,  # since animal names are strings, not integers
    description="Unique name of the animal"
)
print('entity: ', animal_entity)

animal_source = FileSource(
    path="data/preprocessed_data.parquet", # Path to the preprocessed Parquet file
    timestamp_field="event_timestamp", # Column in your data indicating when the event occurred
)
print('animal_source: ', animal_source)

animal_feature_fv = FeatureView(
    name="animal_feature_fv",
    entities=[animal_entity],
    ttl=timedelta(days=365), # A long TTL as these are mostly static features
    schema=[
        Field(name="animal_name", dtype=String),
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
        Field(name="class_name", dtype=String),
        Field(name="can_fly", dtype=Int64),
        Field(name="can_swim", dtype=Int64),
        Field(name="is_domestic_pet", dtype=Int64),
    ],
    source=animal_source,
    online=True
)

# Define Feature Service
animal_feature_service = FeatureService(
    name="animal_feature_service",
    features=[animal_feature_fv]
)

#  testing or logging the feature values
feature_count = len(animal_feature_fv.schema)
print('total features in feast: ', feature_count)

