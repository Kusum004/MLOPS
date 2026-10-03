"""
Task 5: Feast Feature Store Definitions
Declares Wine Entity and FeatureView tracking physicochemical features:
1. alcohol
2. volatile_acidity
3. sulphates
"""

from datetime import timedelta
import os
from feast import Entity, Field, FeatureView, FileSource
from feast.types import Float32

# Define wine entity
wine_entity = Entity(
    name="wine_id",
    join_keys=["wine_id"],
    description="Unique identifier for wine batch/sample"
)

# FileSource pointing to offline parquet source
parquet_file_path = os.path.join(os.path.dirname(__file__), "data", "wine_features.parquet")
wine_source = FileSource(
    path=parquet_file_path,
    timestamp_field="event_timestamp",
    created_timestamp_column="created"
)

# FeatureView defining key features
wine_features_view = FeatureView(
    name="wine_features",
    entities=[wine_entity],
    ttl=timedelta(days=365),
    schema=[
        Field(name="alcohol", dtype=Float32),
        Field(name="volatile_acidity", dtype=Float32),
        Field(name="sulphates", dtype=Float32),
        Field(name="citric_acid", dtype=Float32),
        Field(name="fixed_acidity", dtype=Float32),
        Field(name="residual_sugar", dtype=Float32),
    ],
    online=True,
    source=wine_source,
    tags={"team": "mlops", "tier": "production"}
)
