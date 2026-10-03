"""
Prepares historical feature data with wine_id entity and event_timestamp for Feast.
"""

import os
from datetime import datetime, timezone
import pandas as pd

def generate_feast_parquet():
    os.makedirs("feature_store/data", exist_ok=True)
    df = pd.read_csv("data/dataset_raw.csv")
    
    # Add entity and timestamps
    df["wine_id"] = range(1, len(df) + 1)
    df["event_timestamp"] = datetime.now(timezone.utc)
    df["created"] = datetime.now(timezone.utc)
    
    parquet_path = "feature_store/data/wine_features.parquet"
    df.to_parquet(parquet_path, index=False)
    print(f"Feast data source generated at {parquet_path} ({len(df)} records).")

if __name__ == "__main__":
    generate_feast_parquet()
