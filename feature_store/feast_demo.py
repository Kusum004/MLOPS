"""
Task 5 Part A: Feast Feature Store Demonstration
Demonstrates:
1. Identifying three critical features: alcohol, volatile_acidity, sulphates.
2. Historical feature retrieval (Point-in-Time correct joins for training).
3. Materialization to Online Store.
4. Real-time sub-millisecond feature retrieval for inference.
"""

import os
from datetime import datetime, timezone
import pandas as pd
from feast import FeatureStore

def run_feast_demo():
    print("\n" + "="*80)
    print(" " * 22 + "TASK 5 PART A: FEAST FEATURE STORE DEMO")
    print("="*80)

    repo_path = os.path.abspath("feature_store")
    store = FeatureStore(repo_path=repo_path)

    # 1. Identified Features
    print("""
1. Three Features Identified for ML Model:
   a) 'alcohol' (Float32): Key chemical indicator of wine body and fermentation quality.
   b) 'volatile_acidity' (Float32): Level of acetic acid; high levels indicate spoilage/sourness.
   c) 'sulphates' (Float32): Antimicrobial and antioxidant additive preserving freshness.
""")

    # 2. Historical Feature Retrieval (Offline Store -> Training)
    print("2. Retrieving Historical Features (Point-in-Time Correct Join for Training)...")
    entity_df = pd.DataFrame({
        "wine_id": [1, 2, 3],
        "event_timestamp": [datetime.now(timezone.utc)] * 3
    })

    features_to_fetch = [
        "wine_features:alcohol",
        "wine_features:volatile_acidity",
        "wine_features:sulphates",
        "wine_features:citric_acid",
    ]

    training_data = store.get_historical_features(
        entity_df=entity_df,
        features=features_to_fetch
    ).to_df()
    print("Historical Features retrieved successfully for training:")
    print(training_data[["wine_id", "alcohol", "volatile_acidity", "sulphates"]])

    # 3. Materialize to Online Store
    print("\n3. Materializing Features to Online Store (SQLite/Redis)...")
    end_date = datetime.now(timezone.utc)
    start_date = datetime(2020, 1, 1, tzinfo=timezone.utc)
    store.materialize(start_date=start_date, end_date=end_date)
    print("Features successfully materialized to online key-value store!")

    # 4. Online Feature Retrieval (Online Store -> Real-time Inference)
    print("\n4. Retrieving Online Features for Real-Time Inference (Low Latency)...")
    entity_rows = [{"wine_id": 1}, {"wine_id": 2}]
    online_response = store.get_online_features(
        features=features_to_fetch,
        entity_rows=entity_rows
    ).to_dict()

    online_df = pd.DataFrame(online_response)
    print("Online Features for real-time serving:")
    print(online_df[["wine_id", "alcohol", "volatile_acidity", "sulphates"]])

    print("\n" + "="*80)
    print("HOW FEAST PREVENTS TRAINING-SERVING SKEW:")
    print("="*80)
    print("""
- Unified Feature Definition: The exact same FeatureView code is used for both training and inference.
- Point-in-time Joins (No Data Leakage): Offline retrieval joins features using historical timestamps.
- Zero Skew: Training sees the exact same transformation logic as the real-time inference service.
""")

if __name__ == "__main__":
    run_feast_demo()
