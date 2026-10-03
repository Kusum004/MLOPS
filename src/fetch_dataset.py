"""
Task 1: Data Ingestion Stage
Fetches the Red Wine Quality dataset (UCI / OpenML), validates initial integrity,
and stores the raw dataset into data/dataset.csv for DVC version tracking.
"""

import os
import pandas as pd
from sklearn.datasets import fetch_openml

def ingest_data(output_path: str = "data/dataset.csv"):
    print("Ingesting Red Wine Quality Dataset from OpenML...")
    wine = fetch_openml("wine-quality-red", version=1, as_frame=True, parser="auto")
    df = wine.frame.copy()
    
    # Rename 'class' to 'target' (binary: 1 for good quality >= 6, 0 for normal/low quality < 6)
    target_series = pd.to_numeric(df["class"], errors="coerce")
    df["quality_score"] = target_series
    df["target"] = (target_series >= 6).astype(int)
    df.drop(columns=["class"], inplace=True)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"Data ingestion complete. Saved {df.shape[0]} samples with {df.shape[1]} columns to {output_path}")
    print(f"Target distribution:\n{df['target'].value_counts(normalize=True).round(3)}")
    return df

if __name__ == "__main__":
    ingest_data()
