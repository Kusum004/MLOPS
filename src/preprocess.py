"""
Task 1 & Task 3: Data Preprocessing Stage
Preprocesses the Red Wine Quality Dataset for DVC Version 2.0 (Modified dataset with added preprocessing).

Key Preprocessing & Feature Engineering:
1. Domain-specific Feature Engineering:
   - bound_sulfur_dioxide = total_sulfur_dioxide - free_sulfur_dioxide
   - acidity_ratio = fixed_acidity / (volatile_acidity + 1e-5)
   - sulphate_to_chloride_ratio = sulphates / (chlorides + 1e-5)
2. Outlier Capping (1% and 99% Winsorization to stabilize extreme outliers)
3. Standard Scaling (Z-score normalization on continuous physicochemical features)
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_data(input_path: str = "data/dataset.csv", output_path: str = "data/dataset.csv"):
    print(f"Loading raw Wine Quality dataset from {input_path}...")
    df = pd.read_csv(input_path)
    print(f"Original shape: {df.shape}")

    # Preserve target
    target_col = "target"
    if target_col not in df.columns:
        raise ValueError(f"'{target_col}' not found in {input_path}")
    
    y = df[target_col].copy()
    
    # Drop quality_score if present to prevent leakage
    drop_cols = [target_col]
    if "quality_score" in df.columns:
        drop_cols.append("quality_score")
    
    X = df.drop(columns=drop_cols).copy()

    # 1. Feature Engineering
    print("Engineering domain features (bound_sulfur_dioxide, acidity_ratio, sulphate_to_chloride_ratio)...")
    X["bound_sulfur_dioxide"] = np.maximum(0, X["total_sulfur_dioxide"] - X["free_sulfur_dioxide"])
    X["acidity_ratio"] = X["fixed_acidity"] / (X["volatile_acidity"] + 1e-5)
    X["sulphate_to_chloride_ratio"] = X["sulphates"] / (X["chlorides"] + 1e-5)

    # 2. Outlier Capping (1% - 99% quantiles)
    print("Applying Outlier Capping (1% - 99% quantiles)...")
    for col in X.select_dtypes(include=[np.number]).columns:
        q_low = X[col].quantile(0.01)
        q_high = X[col].quantile(0.99)
        X[col] = X[col].clip(q_low, q_high)

    # 3. Standard Scaling
    print("Normalizing features with StandardScaler...")
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(scaled_array, columns=X.columns)

    # Recombine with target
    df_processed = X_scaled.copy()
    df_processed[target_col] = y.values

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_processed.to_csv(output_path, index=False)
    print(f"Preprocessed dataset saved to {output_path}")
    print(f"Preprocessed shape: {df_processed.shape} (14 features + 1 target)")
    return df_processed

if __name__ == "__main__":
    preprocess_data()
