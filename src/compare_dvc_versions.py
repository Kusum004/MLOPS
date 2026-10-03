"""
Task 3.4: Compare DVC Dataset Versions & Explain Data Versioning Benefits
Compares v1.0 (Raw Wine Quality) and v2.0 (Preprocessed & Feature-Engineered).
"""

import os
import pandas as pd
import numpy as np

def compare_dataset_versions(v1_path: str = "data/dataset_raw.csv", v2_path: str = "data/dataset.csv"):
    if not os.path.exists(v1_path) or not os.path.exists(v2_path):
        print(f"Error: Make sure both {v1_path} and {v2_path} exist.")
        return

    df_v1 = pd.read_csv(v1_path)
    df_v2 = pd.read_csv(v2_path)

    print("\n" + "="*80)
    print(" " * 20 + "DVC DATASET VERSION COMPARISON (v1.0 vs v2.0)")
    print("="*80)

    comparison_metrics = [
        {"Metric": "Dataset Version", "v1.0 (Original Raw)": "Raw Wine Quality", "v2.0 (Modified & Preprocessed)": "Engineered & Scaled"},
        {"Metric": "Total Rows (Samples)", "v1.0 (Original Raw)": str(df_v1.shape[0]), "v2.0 (Modified & Preprocessed)": str(df_v2.shape[0])},
        {"Metric": "Total Columns", "v1.0 (Original Raw)": str(df_v1.shape[1]), "v2.0 (Modified & Preprocessed)": str(df_v2.shape[1])},
        {"Metric": "Feature Count", "v1.0 (Original Raw)": str(df_v1.shape[1] - 1), "v2.0 (Modified & Preprocessed)": str(df_v2.shape[1] - 1)},
        {"Metric": "Missing Values", "v1.0 (Original Raw)": str(df_v1.isna().sum().sum()), "v2.0 (Modified & Preprocessed)": str(df_v2.isna().sum().sum())},
        {"Metric": "Feature Engineering", "v1.0 (Original Raw)": "None (Raw attributes)", "v2.0 (Modified & Preprocessed)": "+3 Ratios (bound_SO2, acidity, sulphate/chloride)"},
        {"Metric": "Feature Scaling", "v1.0 (Original Raw)": "None (Unscaled physical units)", "v2.0 (Modified & Preprocessed)": "StandardScaler (Zero mean, unit variance)"},
        {"Metric": "Outlier Treatment", "v1.0 (Original Raw)": "Raw unclipped outliers", "v2.0 (Modified & Preprocessed)": "1% - 99% Percentile Capping (Winsorized)"},
    ]

    cmp_df = pd.DataFrame(comparison_metrics)
    print(cmp_df.to_string(index=False))

    os.makedirs("artifacts", exist_ok=True)
    cmp_df.to_csv("artifacts/dvc_version_comparison.csv", index=False)
    print(f"\nComparison table saved to artifacts/dvc_version_comparison.csv")

    # Column differences
    new_cols = set(df_v2.columns) - set(df_v1.columns)
    print(f"\nNew features added in v2.0: {list(new_cols)}")

    print("\n" + "="*80)
    print(" " * 22 + "BENEFITS OF DATA VERSIONING WITH DVC")
    print("="*80)
    print("""
1. Exact Experiment Reproducibility:
   - Traditional Git only tracks code. DVC couples specific git commits (code) with specific data hashes (data).
   - Any team member or CI/CD runner can run 'git checkout <tag> && dvc pull' to restore the exact dataset state that trained a given model.

2. Prevention of Git Bloat:
   - Git is not designed to store large binary or CSV datasets. DVC stores lightweight metadata pointer files (*.dvc) in Git, while the actual data resides in dedicated storage (S3, GCS, Azure Blob, or local network share).

3. Data Lineage and Auditing:
   - DVC pipelines track the complete lineage (DAG) from raw ingestion -> preprocessing -> training.
   - You can audit exactly what transformations or pipeline stages generated version 2.0 from version 1.0.

4. Seamless Collaboration & Cost Efficiency:
   - Storage deduplication: DVC uses content-addressable storage (MD5 hash). Unchanged files between versions are never duplicated, saving cloud storage costs and bandwidth.
""")

if __name__ == "__main__":
    compare_dataset_versions()
