"""
Task 2.3: Compare Experimental Results using MLflow
Fetches run metrics from MLflow, prints comparative table, and generates comparison visualization.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient

# Use SQLite backend to match train.py
DB_PATH = os.path.abspath("mlflow.db").replace("\\", "/")
mlflow.set_tracking_uri(f"sqlite:///{DB_PATH}")

def compare_runs(experiment_name: str = "wine-quality-cia1"):
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    
    if not experiment:
        print(f"Experiment '{experiment_name}' not found. Please run train.py first.")
        return

    # Search all runs in experiment
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.test_accuracy DESC"]
    )
    
    records = []
    for r in runs:
        run_name = r.data.tags.get("mlflow.runName", r.info.run_id[:8])
        records.append({
            "Run ID": r.info.run_id[:8],
            "Model": run_name,
            "Train Accuracy": round(r.data.metrics.get("train_accuracy", 0.0), 4),
            "Test Accuracy": round(r.data.metrics.get("test_accuracy", 0.0), 4),
            "Precision": round(r.data.metrics.get("precision", 0.0), 4),
            "Recall": round(r.data.metrics.get("recall", 0.0), 4),
            "F1-Score": round(r.data.metrics.get("f1_score", 0.0), 4),
            "ROC AUC": round(r.data.metrics.get("roc_auc", 0.0), 4) if "roc_auc" in r.data.metrics else "N/A",
        })

    df_cmp = pd.DataFrame(records)
    print("\n" + "="*85)
    print(" " * 25 + f"MLFLOW EXPERIMENT COMPARISON: {experiment_name}")
    print("="*85)
    print(df_cmp.to_string(index=False))
    
    # Save CSV
    os.makedirs("artifacts", exist_ok=True)
    df_cmp.to_csv("artifacts/mlflow_comparison.csv", index=False)
    print("\nComparison table saved to artifacts/mlflow_comparison.csv")

    # Generate Comparative Bar Chart
    if not df_cmp.empty and "Test Accuracy" in df_cmp:
        metrics = ["Test Accuracy", "Precision", "Recall", "F1-Score"]
        plot_df = df_cmp.set_index("Model")[metrics]
        
        plt.figure(figsize=(10, 6))
        ax = plot_df.plot(kind="bar", figsize=(10, 6), colormap="viridis", edgecolor="black")
        plt.title(f"Model Performance Comparison (MLflow Experiment: {experiment_name})", fontsize=14, fontweight="bold")
        plt.xlabel("Model", fontsize=12)
        plt.ylabel("Score", fontsize=12)
        plt.ylim(0.8, 1.02)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.xticks(rotation=0)
        plt.legend(loc="lower right")
        plt.tight_layout()
        plot_path = "artifacts/mlflow_model_comparison.png"
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Comparative visualization saved to {plot_path}")

    # Best model highlight
    if not df_cmp.empty:
        best_run = df_cmp.iloc[0]
        print("\n" + "-"*85)
        print(f"Top Performing Model: {best_run['Model']} with Test Accuracy: {best_run['Test Accuracy']} and F1: {best_run['F1-Score']}")
        print("-"*85)

if __name__ == "__main__":
    compare_runs()
