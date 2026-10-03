"""
Task 4: Automated Workflow Execution Runner
Simulates the 5-stage Kubeflow Pipeline DAG locally:
1. Data Collection
2. Data Validation
3. Model Training
4. Model Evaluation & Gating
5. Deployment / Serving Registration
"""

import os
import json
import time
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def run_workflow_pipeline(
    data_path: str = "data/dataset.csv",
    accuracy_threshold: float = 0.75,
    output_dir: str = "artifacts/pipeline_run"
):
    os.makedirs(output_dir, exist_ok=True)
    pipeline_log = {
        "pipeline_name": "wine-quality-end-to-end-mlops-pipeline",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stages": {}
    }

    print("\n" + "="*80)
    print(" " * 20 + "STARTING AUTOMATED WORKFLOW PIPELINE")
    print("="*80)

    # -------------------------------------------------------------
    # Stage 1: Data Collection
    # -------------------------------------------------------------
    print("\n[Stage 1/5] Running Data Collection...")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data source {data_path} not found.")
    df = pd.read_csv(data_path)
    stage1_info = {
        "status": "SUCCESS",
        "samples_collected": int(df.shape[0]),
        "total_columns": int(df.shape[1]),
        "features": [c for c in df.columns if c not in ["target", "quality_score"]]
    }
    pipeline_log["stages"]["1_data_collection"] = stage1_info
    print(f" -> Collected {stage1_info['samples_collected']} records with {len(stage1_info['features'])} feature columns.")

    # -------------------------------------------------------------
    # Stage 2: Data Validation
    # -------------------------------------------------------------
    print("\n[Stage 2/5] Running Data Validation...")
    null_count = int(df.isna().sum().sum())
    has_target = "target" in df.columns
    row_count_valid = len(df) >= 1000

    validation_passed = (null_count == 0) and has_target and row_count_valid
    stage2_info = {
        "status": "PASSED" if validation_passed else "FAILED",
        "missing_values": null_count,
        "target_column_present": has_target,
        "sufficient_rows": row_count_valid
    }
    pipeline_log["stages"]["2_data_validation"] = stage2_info
    if not validation_passed:
        raise ValueError(f"Pipeline stopped at Stage 2: Data Validation Failed! {stage2_info}")
    print(f" -> Data Validation Passed: 0 missing values, valid schema, rows={len(df)}")

    # -------------------------------------------------------------
    # Stage 3: Training
    # -------------------------------------------------------------
    print("\n[Stage 3/5] Running Model Training (Random Forest)...")
    drop_cols = ["target"]
    if "quality_score" in df.columns:
        drop_cols.append("quality_score")
        
    X = df.drop(columns=drop_cols)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    rf_model = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)
    rf_model.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, rf_model.predict(X_train))
    model_path = os.path.join(output_dir, "deployed_model.joblib")
    joblib.dump(rf_model, model_path)

    stage3_info = {
        "status": "SUCCESS",
        "model_type": "RandomForestClassifier",
        "hyperparameters": {"n_estimators": 150, "max_depth": 10},
        "train_accuracy": round(float(train_acc), 4),
        "model_artifact": model_path
    }
    pipeline_log["stages"]["3_training"] = stage3_info
    print(f" -> Training Complete. Train Accuracy: {train_acc:.4f}. Model saved to {model_path}")

    # -------------------------------------------------------------
    # Stage 4: Evaluation & Quality Gate
    # -------------------------------------------------------------
    print("\n[Stage 4/5] Running Model Evaluation & Automated Quality Gate...")
    test_preds = rf_model.predict(X_test)
    test_probs = rf_model.predict_proba(X_test)[:, 1]

    test_acc = accuracy_score(y_test, test_preds)
    precision = precision_score(y_test, test_preds, zero_division=0)
    recall = recall_score(y_test, test_preds, zero_division=0)
    f1 = f1_score(y_test, test_preds, zero_division=0)
    roc_auc = roc_auc_score(y_test, test_probs)

    passes_gate = test_acc >= accuracy_threshold
    stage4_info = {
        "status": "PASSED" if passes_gate else "REJECTED",
        "metrics": {
            "test_accuracy": round(float(test_acc), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1_score": round(float(f1), 4),
            "roc_auc": round(float(roc_auc), 4)
        },
        "threshold_required": accuracy_threshold,
        "gate_passed": passes_gate
    }
    pipeline_log["stages"]["4_evaluation"] = stage4_info
    print(f" -> Evaluation Metrics: Acc={test_acc:.4f}, F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, AUC={roc_auc:.4f}")
    print(f" -> Quality Gate Status: {'PASSED (Ready for deployment)' if passes_gate else 'REJECTED'}")

    # -------------------------------------------------------------
    # Stage 5: Deployment
    # -------------------------------------------------------------
    print("\n[Stage 5/5] Running Deployment Stage...")
    if passes_gate:
        stage5_info = {
            "status": "DEPLOYED",
            "endpoint": "http://127.0.0.1:8000/predict",
            "strategy": "Blue-Green / Staging Promotion",
            "message": "Model deployed and ready for inference."
        }
        print(" -> Deployment Triggered: Model approved and registered for serving.")
    else:
        stage5_info = {
            "status": "BLOCKED",
            "reason": "Failed accuracy threshold gate."
        }
        print(" -> Deployment Blocked: Quality gate check failed.")
    
    pipeline_log["stages"]["5_deployment"] = stage5_info

    # Save pipeline execution summary
    log_file = os.path.join(output_dir, "pipeline_execution_log.json")
    with open(log_file, "w") as f:
        json.dump(pipeline_log, f, indent=4)

    print("\n" + "="*80)
    print(f"PIPELINE EXECUTION COMPLETE. Execution log saved to {log_file}")
    print("="*80)
    return pipeline_log

if __name__ == "__main__":
    run_workflow_pipeline()
