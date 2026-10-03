"""
Task 2: Experiment Tracking using MLflow
Trains multiple classification models (KNN, SVM, Random Forest, Logistic Regression)
on the Wine Quality dataset and tracks parameters, metrics, plots, and models.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
)

# Use SQLite backend to ensure robust tracking and eliminate Windows URI path issues
DB_PATH = os.path.abspath("mlflow.db").replace("\\", "/")
mlflow.set_tracking_uri(f"sqlite:///{DB_PATH}")

def run_experiment(data_path: str = "data/dataset.csv", experiment_name: str = "wine-quality-cia1"):
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Drop quality_score if present to prevent target leakage
    drop_cols = ["target"]
    if "quality_score" in df.columns:
        drop_cols.append("quality_score")
        
    X = df.drop(columns=drop_cols)
    y = df["target"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features for distance/margin sensitive models (KNN, SVM, LogReg)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Dataset split: Train={X_train.shape}, Test={X_test.shape}")
    
    # Define models and their hyperparameters to track
    models = {
        "KNN": (
            KNeighborsClassifier(n_neighbors=7, weights="distance"),
            {"model_type": "KNeighborsClassifier", "n_neighbors": 7, "weights": "distance"},
            True # requires scaled
        ),
        "SVM": (
            SVC(C=1.5, kernel="rbf", probability=True, random_state=42),
            {"model_type": "SVC", "C": 1.5, "kernel": "rbf", "gamma": "scale"},
            True # requires scaled
        ),
        "RandomForest": (
            RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42),
            {"model_type": "RandomForestClassifier", "n_estimators": 150, "max_depth": 10},
            False # tree models handle unscaled natively
        ),
        "LogisticRegression": (
            LogisticRegression(max_iter=1000, C=1.0, random_state=42),
            {"model_type": "LogisticRegression", "max_iter": 1000, "C": 1.0, "solver": "lbfgs"},
            True # requires scaled
        ),
    }

    # Set MLflow experiment
    mlflow.set_experiment(experiment_name)
    os.makedirs("artifacts/plots", exist_ok=True)
    
    results = []

    print("\n" + "="*70)
    print("STARTING MLFLOW EXPERIMENT TRACKING: " + experiment_name)
    print("="*70)

    for name, (model, params, use_scaled) in models.items():
        with mlflow.start_run(run_name=name) as run:
            run_id = run.info.run_id
            print(f"\n[MLflow Run] Model: {name} (Run ID: {run_id[:8]})")
            
            # Select train/test data based on model sensitivity
            X_tr = X_train_scaled if use_scaled else X_train
            X_te = X_test_scaled if use_scaled else X_test
            
            # 1. Fit Model
            model.fit(X_tr, y_train)
            
            # 2. Predict
            train_preds = model.predict(X_tr)
            test_preds = model.predict(X_te)
            test_probs = model.predict_proba(X_te)[:, 1] if hasattr(model, "predict_proba") else None
            
            # 3. Calculate Metrics
            train_acc = accuracy_score(y_train, train_preds)
            test_acc = accuracy_score(y_test, test_preds)
            precision = precision_score(y_test, test_preds, zero_division=0)
            recall = recall_score(y_test, test_preds, zero_division=0)
            f1 = f1_score(y_test, test_preds, zero_division=0)
            roc_auc = roc_auc_score(y_test, test_probs) if test_probs is not None else float("nan")
            
            # 4. Log Parameters to MLflow
            mlflow.log_param("model_name", name)
            mlflow.log_param("num_features", X_train.shape[1])
            mlflow.log_param("train_samples", X_train.shape[0])
            mlflow.log_param("test_samples", X_test.shape[0])
            mlflow.log_param("scaled_input", use_scaled)
            mlflow.log_params(params)
            
            # 5. Log Metrics to MLflow
            mlflow.log_metric("train_accuracy", train_acc)
            mlflow.log_metric("test_accuracy", test_acc)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            if not np.isnan(roc_auc):
                mlflow.log_metric("roc_auc", roc_auc)
            
            # 6. Generate and Log Confusion Matrix Plot Artifact
            fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
            cm = confusion_matrix(y_test, test_preds)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Low Quality (0)", "Good Quality (1)"])
            disp.plot(cmap="Purples", ax=ax_cm)
            ax_cm.set_title(f"{name} - Confusion Matrix")
            plt.tight_layout()
            cm_plot_path = f"artifacts/plots/{name}_confusion_matrix.png"
            fig_cm.savefig(cm_plot_path)
            plt.close(fig_cm)
            mlflow.log_artifact(cm_plot_path, artifact_path="plots")
            
            # 7. Generate and Log ROC Curve Plot Artifact
            if test_probs is not None:
                fig_roc, ax_roc = plt.subplots(figsize=(5, 4))
                fpr, tpr, _ = roc_curve(y_test, test_probs)
                ax_roc.plot(fpr, tpr, color="#8b0000", lw=2, label=f"AUC = {roc_auc:.4f}")
                ax_roc.plot([0, 1], [0, 1], "k--")
                ax_roc.set_xlabel("False Positive Rate")
                ax_roc.set_ylabel("True Positive Rate")
                ax_roc.set_title(f"{name} - ROC Curve")
                ax_roc.legend(loc="lower right")
                plt.tight_layout()
                roc_plot_path = f"artifacts/plots/{name}_roc_curve.png"
                fig_roc.savefig(roc_plot_path)
                plt.close(fig_roc)
                mlflow.log_artifact(roc_plot_path, artifact_path="plots")

            # 8. Log Model Artifact
            mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                serialization_format="pickle",
                registered_model_name=f"WineQuality_{name}"
            )
            
            results.append({
                "Model": name,
                "Train Acc": f"{train_acc:.4f}",
                "Test Acc": f"{test_acc:.4f}",
                "Precision": f"{precision:.4f}",
                "Recall": f"{recall:.4f}",
                "F1 Score": f"{f1:.4f}",
                "ROC AUC": f"{roc_auc:.4f}" if not np.isnan(roc_auc) else "N/A",
            })
            print(f" -> {name} Logged: Test Acc={test_acc:.4f}, F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, AUC={roc_auc:.4f}")

    print("\n" + "="*70)
    print("EXPERIMENT COMPARISON SUMMARY")
    print("="*70)
    summary_df = pd.DataFrame(results)
    print(summary_df.to_string(index=False))
    summary_df.to_csv("artifacts/experiment_comparison.csv", index=False)
    print("\nSummary saved to artifacts/experiment_comparison.csv")
    return summary_df

if __name__ == "__main__":
    run_experiment()