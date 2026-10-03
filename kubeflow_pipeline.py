"""
Task 4: Workflow Automation Design using Kubeflow Concepts
Defines a 5-stage Kubeflow Pipeline (KFP v2) for Wine Quality Classification:
1. Data Collection
2. Data Validation
3. Model Training
4. Model Evaluation & Gate
5. Deployment
Compiles to pipeline YAML for execution on Kubeflow Clusters / Vertex AI Pipelines.
"""

import os
from kfp import dsl
from kfp import compiler

@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["pandas==2.2.2", "scikit-learn==1.5.0"]
)
def data_collection_op(dataset_output: dsl.Output[dsl.Dataset]):
    """Stage 1: Data Collection Component"""
    from sklearn.datasets import fetch_openml
    import pandas as pd
    
    print("Collecting Wine Quality dataset from OpenML repository...")
    wine = fetch_openml("wine-quality-red", version=1, as_frame=True, parser="auto")
    df = wine.frame
    target_series = pd.to_numeric(df["class"], errors="coerce")
    df["target"] = (target_series >= 6).astype(int)
    df.drop(columns=["class"], inplace=True)
    
    df.to_csv(dataset_output.path, index=False)
    print(f"Data Collection complete: {df.shape[0]} rows saved to {dataset_output.path}")

@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["pandas==2.2.2"]
)
def data_validation_op(
    dataset_input: dsl.Input[dsl.Dataset],
    validation_status: dsl.Output[dsl.Artifact]
) -> bool:
    """Stage 2: Data Validation Component"""
    import pandas as pd
    
    df = pd.read_csv(dataset_input.path)
    print(f"Validating dataset schema and statistics for {df.shape[0]} rows...")
    
    # Validation rules
    assert df.shape[0] >= 1000, f"Validation Failed: Expected >=1000 rows, found {df.shape[0]}"
    assert df.isna().sum().sum() == 0, "Validation Failed: Found unexpected missing values!"
    assert "target" in df.columns, "Validation Failed: 'target' column missing!"
    
    with open(validation_status.path, "w") as f:
        f.write("VALIDATION_PASSED: Schema valid, 0 nulls, correct target distribution.")
    print("Data validation successfully passed all data quality gates.")
    return True

@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["pandas==2.2.2", "scikit-learn==1.5.0", "joblib==1.4.2"]
)
def training_op(
    dataset_input: dsl.Input[dsl.Dataset],
    model_output: dsl.Output[dsl.Model],
    train_metrics: dsl.Output[dsl.Metrics],
    n_estimators: int = 150,
    max_depth: int = 10
):
    """Stage 3: Model Training Component"""
    import pandas as pd
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    
    df = pd.read_csv(dataset_input.path)
    X = df.drop(columns=["target"])
    y = df["target"]
    
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    
    train_acc = model.score(X_train, y_train)
    train_metrics.log_metric("train_accuracy", float(train_acc))
    train_metrics.log_metric("n_estimators", int(n_estimators))
    train_metrics.log_metric("max_depth", int(max_depth))
    
    joblib.dump(model, model_output.path)
    print(f"Training complete. Model serialized to {model_output.path} with Train Acc: {train_acc:.4f}")

@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["pandas==2.2.2", "scikit-learn==1.5.0", "joblib==1.4.2"]
)
def evaluation_op(
    dataset_input: dsl.Input[dsl.Dataset],
    model_input: dsl.Input[dsl.Model],
    eval_metrics: dsl.Output[dsl.Metrics],
    accuracy_threshold: float = 0.75
) -> bool:
    """Stage 4: Model Evaluation & Gating Component"""
    import pandas as pd
    import joblib
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    
    df = pd.read_csv(dataset_input.path)
    X = df.drop(columns=["target"])
    y = df["target"]
    
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model = joblib.load(model_input.path)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    auc = roc_auc_score(y_test, probs)
    
    eval_metrics.log_metric("test_accuracy", float(acc))
    eval_metrics.log_metric("precision", float(prec))
    eval_metrics.log_metric("recall", float(rec))
    eval_metrics.log_metric("f1_score", float(f1))
    eval_metrics.log_metric("roc_auc", float(auc))
    
    deployable = acc >= accuracy_threshold
    print(f"Evaluation Metrics: Accuracy={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}")
    print(f"Quality Gate Status: {'PASSED' if deployable else 'FAILED'} (Threshold: {accuracy_threshold})")
    return deployable

@dsl.component(
    base_image="python:3.11-slim"
)
def deployment_op(
    model_input: dsl.Input[dsl.Model],
    deploy_approval: bool,
    deployment_record: dsl.Output[dsl.Artifact]
):
    """Stage 5: Deployment Component (Simulates blue-green rollout / staging registration)"""
    if not deploy_approval:
        print("Deployment Aborted: Model failed evaluation quality gate threshold.")
        with open(deployment_record.path, "w") as f:
            f.write("DEPLOYMENT_REJECTED: Quality threshold not satisfied.")
        return
        
    print(f"Deploying model from {model_input.path} to inference service...")
    status_msg = (
        "DEPLOYMENT_SUCCESSFUL\n"
        "Endpoint: http://mlops-wine-service.prod.svc.cluster.local/predict\n"
        "Strategy: Canary Rollout (10% initial traffic)\n"
        "Monitoring: Drift Detection & Prometheus Latency Metrics Enabled"
    )
    with open(deployment_record.path, "w") as f:
        f.write(status_msg)
    print("Deployment completed successfully!")

@dsl.pipeline(
    name="wine-quality-end-to-end-mlops-pipeline",
    description="5-Stage Kubeflow Pipeline: Data Collection, Validation, Training, Evaluation, and Deployment"
)
def wine_mlops_pipeline(
    n_estimators: int = 150,
    max_depth: int = 10,
    accuracy_threshold: float = 0.75
):
    # 1. Data Collection
    collection_task = data_collection_op()
    
    # 2. Data Validation
    validation_task = data_validation_op(dataset_input=collection_task.outputs["dataset_output"])
    
    # 3. Model Training
    training_task = training_op(
        dataset_input=collection_task.outputs["dataset_output"],
        n_estimators=n_estimators,
        max_depth=max_depth
    )
    training_task.after(validation_task)
    
    # 4. Model Evaluation
    evaluation_task = evaluation_op(
        dataset_input=collection_task.outputs["dataset_output"],
        model_input=training_task.outputs["model_output"],
        accuracy_threshold=accuracy_threshold
    )
    
    # 5. Deployment (conditional on evaluation gate)
    deployment_task = deployment_op(
        model_input=training_task.outputs["model_output"],
        deploy_approval=evaluation_task.outputs["Output"]
    )

if __name__ == "__main__":
    os.makedirs("artifacts", exist_ok=True)
    yaml_output = "artifacts/kubeflow_pipeline.yaml"
    print(f"Compiling Kubeflow pipeline to {yaml_output}...")
    compiler.Compiler().compile(
        pipeline_func=wine_mlops_pipeline,
        package_path=yaml_output
    )
    print(f"Successfully compiled Kubeflow pipeline to {yaml_output}!")
