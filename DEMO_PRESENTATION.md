# 5-Minute MLOps Demo Presentation Guide
**Assignment Title**: Design and Implement a Basic MLOps Pipeline for an ML Application  
**Student**: Kusum Gajare  
**Dataset**: Red Wine Quality Classification  
**Repository**: [https://github.com/Kusum004/MLOPS.git](https://github.com/Kusum004/MLOPS.git)  

---

## Presentation Timing & Slide Breakdown (Total: 5 Minutes)

| Time | Slide / Topic | What to Say (Speaker Notes) | Live Demo Action / Terminal Command |
|---|---|---|---|
| **0:00 - 1:00** | **Slide 1: Overview & Lifecycle Analysis (Task 1)** | "Good morning/afternoon. Today I present our complete MLOps pipeline for Wine Quality classification. We designed a continuous 6-stage lifecycle: Ingestion from OpenML, preprocessing with domain feature engineering, versioning with DVC, experiment tracking with MLflow, Kubeflow pipeline automation, and Feast feature store management." | Open `MLOps lifecycle.png` and explain the 6 stages. |
| **1:00 - 2:00** | **Slide 2: Experiment Tracking with MLflow (Task 2)** | "In Task 2, we tracked experiments across 4 algorithms: KNN, SVM, Random Forest, and Logistic Regression. For every run, we captured hyperparameters, train/test accuracy, precision, recall, F1, ROC-AUC, and visual artifacts like confusion matrices and ROC curves. KNN and Random Forest delivered top performance with test accuracy above 80% and ROC-AUC of 0.89." | Run in terminal:<br>`python compare_experiments.py`<br>Show `artifacts/mlflow_model_comparison.png`. |
| **2:00 - 3:00** | **Slide 3: Dataset Versioning with DVC (Task 3)** | "In Task 3, we solved data drift and reproducibility challenges using DVC. We created two dataset versions: Version 1.0 contains the raw ingested data (Git tag `v1.0`), while Version 2.0 adds 3 engineered chemical ratios, outlier Winsorization, and StandardScaler normalization (Git tag `v2.0`). Git only tracks the lightweight 100-byte `.dvc` hash file while DVC manages the physical storage." | Run in terminal:<br>`python src/compare_dvc_versions.py`<br>`git tag -n` |
| **3:00 - 4:00** | **Slide 4: Workflow Automation with Kubeflow (Task 4)** | "Task 4 automates our pipeline using Kubeflow Pipeline (KFP v2) concepts. The pipeline consists of 5 modular containerized stages: Data Collection, Validation, Training, Evaluation, and Deployment. Automated quality gates ensure that if the test accuracy falls below 0.75, the model is rejected and deployment is blocked, guaranteeing production safety." | Run in terminal:<br>`python kubeflow_pipeline.py`<br>`python run_pipeline.py` |
| **4:00 - 5:00** | **Slide 5: Feast Feature Store & Cloud Platform Study (Task 5)** | "In Task 5, we addressed training-serving skew using Feast Feature Store. We identified alcohol, volatile acidity, and sulphates as key features, demonstrating both historical point-in-time joins for training and sub-millisecond retrieval from an online SQLite store for real-time serving. Finally, we compared AWS SageMaker and GCP Vertex AI, highlighting Vertex AI's native Kubeflow compatibility." | Run in terminal:<br>`python feature_store/feast_demo.py`<br>Show FastAPI inference result. |

---

## Live Commands Cheat Sheet for Demo

```bash
# 1. Show MLflow Model Comparison Table & Chart
python compare_experiments.py

# 2. Show DVC Dataset Versions Comparison
python src/compare_dvc_versions.py

# 3. Compile and Run the 5-Stage Automated Workflow Pipeline
python kubeflow_pipeline.py
python run_pipeline.py

# 4. Demonstrate Feast Feature Store (Point-in-Time Joins & Online Retrieval)
python feature_store/feast_demo.py
```
