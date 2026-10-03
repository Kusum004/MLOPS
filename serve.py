"""
Task 1 & Task 4: Deployment Stage - Model Serving API
FastAPI service to serve real-time predictions for Wine Quality Classification.
"""

import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="Wine Quality Prediction Service",
    description="MLOps Serving API for Wine Quality Classification",
    version="1.0.0"
)

MODEL_PATH = "artifacts/pipeline_run/deployed_model.joblib"
model = None

def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print(f"Model loaded successfully from {MODEL_PATH}")
    else:
        print(f"Warning: Model not found at {MODEL_PATH}. Run run_pipeline.py first.")

# Load immediately on module load
load_model()

class WineSample(BaseModel):
    fixed_acidity: float = Field(7.4, description="Fixed acidity (g/dm3)")
    volatile_acidity: float = Field(0.70, description="Volatile acidity (g/dm3)")
    citric_acid: float = Field(0.00, description="Citric acid (g/dm3)")
    residual_sugar: float = Field(1.9, description="Residual sugar (g/dm3)")
    chlorides: float = Field(0.076, description="Chlorides (g/dm3)")
    free_sulfur_dioxide: float = Field(11.0, description="Free SO2 (mg/dm3)")
    total_sulfur_dioxide: float = Field(34.0, description="Total SO2 (mg/dm3)")
    density: float = Field(0.9978, description="Density (g/cm3)")
    pH: float = Field(3.51, description="pH level")
    sulphates: float = Field(0.56, description="Sulphates (g/dm3)")
    alcohol: float = Field(9.4, description="Alcohol percentage (%)")
    # Derived engineered features (calculated automatically if not supplied)
    bound_sulfur_dioxide: float = Field(None, description="Total SO2 - Free SO2")
    acidity_ratio: float = Field(None, description="fixed_acidity / volatile_acidity")
    sulphate_to_chloride_ratio: float = Field(None, description="sulphates / chlorides")

@app.get("/")
def health():
    return {
        "status": "healthy",
        "service": "Wine Quality Prediction API",
        "model_loaded": model is not None
    }

@app.post("/predict")
def predict(sample: WineSample):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    
    data = sample.dict()
    # Compute feature engineering if null
    if data["bound_sulfur_dioxide"] is None:
        data["bound_sulfur_dioxide"] = max(0, data["total_sulfur_dioxide"] - data["free_sulfur_dioxide"])
    if data["acidity_ratio"] is None:
        data["acidity_ratio"] = data["fixed_acidity"] / (data["volatile_acidity"] + 1e-5)
    if data["sulphate_to_chloride_ratio"] is None:
        data["sulphate_to_chloride_ratio"] = data["sulphates"] / (data["chlorides"] + 1e-5)

    input_df = pd.DataFrame([data])
    
    # Reorder columns to match training features
    if hasattr(model, "feature_names_in_"):
        cols = model.feature_names_in_
        input_df = input_df[cols]

    prediction = int(model.predict(input_df)[0])
    probabilities = model.predict_proba(input_df)[0]
    good_quality_prob = float(probabilities[1])

    return {
        "prediction": prediction,
        "label": "High Quality Wine (>=6)" if prediction == 1 else "Normal / Low Quality Wine (<6)",
        "confidence": round(good_quality_prob if prediction == 1 else (1 - good_quality_prob), 4),
        "probabilities": {
            "low_quality": round(float(probabilities[0]), 4),
            "high_quality": round(float(probabilities[1]), 4)
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("serve:app", host="127.0.0.1", port=8000, reload=False)
