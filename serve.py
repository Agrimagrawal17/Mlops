import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import mlflow
import mlflow.sklearn
import pandas as pd
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
TRACKING_URI = os.getenv("MLFLOW_TRACKING", "http://127.0.0.1:5000")
MODEL_URI = "models:/house-price-predictor@champion"
FEATURES = ["sqft", "bedrooms", "bathrooms", "age_years", "garage", "location_score"]

mlflow.set_tracking_uri(TRACKING_URI)

# Model load
model = mlflow.sklearn.load_model(MODEL_URI)

app = FastAPI(title="House Price Predictor")


class HouseFeatures(BaseModel):
    sqft: float = Field(..., gt=0, le=20000)
    bedrooms: int = Field(..., gt=0, le=20)
    bathrooms: int = Field(..., gt=0, le=200)
    age_years: int = Field(..., ge=0, le=100)
    garage: int = Field(..., ge=0, le=10)
    location_score: int = Field(..., ge=1, le=10)


@app.get("/health")
def health():
    return {"status": "healthy", "model": MODEL_URI}


@app.post("/predict")
def predict(features: HouseFeatures):
    input_df = pd.DataFrame([features.model_dump()], columns=FEATURES)
    prediction = model.predict(input_df)[0]
    return {"predicted_price": round(float(prediction), 2)}


# Static files mount
static_dir = BASE_DIR / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
def frontend():
    index_path = BASE_DIR / "static" / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "Frontend file static/index.html not found"}