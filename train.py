from io import StringIO
import boto3
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# S3 Client and Path configuration
s3 = boto3.client("s3")
BUCKET = "agrim-my-bucket-2026"
KEY = "processed/2026-09-25/Mlops_house_predication_clean_v1.csv"


def fectch_data():
    obj = s3.get_object(Bucket=BUCKET, Key=KEY)
    df = pd.read_csv(StringIO(obj["Body"].read().decode("utf-8")))
    return df


df = fectch_data()
print(f"Fetched shape: {df.shape}")

# Features and Target
X = df[["sqft", "bedrooms", "bathrooms", "age_years", "garage", "location_score"]]
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# MLflow Tracking Setup
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("mlops-house-prediction")

MODEL_NAME = "house-price-predictor"

with mlflow.start_run():
    n_estimators = 150
    max_depth = 8

    model = RandomForestRegressor(
        n_estimators=n_estimators, max_depth=max_depth, random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    # Logging Parameters & Metrics
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("data_source", f"s3://{BUCKET}/{KEY}")
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2_score", r2)

    # FIXED: Added skops_trusted_types to prevent UntrustedTypesFoundException
    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name=MODEL_NAME,
        skops_trusted_types=["sklearn.tree._tree.Tree"],
    )

    # Set '@champion' Alias
    client = MlflowClient()
    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias="champion",
        version=model_info.registered_model_version,
    )

    print(f"\n MAE: {mae:.2f} | RMSE: {rmse:.2f} | R2: {r2:.4f}")
    print(
        f"SUCCESS: Model registered as '{MODEL_NAME}' version {model_info.registered_model_version} with alias '@champion'!"
    )