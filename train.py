import pandas as pd
import boto3
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
import numpy as np

# Fetch cleaned data from S3
s3 = boto3.client('s3')
BUCKET = 'agrim-my-bucket-2026'
KEY = "processed/2026-09-25/Mlops_house_predication_clean_v1.csv"

def fetch_data():
    obj = s3.get_object(Bucket=BUCKET, Key=KEY)
    # Fix 1: obj['Body'].read().decode('utf-8')
    # Fix 2: pd.read_csv (with underscore)
    df = pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))
    return df

df = fetch_data()
print(f"Fetched shape: {df.shape}")

#features / Targets
X = df[{'sqft','bedrooms','bathrooms','age_years',"garage","location_score"}]
y= df['price']

X_train,X_test,y_test,y_train = train_test_split(x,y,test_size = 0.2, random_state = 42)
mlflow.set_experiment("mlflow_house_prediction")
