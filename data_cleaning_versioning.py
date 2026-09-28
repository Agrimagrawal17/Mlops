from datetime import date
import boto3
import pandas as pd

# local path
path = r"C:\Users\HP\Downloads\Mlops_house_predication_raw_data.csv"
data = pd.read_csv(path)
df = pd.DataFrame(data)

print("=========== Before Cleaning ============")
print(df.isnull().sum())
print(f"Shape Before : {df.shape}")

print()
df_clean = df.dropna()  # data cleaning
print("=========== After Cleaning ============")
print(df_clean.isnull().sum())
print(f"Shape After : {df_clean.shape}")

# saved cleaned CSV locally
clean_path = r"C:\Users\HP\Downloads\Mlops_house_predication_clean_v2.csv"
df_clean.to_csv(clean_path, index=False)

# upload to s3 as proccessed version
s3 = boto3.client("s3")
BUCKET = "agrim-my-bucket-2026"


def upload_proccessed_data(local_path):
    key = f"processed/{date.today()}/Mlops_house_predication_clean_v1.csv"
    s3.upload_file(local_path, BUCKET, key)
    print(f"\nUploaded to s3://{BUCKET}/{key}")
    return key


upload_proccessed_data(clean_path)