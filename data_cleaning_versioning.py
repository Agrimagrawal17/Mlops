import pandas as pd
import boto3
from datetime import date

# --------------------------------------------------
# 1. LOAD RAW CSV
# --------------------------------------------------

path = r"C:\Users\HP\Desktop\Mlops_house_predication_raw_data.csv"

data = pd.read_csv(path)
df = pd.DataFrame(data)

print("---------- Before Cleaning --------------")
print(df.isnull().sum())
print(f"Shape Before: {df.shape}")

# --------------------------------------------------
# 2. CLEAN DATA
# --------------------------------------------------

df_clean = df.dropna()

print()
print("---------- After Cleaning --------------")
print(df_clean.isnull().sum())
print(f"Shape After: {df_clean.shape}")

# --------------------------------------------------
# 3. SAVE CLEANED CSV LOCALLY
# --------------------------------------------------

clean_path = r"C:\Users\HP\Desktop\Mlops_house_predication_clean_v1.csv"

df_clean.to_csv(clean_path, index=False)

print()
print(f"Cleaned file saved at: {clean_path}")

# --------------------------------------------------
# 4. CONNECT TO AWS S3
# --------------------------------------------------

s3 = boto3.client("s3")

BUCKET = "agrim-my-bucket-2026"

# --------------------------------------------------
# 5. UPLOAD CLEANED DATA TO S3
# --------------------------------------------------

def upload_processed_data(local_path):

    key = f"processed/{date.today()}/Mlops_house_predication_clean_v1.csv"

    s3.upload_file(local_path, BUCKET, key)

    print()
    print("---------- S3 Upload Successful ----------")
    print(f"Bucket : {BUCKET}")
    print(f"S3 Key : {key}")
    print(f"S3 Path: s3://{BUCKET}/{key}")


# --------------------------------------------------
# 6. RUN UPLOAD
# --------------------------------------------------

upload_processed_data(clean_path)