# 🏠 MLOps House Price Prediction

An end-to-end **Machine Learning + MLOps pipeline** for House Price Prediction, designed to demonstrate how a machine learning project can be developed, tracked, containerized, automated, and deployed using modern MLOps practices.

The project covers the complete lifecycle — from **raw data ingestion and preprocessing to experiment tracking, model management, API serving, Docker containerization, CI/CD, and cloud deployment**.

---

## 🚀 Project Overview

Traditional machine learning projects often stop after training a model.

This project goes beyond model training and implements an **MLOps workflow** where data processing, model training, experiment tracking, application serving, and deployment are organized into an automated and reproducible pipeline.

### Workflow

```text
Raw Dataset
     │
     ▼
Data Ingestion
     │
     ▼
Data Cleaning & Preprocessing
     │
     ▼
Processed Dataset
     │
     ▼
AWS S3
     │
     ▼
Model Training
     │
     ▼
MLflow Experiment Tracking
     │
     ├── Parameters
     ├── Metrics
     ├── Artifacts
     └── Model
     │
     ▼
Best Model
     │
     ▼
FastAPI
     │
     ▼
Docker Container
     │
     ▼
GitHub
     │
     ▼
GitHub Actions CI/CD
     │
     ▼
AWS Deployment
```

---

## 🎯 Objectives

The main objectives of this project are:

* Build a complete machine learning pipeline.
* Clean and preprocess raw datasets programmatically.
* Store processed data in **AWS S3**.
* Train a machine learning model on processed data.
* Track experiments using **MLflow**.
* Log model parameters, metrics, and artifacts.
* Serve predictions through **FastAPI**.
* Containerize the application using **Docker**.
* Automate the workflow using **GitHub Actions**.
* Deploy the application to **AWS**.
* Create a reproducible and maintainable ML workflow.

---

## 🛠️ Tech Stack

| Technology     | Purpose                                |
| -------------- | -------------------------------------- |
| Python         | Core programming language              |
| Pandas         | Data processing and cleaning           |
| NumPy          | Numerical operations                   |
| Scikit-learn   | Machine Learning                       |
| MLflow         | Experiment tracking & model management |
| FastAPI        | Prediction API                         |
| Uvicorn        | API server                             |
| AWS S3         | Cloud data storage                     |
| Docker         | Containerization                       |
| Git            | Version control                        |
| GitHub         | Source code management                 |
| GitHub Actions | CI/CD automation                       |
| AWS EC2        | Cloud deployment                       |

---

## 📂 Project Structure

```text
Mlops/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── data/
│   │   └── data_cleaning.py
│   │
│   ├── model/
│   │   ├── train.py
│   │   └── predict.py
│   │
│   └── api/
│       └── main.py
│
├── notebooks/
│
├── tests/
│
├── .github/
│   └── workflows/
│       └── deploy.yml
│
├── Dockerfile
├── requirements.txt
├── README.md
└── .gitignore
```

> The exact folder/file structure may evolve as the project is expanded.

---

# 📊 1. Data Processing

The project starts with a raw house-price dataset.

The raw dataset is loaded using Pandas and inspected for:

* Missing values
* Duplicate records
* Data types
* Dataset shape
* Invalid values
* Feature distributions

Example:

```python
import pandas as pd

data = pd.read_csv("raw_data.csv")

print(data.shape)
print(data.isnull().sum())
```

### Data Cleaning

Rows containing missing values are removed during the initial preprocessing stage.

```python
df_clean = data.dropna()
```

The pipeline then compares the dataset before and after cleaning.

Example output:

```text
---------- Before Cleaning ----------

Missing Values:
...

Shape Before: (...., ....)

---------- After Cleaning ----------

Missing Values:
...

Shape After: (...., ....)
```

---

# ☁️ 2. AWS S3 Data Storage

After preprocessing, the cleaned dataset can be uploaded to **Amazon S3**.

The project uses `boto3` to communicate with AWS S3.

Example workflow:

```text
Local Raw Data
      │
      ▼
Pandas Cleaning
      │
      ▼
Processed CSV
      │
      ▼
AWS S3
```

Processed files can be organized using a structure such as:

```text
processed/
    2026-09-28/
        cleaned_house_data.csv
```

This provides a centralized cloud location for processed datasets.

---

# 🤖 3. Machine Learning

The processed dataset is used for training the House Price Prediction model.

The machine learning workflow includes:

```text
Processed Dataset
       │
       ▼
Feature Selection
       │
       ▼
Train / Test Split
       │
       ▼
Model Training
       │
       ▼
Prediction
       │
       ▼
Evaluation
```

Typical regression evaluation metrics include:

* MAE
* MSE
* RMSE
* R² Score

Example:

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
```

---

# 📈 4. MLflow Experiment Tracking

MLflow is used to track machine learning experiments.

Instead of manually recording every experiment, MLflow stores important information such as:

### Parameters

```text
model type
hyperparameters
training configuration
```

### Metrics

```text
MAE
MSE
RMSE
R²
```

### Artifacts

```text
trained model
plots
processed files
other experiment outputs
```

The basic workflow is:

```text
Model Training
      │
      ▼
MLflow Experiment
      │
      ├── Parameters
      ├── Metrics
      ├── Artifacts
      └── Model
```

MLflow allows different experiments to be compared and helps maintain a history of model training runs.

---

# 🔬 5. MLflow Tracking Server

The project also uses an MLflow tracking server for experiment management.

The MLflow UI can be accessed locally through:

```text
http://127.0.0.1:5000
```

The server provides a centralized interface for viewing:

* Experiments
* Runs
* Parameters
* Metrics
* Artifacts
* Models

### Important

If port `5000` is already being used by another process, MLflow may return:

```text
[WinError 10048]
Only one usage of each socket address...
```

In that case, another process is already listening on port `5000`, and MLflow should be started on another available port or the existing process should be stopped.

---

# ⚡ 6. FastAPI Prediction API

After model training, the trained model can be exposed through a REST API using FastAPI.

The API allows a client/application to send house-related features and receive a predicted price.

Example:

```text
Client
  │
  │ POST /predict
  ▼
FastAPI
  │
  ▼
Trained ML Model
  │
  ▼
Predicted House Price
```

Example response:

```json
{
    "predicted_price": 4250000
}
```

FastAPI also provides automatic interactive API documentation.

```text
/docs
```

---

# 🐳 7. Docker Containerization

The application is containerized using Docker.

Instead of depending on the local machine environment, Docker packages the application and its dependencies into a reproducible container.

### Docker Workflow

```text
Application
    │
    ├── Python
    ├── Dependencies
    ├── ML Model
    └── FastAPI
          │
          ▼
       Docker Image
          │
          ▼
      Docker Container
```

Example Docker command:

```bash
docker build -t house-price-mlops .
```

Run the container:

```bash
docker run -p 8000:8000 house-price-mlops
```

---

# 🔄 8. CI/CD with GitHub Actions

The project uses GitHub Actions to automate the development and deployment workflow.

Whenever changes are pushed to the repository, the CI/CD pipeline can automatically:

```text
Developer Push
      │
      ▼
GitHub Repository
      │
      ▼
GitHub Actions
      │
      ├── Checkout Code
      ├── Install Dependencies
      ├── Run Tests
      ├── Build Docker Image
      └── Deploy
```

This reduces manual deployment steps and provides a repeatable deployment process.

---

# ☁️ 9. AWS Deployment

The application can be deployed to an AWS EC2 instance.

High-level deployment workflow:

```text
Local Development
       │
       ▼
GitHub
       │
       ▼
GitHub Actions
       │
       ▼
Docker Image
       │
       ▼
AWS EC2
       │
       ▼
Docker Container
       │
       ▼
FastAPI Application
```

The EC2 server acts as the production environment for the containerized application.

---

# 🔐 10. Environment Variables

Sensitive information such as AWS credentials should **never be hard-coded** inside the source code.

Use environment variables instead.

Example:

```text
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
AWS_DEFAULT_REGION
S3_BUCKET_NAME
MLFLOW_TRACKING_URI
```

A `.env` file can be used locally.

Example:

```env
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=ap-south-1
S3_BUCKET_NAME=your_bucket
MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

The `.env` file should be included in `.gitignore`.

---

# 🧪 11. Reproducibility

One of the main goals of this project is reproducibility.

The project keeps track of:

* Source code
* Dataset processing
* Dependencies
* Experiments
* Model metrics
* Model artifacts
* Docker environment
* Deployment configuration

This makes it easier to reproduce experiments and deploy the same application across different environments.

---

# 📌 End-to-End Architecture

```text
                    ┌──────────────────┐
                    │   Raw Dataset    │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Data Cleaning    │
                    │    Pandas        │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │   AWS S3         │
                    │ Processed Data   │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ ML Training      │
                    │ Scikit-learn    │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │     MLflow       │
                    │ Experiments      │
                    │ Metrics          │
                    │ Artifacts        │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │  Trained Model   │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │     FastAPI      │
                    │ Prediction API   │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │     Docker       │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ GitHub Actions   │
                    │     CI/CD        │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │    AWS EC2       │
                    │   Deployment     │
                    └──────────────────┘
```

---

# 💻 Local Setup

## 1. Clone Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Mlops
```

## 2. Create Virtual Environment

```bash
python -m venv agrim
```

Activate it on Windows:

```powershell
.\agrim\Scripts\activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run MLflow

Start the MLflow tracking server:

```bash
mlflow server --host 127.0.0.1 --port 5000
```

Open:

```text
http://127.0.0.1:5000
```

If port `5000` is occupied, use another port:

```bash
mlflow server --host 127.0.0.1 --port 5001
```

---

# ▶️ Run FastAPI

Start the API using Uvicorn:

```bash
uvicorn src.api.main:app --reload
```

Then open:

```text
http://127.0.0.1:8000/docs
```

---

# 🐳 Run with Docker

Build the image:

```bash
docker build -t house-price-mlops .
```

Run:

```bash
docker run -p 8000:8000 house-price-mlops
```

API:

```text
http://localhost:8000
```

Swagger:

```text
http://localhost:8000/docs
```

---

# 📦 Requirements

The project uses Python packages such as:

```text
pandas
numpy
scikit-learn
mlflow
fastapi
uvicorn
boto3
python-dotenv
```

The complete dependency list is maintained in:

```text
requirements.txt
```

---

# 🔮 Future Improvements

Planned improvements include:

* Automated model retraining
* Model registry
* Data versioning
* Automated model validation
* Model monitoring
* Data drift detection
* Model drift detection
* Automated testing
* Production MLflow deployment
* Cloud-based artifact storage
* Advanced CI/CD pipeline
* Automated rollback
* Production monitoring
* Logging and observability
* Infrastructure automation

---

# 📚 What This Project Demonstrates

This project demonstrates practical understanding of:

### Machine Learning

* Regression
* Feature preprocessing
* Model training
* Model evaluation
* Prediction

### Data Engineering

* Data ingestion
* Data cleaning
* Processed data management
* Cloud storage

### MLOps

* Experiment tracking
* Model management
* Reproducibility
* Model serving
* Containerization
* CI/CD
* Cloud deployment

### Cloud

* AWS S3
* AWS EC2
* Cloud-based application deployment

---

# 👨‍💻 Author

**Agrim Agrawal**

B.Tech — Computer Science & Engineering

Focused on:

```text
Data Science
Machine Learning
MLOps
Generative AI
Cloud & Deployment
```

---

## ⭐ Project

If you find this project useful, consider giving the repository a ⭐ on GitHub.

---

## 📄 License

This project is intended for educational and portfolio purposes.
