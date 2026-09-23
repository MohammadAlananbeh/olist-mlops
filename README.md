
Olist Delivery Prediction

Project Overview

This project predicts whether an Olist order will be delivered late.

The project implements an end-to-end machine learning and MLOps workflow including:

Data preparation
Data validation
Feature engineering
Exploratory data analysis
Data preprocessing
Model training and evaluation
Model artifact management
Data and model versioning with DVC
Experiment tracking with MLflow
Automated testing
FastAPI model serving
Docker containerization
Prediction logging and monitoring

The prediction target is:

is_late_label

where:

1 = order delivered late
0 = order delivered on time

Project Structure
olist-mlops/
│
├── app/
│
├── artifacts/
│
├── config/
│   └── config.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── olist.db
│   └── olist.db.dvc
│
├── logs/
│   └── app.log
│
├── models/
│   ├── classification_threshold.joblib
│   ├── feature_names.json
│   ├── final_model.joblib
│   ├── model_metadata.json
│   ├── preprocessor.joblib
│   └── *.dvc
│
├── notebooks/
│   ├── Notebook 1 — Read & join the tables.ipynb
│   ├── Notebook 2 — Create the labels.ipynb
│   ├── Notebook 3 — Train   validation   test split.ipynb
│   ├── Notebook 4 — EDA (the detailed one).ipynb
│   ├── Notebook 5 — Feature engineering.ipynb
│   └── Notebook 6 — Train, tune, evaluate.ipynb
│
├── requirements/
│   ├── runtime.txt
│   └── dev.txt
│
├── src/
│   ├── api/
│   │   ├── main.py
│   │   └── schemas.py
│   │
│   ├── features/
│   │   ├── feature_engineering.py
│   │   └── feature_selection.py
│   │
│   ├── monitoring/
│   │   └── prediction_logger.py
│   │
│   ├── prediction/
│   │   └── predictor.py
│   │
│   ├── preprocessing/
│   │   └── preprocessor.py
│   │
│   ├── validation/
│   │   ├── data_validation.py
│   │   └── validator.py
│   │
│   ├── config_loader.py
│   ├── features_2.py
│   ├── preprocessing_2.py
│   └── utils/
│       └── logger.py
│
├── tests/
│   ├── test_api.py
│   ├── test_data.py
│   ├── test_features.py
│   ├── test_model.py
│   ├── test_preprocessing.py
│   └── test_utils.py
│
└── README.md

Requirements
Python 3.14
Git
Docker Desktop
Python virtual environment

Python dependencies are separated into:

requirements/runtime.txt
requirements/dev.txt

The project currently uses SQLite for the application database.

Environment Setup

Create the virtual environment:

python -m venv .venv

Activate it:

.venv\Scripts\Activate.ps1

Install runtime dependencies:

pip install -r requirements/runtime.txt

Install development and testing dependencies:

pip install -r requirements/dev.txt

Verify Python:

python --version

The project currently uses Python 3.14.

Configuration

Project configuration is stored in:

config/config.yaml

The configuration contains settings for:

Database location
Data paths
Target variable
Model parameters
Train/validation/test split configuration

The current target is:

is_late_label

The current model is a Random Forest classifier with:

n_estimators = 300
max_depth = 15
random_state = 42
Data

The project uses the Olist Brazilian e-commerce dataset.

Raw source files are stored under:

data/raw/

The project database is:

data/olist.db

Processed data is stored under:

data/processed/

Data versioning is handled using DVC.

The database has a corresponding DVC file:

data/olist.db.dvc
Machine Learning Workflow

The main workflow is implemented through the following notebooks:

Notebook 1 — Read & join the tables

Loads and joins the original Olist datasets.

Notebook 2 — Create the labels

Creates the late-delivery target:

is_late_label
Notebook 3 — Train / validation / test split

Creates the training, validation, and testing datasets.

Notebook 4 — EDA

Performs exploratory data analysis.

Notebook 5 — Feature engineering

Creates model features and prepares the preprocessing pipeline.

The fitted preprocessing object is saved as:

models/preprocessor.joblib
Notebook 6 — Train, tune, evaluate

Trains and evaluates the final machine-learning model.

The trained model is saved as:

models/final_model.joblib
Inference Pipeline

The API uses the existing model inference pipeline rather than creating a separate preprocessing implementation.

The inference flow is:

Raw Order
    ↓
Feature Engineering
    ↓
Saved Preprocessor
    ↓
Final Random Forest Model
    ↓
Prediction Probability
    ↓
Classification Threshold
    ↓
Final Prediction

The saved inference artifacts include:

models/final_model.joblib
models/preprocessor.joblib
models/classification_threshold.joblib
models/feature_names.json
models/model_metadata.json
FastAPI

The API is implemented in:

src/api/main.py

The service provides model-related endpoints including:

GET /health
GET /model
GET /model/version
POST /predict

The health endpoint can be used to verify that the service is running:

http://localhost:8000/health

FastAPI/OpenAPI documentation is available when the service is running at:

http://localhost:8000/docs
Docker

The API can be packaged and executed using Docker.

Example image build:

docker build -t olist-mlops-api .

Run the API container:

docker run -d --name olist-api -p 8000:8000 olist-mlops-api

Verify the service:

http://localhost:8000/health
Testing

Automated tests are located under:

tests/

Run the test suite with:

pytest

The test suite covers areas including:

API
Data
Feature engineering
Model
Preprocessing
Utilities
MLflow

MLflow is used for experiment tracking and model-training information.

The project uses the MLflow experiment:

Olist Delivery Prediction

MLflow tracking information is stored locally in:

mlflow.db
DVC

DVC is used for versioning data and model artifacts.

Tracked artifacts include the database and model-related files such as:

data/olist.db.dvc
models/final_model.joblib.dvc
models/preprocessor.joblib.dvc
models/classification_threshold.joblib.dvc
models/feature_names.json.dvc
models/model_metadata.json.dvc
Data Validation

Data validation is implemented in:

src/validation/data_validation.py
src/validation/validator.py

The current implementation uses Python/pandas-based validation.

Logging and Monitoring

Application logging is implemented through the project logging utilities.

Application logs are stored under:

logs/app.log

Prediction logging is implemented in:

src/monitoring/prediction_logger.py

Prediction records can include information such as:

Timestamp
Prediction
Prediction probability
Model version

Monitoring functionality is being developed as part of the MLOps workflow.

Development

The recommended development environment is the project's virtual environment:

.venv\Scripts\python.exe

For Jupyter notebooks, make sure the notebook kernel uses the project's .venv environment.

Project Goal

The goal of this project is to demonstrate a reproducible end-to-end MLOps workflow for late-delivery prediction, from raw data and model development through validation, versioning, testing, API serving, containerization, and monitoring.