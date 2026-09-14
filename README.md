
Olist Delivery Prediction

Project Overview
This project predicts whether an Olist order will be delivered late.
The project demonstrates an end-to-end machine learning workflow including:
•	Data preparation
•	Feature engineering
•	Exploratory data analysis
•	Data preprocessing
•	Model training
•	Model evaluation
•	Model and configuration management

Project Structure
olist-mlops/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── olist.db
│
├── models/
│
├── notebooks/
│
├── src/
│
├── config.yaml
├── requirements.txt
└── README.md

Requirements
•	Python 3.14
•	Virtual environment
•	Required Python packages listed in requirements.txt

Environment Setup
Create and activate the virtual environment:
python -m venv .venv
.venv\Scripts\Activate.ps1

Install the required packages:
pip install -r requirements.txt

Configuration
Project settings are stored in:
config.yaml
The configuration contains paths for the database, raw data, processed data, models, and notebooks, as well as model settings.

Current model configuration includes:
Model: Random Forest
Trees: 300
Target: is_late_label

Running the Project
The notebooks should be executed in order.
1.	Read and join the source data
2.	Create the late-delivery label
3.	Split the dataset
4.	Perform exploratory data analysis
5.	Perform feature engineering and preprocessing
6.	Train, tune, and evaluate the model

The Jupyter notebook kernel should use the project's virtual environment:
olist-mlops\.venv\Scripts\python.exe
