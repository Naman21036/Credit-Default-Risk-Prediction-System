Credit Default Risk Prediction System

A production-style machine learning system that predicts the probability of credit card default using customer profile, billing history, and repayment behavior.
The project covers the entire ML lifecycle from data ingestion and model training to deployment via an interactive dashboard.

🔍 Problem Statement

Financial institutions need to assess the risk of credit default before extending or managing credit lines.
This project predicts the probability of default in the next billing cycle, enabling risk-aware decision making.

🚀 Solution Overview

Binary classification problem

Predicts default probability instead of just class labels

Uses ensemble machine learning models

Deployed as a dashboard-style web application for business users

🧠 Machine Learning Pipeline
1. Data Ingestion

Loads cleaned dataset

Stratified train-test split

Stores raw, train, and test data as artifacts

2. Data Transformation

Numerical features scaled using StandardScaler

Categorical features encoded using OneHotEncoder

Preprocessing pipeline saved for inference consistency

3. Model Training

Models evaluated:

Logistic Regression

K-Nearest Neighbors

Decision Tree

Random Forest

AdaBoost

XGBoost

CatBoost

Hyperparameter tuning with GridSearchCV

Best model selected using ROC-AUC

Trained model serialized and stored

4. Inference

Uses the saved preprocessing pipeline and trained model

Returns probability of default

Supports real-time predictions via web UI

📊 Best Model Performance

Model: XGBoost Classifier

Metric: ROC-AUC

Score: ~0.78

Probability-based predictions allow flexible risk thresholding.

🖥️ Web Application
Dashboard Features

Two-panel dashboard layout

Dropdown-based categorical inputs to prevent user errors

Visual probability gauge (low → high risk)

Clear risk classification (High / Low)

Fintech-style UI suitable for non-technical stakeholders

Tech Stack

Flask

Jinja2

HTML & CSS
