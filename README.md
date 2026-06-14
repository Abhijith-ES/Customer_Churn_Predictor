# 📉 Customer Churn Prediction API

A Machine Learning powered Customer Churn Prediction System built using Python, Scikit Learn, and FastAPI.

This project predicts whether a telecom customer is likely to churn based on customer demographics, subscription details, service usage patterns, billing information, and contract history.

The trained machine learning model is exposed through a FastAPI REST API and can be tested using Postman, Swagger UI, or any HTTP client.

---

## ⚠️ Important Note

This project was completely handwritten and developed manually for learning purposes.

No AI-generated code, code generators, templates, or copy-paste project implementations were used while building the application. Every module, preprocessing pipeline, model training workflow, evaluation process, hyperparameter tuning strategy, validation rule, API endpoint, and project structure was implemented manually to strengthen practical skills in:

- Python
- Machine Learning
- Data Preprocessing
- Model Evaluation
- FastAPI Development
- Software Engineering

AI was used only as a mentor for learning concepts, reviewing implementations, debugging issues, and discussing best practices.

---

# 🚀 Project Overview

The objective of this project is to build an end-to-end Machine Learning Inference Service capable of:

- Loading customer churn data
- Cleaning and preprocessing data
- Training multiple machine learning models
- Evaluating model performance
- Performing Cross Validation
- Performing Hyperparameter Tuning
- Saving trained artifacts
- Exposing predictions through FastAPI
- Validating requests using Pydantic
- Returning churn predictions with confidence scores

This project demonstrates the complete lifecycle of a production-style machine learning system.

---

# 🏗️ Architecture Diagram

```text
                Customer Dataset
                        │
                        ▼
                 Data Cleaning
                        │
                        ▼
               Feature Engineering
                        │
                        ▼
              Train/Test Split
                        │
                        ▼
               ColumnTransformer
          (Scaling + One Hot Encoding)
                        │
                        ▼
                 Model Training
        ┌─────────────────────────┐
        │ Logistic Regression     │
        │ Random Forest           │
        └─────────────────────────┘
                        │
                        ▼
               Model Evaluation
                        │
                        ▼
               Cross Validation
                        │
                        ▼
             Hyperparameter Tuning
                        │
                        ▼
          Save Model + Preprocessor
                        │
                        ▼
                 FastAPI Startup
                        │
                        ▼
         Load Model + Preprocessor
                        │
                        ▼
                  /predict API
                        │
                        ▼
              Pydantic Validation
                        │
                        ▼
                Churn Prediction
                        │
                        ▼
                 JSON Response
```

---

# 📁 Project Structure

```text
customer_churn_predictor/
│
├── data/
│   └── customer_churn_data.csv
│
├── models/
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   └── preprocessor.pkl
│
├── notebooks/
│   └── exploration.ipynb
│
├── src/
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   ├── cross_validate.py
│   ├── hyperparameter_tuning.py
│   ├── predict.py
│   ├── schemas.py
│   └── utils.py
│
├── api.py
├── main.py
├── validation.py
├── grid_search.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

# 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Scikit Learn
- Joblib
- FastAPI
- Pydantic
- Uvicorn
- Postman

---

# 📊 Machine Learning Workflow

## 1. Data Loading

Customer churn dataset is loaded using Pandas.

---

## 2. Data Cleaning

Performed:

- Missing value handling
- Data type corrections
- TotalCharges conversion
- Binary feature encoding

---

## 3. Feature Selection

Target Variable:

```python
Churn
```

Features:

```python
gender
SeniorCitizen
Partner
Dependents
tenure
PhoneService
MultipleLines
InternetService
OnlineSecurity
OnlineBackup
DeviceProtection
TechSupport
StreamingTV
StreamingMovies
Contract
PaperlessBilling
PaymentMethod
MonthlyCharges
TotalCharges
```

---

## 4. Data Preprocessing

### Numerical Features

```python
tenure
MonthlyCharges
TotalCharges
```

Applied:

```python
StandardScaler
```

### Categorical Features

```python
MultipleLines
InternetService
OnlineSecurity
OnlineBackup
DeviceProtection
TechSupport
StreamingTV
StreamingMovies
Contract
PaymentMethod
```

Applied:

```python
OneHotEncoder
```

### Pipeline

Implemented using:

```python
ColumnTransformer
Pipeline
```

---

## 5. Train Test Split

Dataset Split:

```text
80% Training
20% Testing
```

Result:

```text
Training Samples : 5634
Testing Samples  : 1409
```

---

## 6. Model Training

### Logistic Regression

```python
LogisticRegression()
```

### Random Forest

```python
RandomForestClassifier()
```

---

## 7. Model Evaluation

Metrics Used:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Classification Report

---

## 8. Cross Validation

Implemented:

```python
StratifiedKFold
cross_validate()
```

Purpose:

- Reduce variance from a single train-test split
- Evaluate model robustness
- Detect overfitting

Metrics:

- Accuracy
- Precision
- Recall
- F1 Score

---

## 9. Hyperparameter Tuning

Implemented:

```python
GridSearchCV
```

### Logistic Regression Parameters

```python
C
class_weight
penalty
solver
```

Best Parameters:

```python
{
    'model__C': 0.1,
    'model__class_weight': 'balanced',
    'model__penalty': 'l1',
    'model__solver': 'liblinear'
}
```

---

### Random Forest Parameters

```python
n_estimators
max_depth
max_features
min_samples_leaf
min_samples_split
class_weight
```

Best Parameters:

```python
{
    'model__class_weight': 'balanced',
    'model__max_depth': 10,
    'model__max_features': 'sqrt',
    'model__min_samples_leaf': 2,
    'model__min_samples_split': 10,
    'model__n_estimators': 300
}
```

---

## 10. Model Selection

Business Goal:

```text
Catch as many churn customers as possible.
```

Primary Metric:

```text
Recall
```

Chosen Model:

```text
Random Forest Classifier
```

Reason:

- High Recall
- Good F1 Score
- Better balance between precision and recall

---

## 11. Model Persistence

Saved Artifacts:

```text
random_forest.pkl
logistic_regression.pkl
preprocessor.pkl
```

Implemented using:

```python
joblib
```

---

# 🌐 API Endpoints

## Home Endpoint

```http
GET /
```

Response:

```json
{
    "message": "Customer Churn Predictor Is Running..."
}
```

---

## Health Check Endpoint

```http
GET /health
```

Response:

```json
{
    "status": "Healthy"
}
```

---

## Predict Churn Endpoint

```http
POST /predict
```

Request:

```json
{
  "gender": "Male",
  "SeniorCitizen": 0,
  "Partner": "No",
  "Dependents": "No",
  "tenure": 2,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 95.5,
  "TotalCharges": 191.0
}
```

Response:

```json
{
    "prediction": "Churn",
    "probability": 0.8978
}
```

---

# 🔒 Request Validation

Implemented using:

```python
Pydantic
```

Validation Examples:

- Invalid Contract Type
- Invalid Internet Service
- Invalid Payment Method
- Invalid Tenure
- Missing Required Fields
- Invalid Binary Values

Result:

```http
422 Unprocessable Entity
```

---

# ⚙️ How To Run Locally

## Clone Repository

```bash
git clone https://github.com/Abhijith-ES/Customer_Churn_Predictor.git
```

---

## Move Into Project

```bash
cd customer_churn_predictor
```

---

## Create Virtual Environment

```bash
python -m venv venv
```

---

## Activate Virtual Environment

### Windows

```bash
venv\Scripts\activate
```

### Linux / Mac

```bash
source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Train Models

```bash
python main.py
```

---

## Run API

```bash
uvicorn api:app --reload
```

---

# 📖 API Documentation

Swagger UI

```text
http://127.0.0.1:8000/docs
```

ReDoc

```text
http://127.0.0.1:8000/redoc
```

---

# 🎯 Learning Outcomes

This project helped develop practical experience in:

- Data Cleaning
- Feature Engineering
- One Hot Encoding
- Feature Scaling
- Machine Learning Pipelines
- Logistic Regression
- Random Forest
- Model Evaluation
- Classification Metrics
- Cross Validation
- Hyperparameter Tuning
- Model Persistence
- FastAPI Development
- Pydantic Validation
- REST APIs
- Postman Testing
- Production Style ML Systems

---

# 📌 Future Improvements

- Probability Threshold Tuning
- Precision Recall Tradeoff Analysis
- ROC-AUC Evaluation
- Feature Importance Visualization
- SHAP Explainability
- Dockerization
- CI/CD Pipeline
- Automated Testing
- Model Monitoring
- Cloud Deployment (AWS / Azure / GCP)

---

# 👨‍💻 Author

**E S Abhijith**

Aspiring AI/ML Engineer focused on understanding Machine Learning systems from first principles, building projects manually, and developing strong foundations in AI Engineering, Backend Development, and Production ML Systems.

---