import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


PREPROCESSOR_PATH = Path("models/preprocessor.pkl")

def clean_data(df: pd.DataFrame):
    df.drop(columns = "customerID", inplace=True)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    
    binary_cols = [
        'Churn',       # Target Variable (Churn) : Yes - > 1, No -> 0
        'Partner',
        'Dependents',
        'PhoneService',
        'PaperlessBilling'
    ]

    for col in binary_cols:
        df[col] = df[col].map({'Yes' : 1, 'No' : 0})
    
    df['gender'] = df['gender'].map({'Female' : 0, 'Male' : 1})

    return df

def prepare_data(df: pd.DataFrame):
    X = df.drop(columns = "Churn")
    y = df["Churn"]

    return X,y

def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def save_preprocessor(preprocessor):

    PREPROCESSOR_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(preprocessor, PREPROCESSOR_PATH)

    print("Preprocessor saved successfully.")

def preprocess_train_data(X_train: pd.DataFrame):

    numerical_features = ["tenure", "MonthlyCharges", "TotalCharges"]

    multiclass_features = [
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaymentMethod"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                numerical_features
            ),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                multiclass_features
            )
        ],
        remainder="passthrough"
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    save_preprocessor(preprocessor)
 
    return X_train_processed, preprocessor


def preprocess_test_data(X_test: pd.DataFrame, preprocessor):

    X_test_processed = preprocessor.transform(X_test)
    return X_test_processed


