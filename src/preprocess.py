import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def clean_data(df: pd.DataFrame):
    df.drop(columns = "customerID", inplace=True)
    df['TotalCharges'].replace(r'^\s*$', np.nan, regex=True, inplace=True)
    df['TotalCharges'] = df['TotalCharges'].astype('float64')
    df['TotalCharges'].fillna(0, inplace=True)
    
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


