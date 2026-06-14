from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer

from src.schemas import CustomerRequestData, PredictionResponseModel
import pandas as pd


def clean_input(df: pd.DataFrame):
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    
    binary_cols = [
        'Partner',
        'Dependents',
        'PhoneService',
        'PaperlessBilling'
    ]

    for col in binary_cols:
        df[col] = df[col].map({'Yes' : 1, 'No' : 0})
    
    df['gender'] = df['gender'].map({'Female' : 0, 'Male' : 1})

    return df

def predict_churn(customer_data : CustomerRequestData, model:RandomForestClassifier, preprocessor:ColumnTransformer ):
    customer_dict = customer_data.model_dump()
    customer_inp = pd.DataFrame([customer_dict])
    customer_cleaned_inp = clean_input(customer_inp)

    X_processed = preprocessor.transform(customer_cleaned_inp)

    prediction = model.predict(X_processed)[0]
    prediction = 'Churn' if prediction==1 else 'Not Churn'

    prediction_prob = model.predict_proba(X_processed)[0][1]

    return PredictionResponseModel(prediction=prediction, probability=prediction_prob)