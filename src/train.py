from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import pandas as pd
from pathlib import Path
import joblib


def save_model(model, file_path: Path):
    file_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, file_path)
    print("Model Saved Successfully.")

def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    model_path = Path("models/logistic_regression.pkl")
    save_model(model, model_path)

    print("Model Training Successful and the trained model is saved on 'models' folder")
    return model

def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    model_path = Path("models/random_forest.pkl")
    save_model(model, model_path)

    print("Model Training Successful and the trained model is saved on 'models' folder")
    return model