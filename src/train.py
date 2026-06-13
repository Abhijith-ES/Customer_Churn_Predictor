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
    # Best hyperparameter configuration from tuning results
    model = LogisticRegression(max_iter=1000, C=0.1, class_weight='balanced', penalty='l1', solver='liblinear')
    model.fit(X_train, y_train)

    model_path = Path("models/logistic_regression.pkl")
    save_model(model, model_path)

    print("Model Training Successful and the trained model is saved on 'models' folder")
    return model

def train_random_forest(X_train, y_train):
    # Best hyperparameter configuration from tuning results
    model = RandomForestClassifier(random_state=42, 
                    n_estimators=300, class_weight="balanced", 
                    max_depth=10, max_features="sqrt", 
                    min_samples_leaf=2, min_samples_split=10)
    
    model.fit(X_train, y_train)

    model_path = Path("models/random_forest.pkl")
    save_model(model, model_path)

    print("Model Training Successful and the trained model is saved on 'models' folder")
    return model