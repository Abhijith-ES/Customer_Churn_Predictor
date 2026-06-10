import pandas as pd
import joblib
from pathlib import Path


base_path = Path(__file__).resolve().parent

preprocessor_path = base_path.parent / "models" / "preprocessor.pkl"
lr_model_path = base_path.parent / "models" / "logistic_regression.pkl"
rf_model_path = base_path.parent / "models" / "random_forest.pkl"

def load_preprocessor():
    if not preprocessor_path.exists():
        raise FileNotFoundError(
            f"Preprocessor Not Found at {preprocessor_path}"
        )
    return joblib.load(preprocessor_path)

def load_logistic_regression_model():
    if not lr_model_path.exists():
        raise FileNotFoundError(
            f"Logistic Regression Model Not Found at {lr_model_path}"
        )
    return joblib.load(lr_model_path)

def load_random_forest_model():
    if not rf_model_path.exists():
        raise FileNotFoundError(
            f"Random Forest Model Not Found at {rf_model_path}"
        )
    return joblib.load(rf_model_path)

