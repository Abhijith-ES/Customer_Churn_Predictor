import pandas as pd
from src.data_loader import load_data
from src.preprocess import prepare_data, clean_data, split_data, preprocess_train_data, preprocess_test_data
from src.train import train_logistic_regression, train_random_forest
from src.evaluate import evaluate_model

# Initialising Customer Data
data_path = "data/customer_churn_data.csv"
customer_data = load_data(data_path)

# Data Cleaning : 
cleaned_df = clean_data(customer_data)

# Features and Targets : 
X, y = prepare_data(cleaned_df)

# Train and Test Split : 
X_train, X_test, y_train, y_test = split_data(X, y)

# Preprocessing Training Data : 
X_train_processed, preprocessor = preprocess_train_data(X_train)

# Preprocessing Test Data :
X_test_processed = preprocess_test_data(X_test, preprocessor)

# Training The Model : 
logistic_regression_model = train_logistic_regression(X_train_processed, y_train)
random_forest_model = train_random_forest(X_train_processed, y_train)

# Model Evaludation :
lr_model_results = evaluate_model(logistic_regression_model, X_test_processed, y_test)
rf_model_results = evaluate_model(random_forest_model, X_test_processed, y_test)

print(f'''
    Logistic Regression Model For Churn Prediction Results :
        {lr_model_results}
    Random Forest Model For Churn Prediction Results : 
        {rf_model_results}
      ''')
