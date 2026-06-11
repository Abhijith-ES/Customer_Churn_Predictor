from src.data_loader import load_data
from src.preprocess import prepare_data, clean_data
from src.cross_validate import run_cross_validation

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


data_path = "data/customer_churn_data.csv"
churn_data = load_data(data_path)

cleaned_churn_data = clean_data(churn_data)

X, y = prepare_data(cleaned_churn_data)
model_lr = LogisticRegression(max_iter=1000)
model_rf = RandomForestClassifier(random_state=42)

results_lr = run_cross_validation(X, y, model_lr)
results_rf = run_cross_validation(X, y, model_rf)

print(f'''
Logistic Regression Model's Recall  :  {results_lr["test_recall"].mean():.2f}
Random Forest Model's Recall : {results_rf["test_recall"].mean():.2f}

Logistic Regression Model's F1 Score  :  {results_lr["test_f1"].mean():.2f}
Random Forest Model's F1 Score  :  {results_rf["test_f1"].mean():.2f}

Logistic Regression Model's Accuracy  :  {results_lr["test_accuracy"].mean():.2f}
Random Forest Model's Accuracy  :  {results_rf["test_accuracy"].mean():.2f}

Logistic Regression Model's Precision  :  {results_lr["test_precision"].mean():.2f}
Random Forest Model's Precision  :  {results_rf["test_precision"].mean():.2f}
''')