from src.data_loader import load_data
from src.preprocess import clean_data, prepare_data
from src.hyperparameter_tuning import tune_logistic_regression


data_path = "data/customer_churn_data.csv"
churn_data = load_data(data_path)

cleaned_churn_data = clean_data(churn_data)
X, y = prepare_data(cleaned_churn_data)

grid_search = tune_logistic_regression(X, y)

print(f'''
BEST PARAMETER : {grid_search.best_params_}
BEST SCORE : {grid_search.best_score_}
      ''')