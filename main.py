import pandas as pd
from src.data_loader import load_data, explore_data


# Initialising Customer Data
data_path = "data/customer_churn_data.csv"
customer_data = load_data(data_path)

# Exploring the Dataset
print(customer_data["TotalCharges"].str.strip().eq("").sum())
print(customer_data["TotalCharges"].dtype)
print(len(customer_data))
print(customer_data["Churn"].value_counts())
