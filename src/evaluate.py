from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, confusion_matrix, f1_score
import pandas as pd


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(classification_report(y_test, y_pred))

    return {
        "accuracy" : accuracy,
        "precision" :precision,
        "recall" : recall,
        "confusion_Matrix" : conf_matrix,
        "f1_Score" : f1
        }