from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def run_cross_validation(X, y, model):
    numerical_features = ["tenure", "MonthlyCharges", "TotalCharges"]

    multiclass_features = [
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaymentMethod"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                numerical_features
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                multiclass_features
            )
        ],
        remainder="passthrough"
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    stratified_k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = cross_validate(
        estimator=pipeline,
        X=X,
        y=y,
        cv=stratified_k_fold,
        scoring=[
            "accuracy",
            "precision",
            "recall",
            "f1"
        ]
    )

    return results


