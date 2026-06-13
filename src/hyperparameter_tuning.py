from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# Features to be encoded
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

# Preprocessor to be used in the Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        (
            'num',
            StandardScaler(),
            numerical_features
        ),
        (
            'cat',
            OneHotEncoder(handle_unknown='ignore'),
            multiclass_features
        )
    ],
    remainder='passthrough'
)

def tune_logistic_regression(X, y):
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=1000))
    ])

    param_grid = {
        "model__C" : [0.01, 0.1, 1, 10, 100],
        "model__class_weight" : [ None, "balanced"],
        "model__penalty" : ['l1','l2'],
        "model__solver" : ['liblinear']
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring='f1',
        n_jobs=-1
    )

    grid_search.fit(X, y)
    return grid_search

def tune_random_forest(X, y):
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(random_state=42))
    ])

    param_grid = {
        "model__n_estimators" : [100,200,300],
        "model__max_depth" : [5,10,15,None],
        "model__class_weight" : [None, "balanced"],
        "model__min_samples_split" : [2,5,10],
        "model__min_samples_leaf" : [1,2,4],
        "model__max_features" : ['sqrt','log2']
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring='f1',
        n_jobs=-1,
        cv=cv
    )

    grid_search.fit(X,y)
    return grid_search