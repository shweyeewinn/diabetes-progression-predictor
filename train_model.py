"""Train a diabetes-progression model whose saved .pkl contains the FULL pipeline:
cleaning (imputation) -> feature engineering -> encoding -> scaling -> model.
Only built-in sklearn steps are used, so the .pkl loads anywhere without extra code.
Your app just passes raw values in this column order:
age, sex, bmi, bp, s1, s2, s3, s4, s5, s6
"""
import pickle
from pathlib import Path
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# ---------- Data (raw, real-world units) ----------
X, y = load_diabetes(return_X_y=True, as_frame=True, scaled=False)
print("Missing values:", int(X.isna().sum().sum()), "| rows:", len(X))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
cv = KFold(5, shuffle=True, random_state=42)

numeric = ["age", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]
categorical = ["sex"]

def make(model):
    preprocess = ColumnTransformer([
        # cleaning -> feature engineering (squares + interactions, e.g. bmi*bp) -> scaling
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("poly", PolynomialFeatures(degree=1, include_bias=False)),
                          ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(drop="if_binary", handle_unknown="ignore"))]), categorical),
    ])
    return Pipeline([("preprocess", preprocess), ("model", model)])

# Cross-validation decides whether the engineered features (degree 2) actually help
searches = {
    "Lasso": GridSearchCV(make(Lasso(max_iter=100000)),
                          {"preprocess__num__poly__degree": [1, 2],
                           "model__alpha": [0.1, 0.5, 1, 2, 5]},
                          cv=cv, scoring="r2", n_jobs=-1),
    "RandomForest": GridSearchCV(make(RandomForestRegressor(n_estimators=300, random_state=42)),
                                 {"model__max_depth": [3, 5, None],
                                  "model__min_samples_leaf": [1, 5, 10]},
                                 cv=cv, scoring="r2", n_jobs=-1),
}

for name, search in searches.items():
    search.fit(X_train, y_train)
    pred = search.predict(X_test)
    print(f"{name:13s} CV R2={search.best_score_:.3f}  Test R2={r2_score(y_test, pred):.3f}  "
          f"MAE={mean_absolute_error(y_test, pred):.1f}  {search.best_params_}")

# Pick the winner by cross-validation score (not test score, to avoid overfitting the test set)
best = max(searches, key=lambda k: searches[k].best_score_)
pipeline = searches[best].best_estimator_
pipeline.fit(X, y)  # refit on all 442 rows for the final model

out = Path(__file__).parent / "diabetes_pipeline.pkl"
with open(out, "wb") as f:
    pickle.dump(pipeline, f)
print(f"Saved {out} ({best})")
