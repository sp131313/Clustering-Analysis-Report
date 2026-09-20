"""
Stage 2 - Data Preparation
Recreates the preprocessing pipeline and train/test split.

Input:
    ../Dataset_ATS_v2.csv

Outputs:
    ../Data_Preparation/preprocessed_dataset.csv
    ../Data_Preparation/training_set.csv
    ../Data_Preparation/testing_set.csv
    ../Data_Preparation/preprocessing_pipeline.joblib
"""

from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "Dataset_ATS_v2.csv"
OUT = ROOT / "Data_Preparation"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(INPUT)

target = "Churn"
X = df.drop(columns=[target]).copy()
y = df[target].map({"Yes": 1, "No": 0})

numeric = X.select_dtypes(include=["number"]).columns.tolist()
categorical = X.select_dtypes(exclude=["number"]).columns.tolist()

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipe, numeric),
    ("cat", categorical_pipe, categorical),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

X_all_t = preprocessor.fit_transform(X)
X_train_t = preprocessor.transform(X_train)
X_test_t = preprocessor.transform(X_test)

feature_names = preprocessor.get_feature_names_out()

def make_df(arr, target_values):
    out = pd.DataFrame(arr, columns=feature_names)
    out[target] = target_values.to_numpy()
    return out

make_df(X_all_t, y).to_csv(OUT/"preprocessed_dataset.csv", index=False)
make_df(X_train_t, y_train).to_csv(OUT/"training_set.csv", index=False)
make_df(X_test_t, y_test).to_csv(OUT/"testing_set.csv", index=False)
joblib.dump(preprocessor, OUT/"preprocessing_pipeline.joblib")

print(f"Input shape: {df.shape}")
print(f"Training shape: {X_train_t.shape}")
print(f"Testing shape: {X_test_t.shape}")
print("Data preparation complete.")
