"""
Stage 2 - K-Means Clustering
Runs the elbow analysis, trains the final K-Means model, profiles clusters,
and creates a 2-D PCA visualisation.

Input:
    ../Dataset_ATS_v2.csv

Outputs:
    ../Clustering_Analysis/elbow_method.png
    ../Clustering_Analysis/kmeans_model.joblib
    ../Clustering_Analysis/clustered_customers.csv
    ../Clustering_Analysis/cluster_profiles.csv
    ../Clustering_Analysis/cluster_visualisation.png
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "Dataset_ATS_v2.csv"
OUT = ROOT / "Clustering_Analysis"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(INPUT)

# Churn is deliberately excluded from cluster creation.
X = df.drop(columns=["Churn"]).copy()

numeric = X.select_dtypes(include=["number"]).columns.tolist()
categorical = X.select_dtypes(exclude=["number"]).columns.tolist()

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]), numeric),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]), categorical),
])

X_t = preprocessor.fit_transform(X)

# Elbow method.
ks = range(2, 9)
inertias = []
for k in ks:
    model = KMeans(n_clusters=k, random_state=42, n_init=20)
    model.fit(X_t)
    inertias.append(model.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(list(ks), inertias, marker="o")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Within-cluster sum of squares")
plt.title("Elbow Method for K-Means")
plt.xticks(list(ks))
plt.tight_layout()
plt.savefig(OUT/"elbow_method.png", dpi=180)
plt.close()

# Final model.
kmeans = KMeans(n_clusters=4, random_state=42, n_init=20)
labels = kmeans.fit_predict(X_t)

joblib.dump({"preprocessor": preprocessor, "model": kmeans},
            OUT/"kmeans_model.joblib")

clustered = df.copy()
clustered["Cluster"] = labels
clustered.to_csv(OUT/"clustered_customers.csv", index=False)

profiles = (
    clustered.groupby("Cluster")
    .agg(
        Customers=("Cluster", "size"),
        Average_Tenure=("tenure", "mean"),
        Average_Monthly_Charges=("MonthlyCharges", "mean"),
        Senior_Citizen_Rate=("SeniorCitizen", "mean"),
        Churn_Rate=("Churn", lambda s: (s == "Yes").mean()),
    )
    .reset_index()
)

# Descriptive labels based on the observed profiles.
label_map = {
    0: "Newer Higher-Cost Customers",
    1: "Lower-Cost Mid-Tenure Customers",
    2: "Long-Tenure Higher-Value Customers",
    3: "Senior Higher-Risk Customers",
}
profiles["Segment_Label"] = profiles["Cluster"].map(label_map)
profiles.to_csv(OUT/"cluster_profiles.csv", index=False)

# PCA is for visualisation only.
pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(X_t)

plt.figure(figsize=(8, 6))
for c in sorted(np.unique(labels)):
    mask = labels == c
    plt.scatter(coords[mask, 0], coords[mask, 1], s=10, alpha=0.45, label=f"Cluster {c}")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Customer Clusters Visualised with PCA")
plt.legend()
plt.tight_layout()
plt.savefig(OUT/"cluster_visualisation.png", dpi=180)
plt.close()

print("Selected k: 4")
print(profiles.to_string(index=False))
print("Clustering analysis complete.")
