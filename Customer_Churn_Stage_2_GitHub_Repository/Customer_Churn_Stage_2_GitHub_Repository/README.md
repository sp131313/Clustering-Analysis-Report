# Customer Churn Analysis - Stage 2

This repository contains the team's Stage 2 submission for the Customer Churn Analysis project.

## Stage 2 Deliverables

### Data_Preparation
Contains:
- Preprocessed dataset
- Training set
- Testing set
- Scaling and encoding documentation
- Reusable preprocessing pipeline

### Clustering_Analysis
Contains:
- Elbow-method analysis
- Trained K-Means model
- Clustered customer data
- Cluster profiles and labels
- Cluster visualisation
- Clustering analysis report

### scripts
Reproducible Python scripts for the data preparation and clustering workflow.

## Main Stage 2 Results

- 7,043 customer records
- No missing values in the supplied fields
- 302 exact duplicate rows identified and documented
- 80/20 stratified train/test split
- One-hot encoding for categorical variables
- StandardScaler for numerical variables
- Four K-Means customer segments selected using the elbow method
- PCA visualisation used to present the resulting segments

## Important Data-Quality Note

The source dataset does not contain a customer identifier. Therefore, exact duplicate rows were identified but not automatically removed, because identical attribute combinations could belong to separate customers. This decision is documented in the project materials.

## Reproducibility

Install the packages listed in `requirements.txt`, then run:

```bash
python scripts/01_data_preparation.py
python scripts/02_clustering_analysis.py
```

## Project Links

Jira board:
https://spsinghsethi2000-1786265308078.atlassian.net/jira/software/projects/KAN/boards/2

Confluence project charter:
https://chhavid07.atlassian.net/wiki/spaces/~7120209d1fe97795d0472bb376bfd46e38a859/pages/622593/Customer+Churn+Analysis+Project+Charter

Confluence repository:
https://chhavid07.atlassian.net/wiki/x/BwAK
