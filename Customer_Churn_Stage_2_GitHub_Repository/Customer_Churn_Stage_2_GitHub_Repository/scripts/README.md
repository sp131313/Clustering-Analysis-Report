# Stage 2 Scripts

- `01_data_preparation.py` prepares the raw dataset, handles missing values, one-hot encodes categorical variables, scales numeric features, and creates a stratified 80/20 train/test split.
- `02_clustering_analysis.py` runs the elbow method, trains K-Means with four clusters, saves the model, creates cluster profiles, and generates a PCA visualisation.

Run from the repository root:

```bash
pip install -r requirements.txt
python scripts/01_data_preparation.py
python scripts/02_clustering_analysis.py
```
