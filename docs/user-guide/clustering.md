# Clustering

## 🧭 Purpose

The Clustering section groups records using selected numeric features and a selected clustering algorithm.

## 🧩 Supported Algorithms

| Algorithm | Description | Key Parameters |
|---|---|---|
| kMeans | Partitions records into a fixed number of clusters. | `k` clusters |
| DBSCAN | Identifies dense regions and noise points. | `eps`, `min_samples` |
| Agglomerative | Builds hierarchical clusters bottom-up. | `n_clusters` |

## 🔢 Feature Selection

Select numeric columns that represent the analytical basis for grouping. Schedule-X standardizes selected numeric values before clustering.

## 📊 Outputs

| Output | Description |
|---|---|
| Cluster membership table | Selected features with assigned cluster labels. |
| Cluster counts | Number of records assigned to each cluster. |
| PCA cluster visualization | Two-dimensional PCA projection colored by cluster. |
| Silhouette score | Separation score when valid for the selected labels. |

## ✅ Recommended Use

1. Review missing values and outliers before clustering.
2. Use PCA or correlation review to understand feature redundancy.
3. Start with k-means for a simple baseline.
4. Use DBSCAN when density and noise records matter.
5. Use agglomerative clustering when hierarchical grouping is analytically useful.
6. Compare cluster counts and record examples before interpreting clusters.

## ⚠️ Interpretation Notes

Clusters are algorithmic groupings, not authoritative classifications. Cluster labels should be reviewed against account names, program categories, budget lines, and source-system context.
