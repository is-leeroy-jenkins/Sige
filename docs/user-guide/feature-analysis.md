# Feature Analysis

## 🧭 Purpose

The Feature Analysis section examines relationships among numeric features and explores whether categorical labels separate records in reduced-dimensional space.

## 🔗 Correlation Matrix

Schedule-X computes correlations across selected numeric columns.

| Method | Use |
|---|---|
| Pearson | Linear relationships. |
| Spearman | Monotonic relationships based on ranks. |
| Kendall | Rank association with stronger resistance to small-sample instability. |

## 📉 PCA Projection

Principal Component Analysis projects selected numeric features into lower-dimensional components. The application displays explained variance ratios and a two-dimensional projection when `PC1` and `PC2` are available.

| Output | Meaning |
|---|---|
| Explained variance ratio | Share of variance captured by each component. |
| `PC1` | First principal component. |
| `PC2` | Second principal component. |
| Optional color target | Categorical field used to color projected records. |

## 🎯 Linear Discriminant Analysis

LDA projects numeric features to maximize separation between classes in a selected categorical target.

| Requirement | Description |
|---|---|
| Numeric features | Selected feature columns must be coercible to numeric. |
| Target column | A categorical field with at least two classes. |
| Complete rows | Rows with missing selected fields are dropped. |

## 🧩 Optional k-Means

The Feature Analysis section can run k-means on selected numeric features and summarize cluster counts and feature statistics by cluster.

## ✅ Recommended Use

1. Select numeric columns with common scale or meaningful transformed values.
2. Review correlations before interpreting PCA.
3. Use categorical coloring to inspect known classifications.
4. Use LDA only when the target classes are meaningful.
5. Treat cluster summaries as exploratory segmentation.

## ⚠️ Interpretation Notes

PCA components are mathematical directions of variance, not direct budget categories. LDA separation can be strong when labels encode large structural differences. Correlation does not imply causality or policy effect.
