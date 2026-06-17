# Dimensionality Reduction

## 🧭 Purpose

The Dimensionality Reduction section projects selected numeric columns into lower-dimensional spaces so analysts can inspect structure, separation, and potential groupings.

## 🧮 Supported Methods

| Method | Description | Best Use |
|---|---|---|
| PCA | Linear projection maximizing variance. | Baseline structure and explained variance. |
| TruncatedSVD | Matrix factorization projection. | Sparse or high-dimensional numeric matrices. |
| FactorAnalysis | Latent-factor model. | Exploring shared variance structure. |
| IncrementalPCA | Batch-compatible PCA variant. | Larger data where incremental fitting is useful. |
| t-SNE | Nonlinear local-neighborhood projection. | Visual exploration of local grouping. |
| UMAP | Nonlinear manifold projection when installed. | Visual exploration of local and global manifold structure. |

## 🔢 Feature Selection

Select numeric columns with enough complete observations. The application coerces selected values to numeric and drops rows with missing values before projection.

## 📊 Projection Output

When two or more components are available, Schedule-X displays a two-dimensional scatter plot using the first two projection columns.

## ✅ Recommended Use

1. Start with PCA to establish a baseline.
2. Compare PCA with SVD or FactorAnalysis for structure checks.
3. Use t-SNE or UMAP for exploratory visualization, not statistical proof.
4. Keep selected features consistent when comparing methods.
5. Document selected columns and method parameters for reproducibility.

## ⚠️ Limitations

Dimensionality-reduction methods can produce different views of the same data. Nonlinear projections can show apparent clusters even when the underlying separation is weak. Use projections as exploratory evidence and validate with tabular summaries.
