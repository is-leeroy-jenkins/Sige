# User Guide

## 🧭 Purpose

The user guide provides task-oriented instructions for running Schedule-X, loading data, reviewing data quality, executing analytical workflows, and exporting results.

## ✅ Recommended Sequence

| Step | Page | Purpose |
|---|---|---|
| 1 | Installation | Create the Python environment and start Streamlit. |
| 2 | Data Loading | Load the Excel workbook and confirm the `Data` sheet. |
| 3 | Data Overview | Inspect row samples, data types, missing values, and numeric summaries. |
| 4 | Descriptive Statistics | Understand distribution, spread, skewness, and outlier structure. |
| 5 | Inferential Statistics | Run statistical comparisons and categorical association checks. |
| 6 | Feature Analysis | Review correlations, PCA structure, LDA separation, and optional k-means summaries. |
| 7 | Dimensionality Reduction | Compare projection methods across selected features. |
| 8 | Clustering | Assign and inspect cluster labels. |
| 9 | Anomaly Detection | Identify records requiring follow-up review. |
| 10 | Export | Download raw data or anomaly records. |

## 🧱 Sidebar Sections

Schedule-X is organized through a sidebar radio selector. Each section renders a focused analytical workflow using the currently loaded DataFrame.

| Section | Role |
|---|---|
| Overview | Data inspection and baseline profiling. |
| Descriptive Statistics | Expanded summary statistics and distributions. |
| Inferential Statistics | Statistical testing across numeric and categorical selections. |
| Feature Analysis | Correlation, PCA, LDA, and optional clustering review. |
| Dimensionality Reduction | Alternative low-dimensional projections. |
| Clustering | Group detection using selected numeric features. |
| Anomaly Detection | Outlier detection using selected numeric features. |
| Export | Raw data export. |

## ⚠️ Before Running Analyses

Confirm that the loaded data represents the intended workbook, worksheet, fiscal period, account population, and data cut. Schedule-X does not replace authoritative budget systems or audit reconciliation procedures.
