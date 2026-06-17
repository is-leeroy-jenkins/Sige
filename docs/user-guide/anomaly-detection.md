# Anomaly Detection

## 🧭 Purpose

The Anomaly Detection section identifies records that differ from the selected numeric feature population according to the selected detector.

## 🚨 Supported Detectors

| Detector | Description | Key Parameters |
|---|---|---|
| Isolation Forest | Isolates unusual records through randomized tree partitions. | `n_estimators` |
| One-Class SVM | Learns a boundary around typical records. | `nu` |
| Local Outlier Factor | Compares local density against neighboring records. | `n_neighbors`, `contamination` |
| Elliptic Envelope | Fits a robust covariance estimate under an elliptical assumption. | `contamination` |

## 🔢 Feature Selection

Select numeric columns that define the anomaly context. Different feature selections can produce different anomaly records. Schedule-X standardizes selected numeric values before detection.

## 📊 Outputs

| Output | Meaning |
|---|---|
| `anomaly = -1` | Record flagged as an outlier. |
| `anomaly = 1` | Record treated as an inlier. |
| Anomaly count | Number of records flagged by the selected detector. |
| Anomaly table | Flagged records displayed for review. |
| Anomaly CSV | Downloadable file containing flagged records. |

## ✅ Recommended Review Pattern

1. Select budget-year and account-relevant numeric features.
2. Run Isolation Forest as a baseline.
3. Compare results with LOF or Elliptic Envelope when appropriate.
4. Review flagged records with account descriptors and classification fields.
5. Export anomaly records for follow-up analysis.
6. Record the selected detector and parameters used.

## ⚠️ Analytical Caution

Anomaly detection identifies unusual records relative to the selected data and parameters. A flagged record may represent a valid policy change, emergency supplemental, transfer, rescission, account restructuring, or timing difference. Analyst review is required.
