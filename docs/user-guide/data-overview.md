# Data Overview

## 🧭 Purpose

The Overview section confirms that the workbook loaded correctly and provides the first data-quality review before deeper analysis.

## 📊 Sample Rows

The application displays the first 300 rows of the loaded DataFrame. Use the sample to confirm that the correct workbook, worksheet, fiscal period, and account population were loaded.

## 🧱 Feature Summary

The feature summary table includes:

| Column | Meaning |
|---|---|
| `dtype` | pandas data type for the field. |
| `n_unique` | Count of unique non-null values. |
| `n_missing` | Count of missing values. |

## 🔢 Numeric Snapshot

The numeric snapshot uses pandas descriptive statistics with additional percentiles:

| Metric | Meaning |
|---|---|
| `count` | Number of non-null numeric values. |
| `mean` | Average value. |
| `std` | Standard deviation. |
| `min` | Minimum value. |
| `1%`, `5%`, `10%` | Lower-tail percentiles. |
| `25%`, `50%`, `75%` | Quartile and median values. |
| `90%`, `95%`, `99%` | Upper-tail percentiles. |
| `max` | Maximum value. |

## ✅ Review Checklist

| Check | Action |
|---|---|
| Dataset is correct. | Confirm sample rows match the intended workbook. |
| Numeric columns are present. | Confirm budget-year fields appear in the numeric snapshot. |
| Missing values are acceptable. | Review `n_missing` before modeling or tests. |
| Categorical fields are usable. | Confirm agency, bureau, account, line, or classification fields are present. |
| Outliers are plausible. | Inspect high percentiles and maximum values. |

## ⚠️ When to Stop

Stop and correct the workbook before continuing if the wrong sheet is loaded, numeric year columns are missing, account identifiers are malformed, or missing values make selected analyses unreliable.
