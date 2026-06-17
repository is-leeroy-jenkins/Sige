# Descriptive Statistics

## 🧭 Purpose

The Descriptive Statistics section summarizes selected numeric columns and visualizes their distributions.

## 🔢 Column Selection

Select one or more numeric columns from the multi-select control. Schedule-X defaults to the first available numeric columns.

## 📊 Expanded Statistics

| Metric | Meaning |
|---|---|
| `count` | Number of non-null values. |
| `missing` | Number of missing values. |
| `mean` | Arithmetic average. |
| `median` | Middle value after sorting. |
| `std` | Sample standard deviation. |
| `mad` | Median absolute deviation. |
| `cv` | Coefficient of variation. |
| `skewness` | Direction and intensity of asymmetry. |
| `kurtosis` | Tail weight relative to a normal-like distribution. |
| `min` | Minimum value. |
| `1%`, `25%`, `75%`, `99%` | Quantile values. |
| `max` | Maximum value. |
| `IQR` | Interquartile range. |
| `outliers_iqr` | Count of values outside the IQR outlier rule. |
| `outliers_z` | Count of values with absolute z-score greater than 3. |
| `ks_stat` | Kolmogorov-Smirnov statistic against a normalized distribution. |
| `ks_pval` | Kolmogorov-Smirnov p-value. |

## 📈 Distribution Plots

Each selected numeric column receives a distribution view. Plotly is used when available. Matplotlib and Seaborn are used as fallbacks.

## ✅ Recommended Interpretation

| Pattern | Interpretation |
|---|---|
| High missing count | The column may not be appropriate for models requiring complete rows. |
| Large difference between mean and median | The distribution may be skewed or affected by extreme values. |
| High coefficient of variation | Scale varies substantially relative to the mean. |
| High IQR outlier count | Review records for legitimate budget spikes, coding issues, or anomalies. |
| Strong skewness | Avoid assuming symmetric distribution behavior. |
| Low KS p-value | Normality assumptions may not be appropriate. |

## ⚠️ Analytical Caution

Budget data often contains structural zeros, policy-driven spikes, rescissions, transfers, and account-specific patterns. Treat outlier metrics as triage indicators, not automatic data-quality defects.
