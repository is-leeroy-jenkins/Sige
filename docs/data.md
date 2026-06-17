# Data

## 🧭 Purpose

This page documents the input data expectations for Schedule-X and the checks analysts should perform before running statistical, modeling, clustering, or anomaly-detection workflows.

## 📥 Workbook Input

Schedule-X expects an Excel workbook. The preferred worksheet name is `Data`. When the `Data` sheet is unavailable, the loader attempts to read the first worksheet.

The application checks input sources in this order:

1. Uploaded workbook from the Streamlit sidebar.
2. Optional fallback path entered in the sidebar.
3. `stores/excel/CombinedSchedules.xlsx` relative to the repository.
4. `stores/excel/CombinedSchedules.xlsx` relative to the current working directory.
5. `SCHEDULEX_COMBINED_PATH` environment variable when configured.

## 🧱 Schedule-X Style Fields

| Field Type          | Examples                                                    | Notes                                                                            |
|---------------------|-------------------------------------------------------------|----------------------------------------------------------------------------------|
| Year values         | `PY`, `CY`, `BY`, `OY-1`, `OY-2`, `OY-9`                    | Used for descriptive statistics, projections, clustering, and anomaly detection. |
| Account identifiers | `AgencyCode`, `BureauCode`, `MainAccount`, `TreasurySymbol` | Often categorical even when represented as numbers.                              |
| Account descriptors | `AccountName`, `LineName`, `Classification`                 | Useful for grouping, filtering, and interpretation.                              |
| Budget categories   | `Subfunction`, `BudgetEnforcementCategory`                  | Useful for ANOVA, chi-square, coloring, and grouping.                            |

## ✅ Pre-Analysis Checklist

| Check                                                        | Why It Matters                                                               |
|--------------------------------------------------------------|------------------------------------------------------------------------------|
| Confirm workbook loads successfully.                         | Empty data stops the application workflow.                                   |
| Confirm the correct worksheet is loaded.                     | A wrong sheet can produce misleading summaries.                              |
| Confirm numeric year columns are numeric or coercible.       | Statistical functions coerce values to numeric and drop invalid rows.        |
| Confirm categorical columns are not unintentionally numeric. | Account codes may need to be interpreted as categories.                      |
| Review missing values.                                       | Missing values reduce usable records for projections, tests, and clustering. |
| Review outliers.                                             | Extreme budget values can dominate variance-based methods.                   |

## 🧮 Numeric Columns

Numeric columns are detected through pandas numeric data types, excluding booleans. These columns drive most statistical, dimensionality-reduction, clustering, and anomaly-detection workflows.

## 🔖 Categorical Columns

Categorical columns are detected through object and category data types. These columns support ANOVA grouping, chi-square testing, categorical coloring, and LDA target selection.

## ⚠️ Common Data Issues

| Issue                              | Symptom                                                 | Correction                                                                  |
|------------------------------------|---------------------------------------------------------|-----------------------------------------------------------------------------|
| Missing `Data` sheet               | Loader falls back to first sheet or returns empty data. | Rename the worksheet or confirm the first sheet contains the intended data. |
| Numeric values stored as text      | Columns may not appear in numeric selections.           | Clean the workbook or convert fields before upload.                         |
| Account codes converted to numbers | Leading zeros may be lost.                              | Store account identifiers as text in the source workbook.                   |
| Large workbook                     | Slow upload or memory pressure.                         | Pre-filter the workbook or run locally with sufficient memory.              |
| Sparse out-year columns            | Reduced rows for PCA, clustering, or anomaly detection. | Select columns with adequate coverage.                                      |

## 📤 Exported Data

Schedule-X can export raw data and anomaly records as CSV files. Exported data reflects the current loaded workbook and selected analytical workflow.
