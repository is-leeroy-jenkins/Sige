# Data Loading

## 🧭 Purpose

This page explains how Schedule-X loads Excel data and how to verify that the correct workbook is active.

## 📥 Upload a Workbook

Use the sidebar upload control to select an Excel workbook. The preferred workbook contains a worksheet named `Data`.

```text
Upload CombinedSchedules.xlsx (sheet "Data")
```

## 📂 Use a Fallback Path

When no workbook is uploaded, Schedule-X can attempt to load a workbook from a local path entered in the sidebar.

Common fallback location:

```text
stores/excel/CombinedSchedules.xlsx
```

## 🔁 Loader Order

Schedule-X attempts to load data in this sequence:

1. Uploaded workbook.
2. User-entered fallback path.
3. Repository-relative `stores/excel/CombinedSchedules.xlsx`.
4. Current-working-directory `stores/excel/CombinedSchedules.xlsx`.
5. `SCHEDULEX_COMBINED_PATH` environment variable.

## ✅ Successful Load Indicators

A successful load produces:

| Indicator | Meaning |
|---|---|
| Overview section renders. | A non-empty DataFrame was loaded. |
| Sample rows appear. | The workbook contains readable tabular data. |
| Feature summary appears. | Data types and missing counts were computed. |
| Numeric snapshot appears. | Numeric columns were detected. |

## ⚠️ Empty Data Handling

If no workbook is loaded or no readable data is found, Schedule-X displays a warning and stops the workflow. Correct the workbook, sheet name, fallback path, or environment variable and reload the app.

## 🧪 Data Verification

After loading data, review the Overview page before running other workflows. Confirm that row samples, account fields, year fields, and classification fields are present and correctly typed.
