# Export

## 🧭 Purpose

The Export section provides downloadable CSV outputs from Schedule-X.

## 📤 Raw Data Export

The raw data export serializes the currently loaded DataFrame to CSV.

| Export | Description |
|---|---|
| `schedulex_raw.csv` | CSV version of the loaded workbook data. |

## 🚨 Anomaly Export

The Anomaly Detection section provides a separate export when anomalies are detected.

| Export | Description |
|---|---|
| `schedulex_anomalies.csv` | Records flagged as anomalies by the selected detector. |

## ✅ Recommended Export Practice

1. Confirm the correct workbook is loaded.
2. Confirm the selected workflow and parameters.
3. Export raw data when preserving the full analysis population.
4. Export anomaly data when preserving records for follow-up.
5. Store exported files with notes describing selected columns, detector, parameters, and run date.

## ⚠️ Reproducibility Notes

CSV exports do not automatically include sidebar selections, detector parameters, or analyst notes. Preserve those details separately when results are used for audit support, peer review, or briefing materials.
