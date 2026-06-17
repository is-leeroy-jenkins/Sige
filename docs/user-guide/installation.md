# Installation

## 🧭 Purpose

This page explains how to install Schedule-X dependencies and run the Streamlit application locally.

## 🧱 Requirements

Schedule-X requires Python and the packages listed in `requirements.txt`.

Primary runtime packages include:

| Package | Role |
|---|---|
| `streamlit` | Web application framework. |
| `pandas` | DataFrame loading, transformation, and tabular analysis. |
| `numpy` | Numeric arrays and computations. |
| `scipy` | Statistical tests. |
| `scikit-learn` | PCA, clustering, anomaly detection, LDA, and preprocessing. |
| `matplotlib` | Plotting fallback. |
| `seaborn` | Statistical plotting fallback. |
| `plotly` | Interactive charting when available. |
| `openpyxl` | Excel workbook loading. |
| `umap-learn` | Optional UMAP projection support. |

## 🪟 Windows PowerShell Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

## 🐧 macOS or Linux Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

## 🌐 Open the Application

After Streamlit starts, open the local URL printed in the terminal. The default local address is usually:

```text
http://localhost:8501
```

## 🛠️ Common Startup Issues

| Issue | Cause | Fix |
|---|---|---|
| `streamlit` is not recognized | Virtual environment is not active or dependencies were not installed. | Activate `.venv` and rerun `pip install -r requirements.txt`. |
| Excel import fails | `openpyxl` is missing or workbook is invalid. | Confirm `openpyxl` is installed and the workbook opens normally. |
| Port already in use | Another Streamlit process is running. | Run `streamlit run app.py --server.port 8502`. |
| PowerShell activation blocked | Script execution policy prevents activation. | Use `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` if permitted. |

## ✅ Installation Check

The installation is ready when the Streamlit page opens, the Schedule-X sidebar appears, and the application prompts for an Excel workbook or fallback path.
