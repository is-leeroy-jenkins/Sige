##### Sige
![](https://github.com/is-leeroy-jenkins/Sige/blob/master/resources/assets/img/git/SigeProject.png)
– Federal Outlay Projections with Python


Sige is a specialized machine learning and time series forecasting tool built for federal analysts, budget officers, and data scientists working with  Treasury Account Symbols (TAS)  and  federal outlays. It combines classical regression, decision trees, gradient boosting, support vector machines, and time series modeling to project **monthly or quarterly outlays** at the Treasury Account level.

## 🎯 Colab   

<a href="https://colab.research.google.com/github/is-leeroy-jenkins/Sige/blob/master/outlays.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

![](https://github.com/is-leeroy-jenkins/Sige/blob/master/resources/assets/Sige-nb.gif) 


## 🚀 Streamlit  

[![Streamlit App](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://sige-py.streamlit.app/)

![](https://github.com/is-leeroy-jenkins/Sige/blob/master/resources/assets/img/sige-demo.gif)

## 🧱 Databricks
[![Databricks Notebook](https://img.shields.io/badge/Databricks%20Repo-Sake--Py-FF3621?logo=databricks&logoColor=white)](https://dbc-a0c21f80-7bb3.cloud.databricks.com/editor/notebooks/1460524320197777?o=7474645703081351)
- A data engineering, analytics, and artificial intelligence collaborative workspace
- Codebase
  
## 🚀 Features

#### 📈 Full-Funnel Forecasting Pipeline
- Load and structure TAS-level outlay data from SF-133-style or custom tabular reports
- Extract features from **monthly or quarterly outlay data**
- Train and compare multiple models across time slices


## 🧠 Machine Learning Regressors
| Model                         | Type            | Library                                |
|------------------------------|------------------|----------------------------------------|
| Linear Regression            | Parametric       | `sklearn.linear_model.LinearRegression` |
| Decision Tree Regressor      | Tree-Based       | `sklearn.tree.DecisionTreeRegressor`    |
| Random Forest Regressor      | Ensemble         | `sklearn.ensemble.RandomForestRegressor` |
| Gradient Boosting Regressor  | Ensemble         | `sklearn.ensemble.GradientBoostingRegressor` |
| XGBoost Regressor            | Boosting (Extra) | `xgboost.XGBRegressor` *(optional)*     |
| SVR (Support Vector Regressor) | Kernel-Based   | `sklearn.svm.SVR`                       |


## 🕰️ Time Series Models
| Model     | Library                          |
|-----------|----------------------------------|
| ARIMA     | `statsmodels.tsa.arima.model.ARIMA` |




## 🧮 Visual Analysis
- Forecast vs. Actual Plots
- Residual error visualization
- Multi-model comparison tables
- Rolling and lagged error summaries

## 🏛️ Government-Focused Design
- Built for **federal financial analysts**
- Structured around **Treasury Account Symbols (TAS)**
- Extendable to SF-133 and MAX Schedule A/B formats



## 🔬 Features

#### 📈 End-to-End Forecasting Pipeline
- Ingests TAS outlay data from SF-133-style reports
- Extracts and transforms lagged, rolling, and time-based features
- Evaluates and compares multiple ML and time series models
- Supports target normalization and log transformations

#### 🔁 Feature Engineering
- Rolling average features (e.g., 3-month trailing)
- Lag features (e.g., t-1, t-2)
- Month and quarter encodings
- Log1p transforms to stabilize variance in skewed data

#### 📊 Evaluation Metrics
- **R²**, **MAE**, **MSE**, **RMSE**
- **MAPE** and percent error (planned)
- Comparison plots across models

#### 🧮 Visual Diagnostics
- Forecast vs. Actual line plots
- Scatter plots of predicted vs. actual
- Residual error distributions
- Multi-model score bar charts
- Time-based faceted error panels

## 🏛️ Government-Tailored Design
- Treasury Account Symbol (TAS)-centric processing
- Integrates seamlessly with SF-133, GTAS, MAX A/B formats
- Ideal for fiscal year and policy-driven forecasting



## 🧪 Output Artifacts

| Output                          | Description                                 |
|---------------------------------|---------------------------------------------|
| Model Metrics CSV               | Tabular file of all evaluation scores       |
| Plots (PNG/HTML)                | All forecast visualizations                 |
| Residual Error Charts           | Distribution & temporal residual patterns   |
| Forecast Sheets (Excel planned) | Side-by-side predicted vs. actual tables    |
| Model Timings                   | Training + inference time profiling         |



## 📁 How to Use

1. Click the **Open In Colab** badge above.
2. Upload your CSV or mount Google Drive.
3. Set `DATA_PATH` near the top of the notebook.
4. **Runtime → Run all**.


#### 🔧 Setup

```bash
git clone https://github.com/your-username/sige.git
cd sige
pip install -r requirements.txt
jupyter notebook outlays.ipynb
```

#### Option B — Local (conda or venv)

```
bash
# 1) Create environment
conda create -n sige python=3.11 -y
conda activate sige

# 2) Install dependencies
pip install -U pip wheel setuptools
pip install pandas numpy scipy matplotlib seaborn scikit-learn jupyter

# 3) Launch Jupyter
jupyter notebook
```

Open `ipynb/sige.ipynb` and run cells top-to-bottom.



- This application is delivered as a Streamlit web app (`app.py`) that allows users to upload historical federal outlay data (Excel format), explore trends, and generate multi-model forecasts.

### Prerequisites

Before you begin, ensure you have the following installed:

* **Python 3.10 or newer**
* **Git**
* **pip** (comes with Python)

You can verify your Python version with:

```
python --version
```

### 1️⃣ Clone the Repository

Clone the Sige repository to your local machine:

```
git clone https://github.com/<your-org-or-username>/sige.git
cd sige
```

---

### 2️⃣ Create a Virtual Environment (Recommended)

Using a virtual environment is strongly recommended to avoid dependency conflicts.

**Windows (PowerShell):**

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```
python3 -m venv .venv
source .venv/bin/activate
```

---

### 3️⃣ Install Dependencies

Install the required Python packages:

```
pip install -r requirements.txt
```

If a `requirements.txt` file is not present, the core dependencies are:

* streamlit
* pandas
* numpy
* matplotlib
* scikit-learn
* statsmodels
* openpyxl

---

### 4️⃣ Launch the Streamlit App

Run the application with:

```
streamlit run app.py
```

After a few seconds, Streamlit will open the app automatically in your default web browser (typically at `http://localhost:8501`).

---

### 5️⃣ Using the Application

1. Upload an Excel file containing historical outlays with fiscal year columns (e.g., `FY2012`, `FY2013`, …).
2. Specify the worksheet name (default: `Data`).
3. Configure the training window and forecast years in the sidebar.
4. Review:

   * Aggregated outlays by fiscal year
   * Regression model performance
   * Forecasts from regression, ARIMA, and Holt-Winters models
5. Export results as CSV files from the **Export** tab.

---

### Notes

* The app runs entirely locally; no data is transmitted externally.
* Large Excel files may take a few seconds to process on first upload.
* Refreshing the browser will reset the app state.




## 📊 Descriptive Statistics

| Statistic         | Description                             | Use in Budget Analysis                                               |
|------------------|-----------------------------------------|----------------------------------------------------------------------|
| **Mean**         | Average value                           | Avg. Outlays, Obligations, etc., across accounts                |
| **Median**       | Middle value                            | Robust central tendency in skewed financial data                    |
| **Mode**         | Most frequent value                     | Identify common MainAccountCodes or Availability categories     |
| **Standard Deviation** | Spread around the mean                | Indicates variability in execution rates or balances                |
| **Variance**     | Square of standard deviation            | Used in statistical tests and model diagnostics                     |
| **Range**        | Difference between max and min          | Measures total spread of financial metrics                          |
| **Interquartile Range (IQR)** | Spread of middle 50% of data           | Identifies budget outliers and extreme accounts                     |
| **Skewness**     | Asymmetry of distribution               | Skewed obligations suggest few accounts dominate totals             |
| **Kurtosis**     | "Peakedness" of distribution            | High values indicate outlier-prone financial data                   |





## 🔍 Inferrential Statistics


| Metric           | Description                                            | Use in Budget Analysis                                               |
|-------------------------|--------------------------------------------------------|----------------------------------------------------------------------|
| **Pearson Correlation** | Linear relationship between variables                  | E.g., TotalResources vs. Obligations                                 |
| **Spearman Correlation**| Monotonic (rank-based) relationship                    | More robust to non-linear trends in financial execution              |
| **t-test**              | Compare means between 2 groups                         | Discretionary vs. Mandatory accounts' execution rates                |
| **ANOVA**               | Compare means across multiple groups                   | Obligations across availability periods or account types             |
| **Chi-square Test**     | Categorical independence                               | Are Main Account Codes related to availability or a specific agency? |
| **Confidence Intervals**| Estimate range of a population mean                    | Upper and lower bound expected obligations or recoveries             |
| **Regression Coefficients (p-values)** | Test variable significance                             | Are Recoveries a significant predictor of UnobligatedBalance?        |
| **F-statistic (overall regression)**   | Test whole model fit                                   | Determines the combined influence of all predictors                  |
| **Z-score / Outlier Tests** | Deviation from standard mean                           | Identify abnormal balances or lapse rates                            |
| **Boxplots**            | Visual outlier detection                               | Discover obligation anomalies within agencies                        |



## 📦 Dependencies

| Package          | Description                                                      | Link                                                  |
|------------------|------------------------------------------------------------------|-------------------------------------------------------|
| numpy            | Numerical computing library                                      | [numpy.org](https://numpy.org/)                      |
| pandas           | Data manipulation and DataFrames                                 | [pandas.pydata.org](https://pandas.pydata.org/)      |
| matplotlib       | Plotting and visualization                                       | [matplotlib.org](https://matplotlib.org/)            |
| seaborn          | Statistical data visualization                                   | [seaborn.pydata.org](https://seaborn.pydata.org/)    |
| scikit-learn     | ML modeling and metrics                                          | [scikit-learn.org](https://scikit-learn.org/stable/) |
| xgboost          | Gradient boosting framework (optional)                          | [xgboost.readthedocs.io](https://xgboost.readthedocs.io/) |
| torch            | PyTorch deep learning library                                    | [pytorch.org](https://pytorch.org/)                  |
| tensorflow       | End-to-end ML platform                                           | [tensorflow.org](https://www.tensorflow.org/)        |
| openai           | OpenAI’s Python API client                                       | [openai-python](https://github.com/openai/openai-python) |
| requests         | HTTP requests for API and web access                             | [requests.readthedocs.io](https://requests.readthedocs.io/) |
| PySimpleGUI      | GUI framework for desktop apps                                   | [pysimplegui.readthedocs.io](https://pysimplegui.readthedocs.io/) |
| typing           | Type hinting standard library                                    | [typing Docs](https://docs.python.org/3/library/typing.html) |
| pyodbc           | ODBC database connector                                          | [pyodbc GitHub](https://github.com/mkleehammer/pyodbc) |
| fitz             | PDF document parser via PyMuPDF                                  | [pymupdf](https://pymupdf.readthedocs.io/)           |
| pillow           | Image processing library                                         | [python-pillow.org](https://python-pillow.org/)       |
| openpyxl         | Excel file processing                                            | [openpyxl Docs](https://openpyxl.readthedocs.io/)     |
| soundfile        | Read/write sound file formats                                    | [pysoundfile](https://pysoundfile.readthedocs.io/)    |
| sounddevice      | Audio I/O interface                                              | [sounddevice Docs](https://python-sounddevice.readthedocs.io/) |
| loguru           | Structured, elegant logging                                      | [loguru GitHub](https://github.com/Delgan/loguru)     |
| statsmodels      | Statistical tests and regression diagnostics                     | [statsmodels.org](https://www.statsmodels.org/)       |
| dotenv           | Load environment variables from `.env`                          | [python-dotenv GitHub](https://github.com/theskumar/python-dotenv) |
| python-dotenv    | Same as above (modern usage)                                     | [python-dotenv](https://saurabh-kumar.com/python-dotenv/) |



## 🧪 How to Run

```bash
git clone https://github.com/your-username/balance-projector.git
cd balance-projector
pip install -r requirements.txt
jupyter notebook balances.ipynb
```



## 📁 Customize Dataset

Replace dataset ingestion cell with:

```python
import pandas as pd
df = pd.read_csv("your_dataset.csv")
X = df.drop("target_column", axis=1)
y = df["target_column"]
```



## 📊 Outputs

- R², MAE, MSE for each model
- Bar plots of performance scores
- Visual predicted vs. actual scatter charts
- Residual error analysis



> **Disclaimer**: This is for analytical exploration and research purposes.  
> It is **not** an official government product; validate against authoritative sources before use.

## 📝 License

Sige is published under the [MIT General Public License v3](https://github.com/is-leeroy-jenkins/Sige/blob/master/LICENSE.txt).


