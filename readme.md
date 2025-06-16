##### Sige
![](https://github.com/is-leeroy-jenkins/Sige/blob/master/resources/assets/img/git/SigeProject.png)
– Federal Outlay Projections with Python


Sige is a specialized machine learning and time series forecasting tool built for federal analysts, budget officers, and data scientists working with  Treasury Account Symbols (TAS)  and  federal outlays. It combines classical regression, decision trees, gradient boosting, support vector machines, and time series modeling to project **monthly or quarterly outlays** at the Treasury Account level.

### Outlay Projection Tool
<a href="https://colab.research.google.com/github/is-leeroy-jenkins/Sige/blob/master/outlays.ipynb" target="_parent">
<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

___

### 🚀 Features

#### 📈 Full-Funnel Forecasting Pipeline
- Load and structure TAS-level outlay data from SF-133-style or custom tabular reports
- Extract features from **monthly or quarterly outlay data**
- Train and compare multiple models across time slices


### 🧠 Machine Learning Regressors
| Model                         | Type            | Library                                |
|------------------------------|------------------|----------------------------------------|
| Linear Regression            | Parametric       | `sklearn.linear_model.LinearRegression` |
| Decision Tree Regressor      | Tree-Based       | `sklearn.tree.DecisionTreeRegressor`    |
| Random Forest Regressor      | Ensemble         | `sklearn.ensemble.RandomForestRegressor` |
| Gradient Boosting Regressor  | Ensemble         | `sklearn.ensemble.GradientBoostingRegressor` |
| XGBoost Regressor            | Boosting (Extra) | `xgboost.XGBRegressor` *(optional)*     |
| SVR (Support Vector Regressor) | Kernel-Based   | `sklearn.svm.SVR`                       |

### 🕰️ Time Series Models
| Model     | Library                          |
|-----------|----------------------------------|
| ARIMA     | `statsmodels.tsa.arima.model.ARIMA` |


---

### 🧮 Visual Analysis
- Forecast vs. Actual Plots
- Residual error visualization
- Multi-model comparison tables
- Rolling and lagged error summaries

### 🏛️ Government-Focused Design
- Built for **federal financial analysts**
- Structured around **Treasury Account Symbols (TAS)**
- Extendable to SF-133 and MAX Schedule A/B formats

---

## 🔬 Features

### 📈 End-to-End Forecasting Pipeline
- Ingests TAS outlay data from SF-133-style reports
- Extracts and transforms lagged, rolling, and time-based features
- Evaluates and compares multiple ML and time series models
- Supports target normalization and log transformations

### 🔁 Feature Engineering
- Rolling average features (e.g., 3-month trailing)
- Lag features (e.g., t-1, t-2)
- Month and quarter encodings
- Log1p transforms to stabilize variance in skewed data

### 📊 Evaluation Metrics
- **R²**, **MAE**, **MSE**, **RMSE**
- **MAPE** and percent error (planned)
- Comparison plots across models

### 🧮 Visual Diagnostics
- Forecast vs. Actual line plots
- Scatter plots of predicted vs. actual
- Residual error distributions
- Multi-model score bar charts
- Time-based faceted error panels

### 🏛️ Government-Tailored Design
- Treasury Account Symbol (TAS)-centric processing
- Integrates seamlessly with SF-133, GTAS, MAX A/B formats
- Ideal for fiscal year and policy-driven forecasting

---

## 🧪 Output Artifacts

| Output                          | Description                                 |
|---------------------------------|---------------------------------------------|
| Model Metrics CSV               | Tabular file of all evaluation scores       |
| Plots (PNG/HTML)                | All forecast visualizations                 |
| Residual Error Charts           | Distribution & temporal residual patterns   |
| Forecast Sheets (Excel planned) | Side-by-side predicted vs. actual tables    |
| Model Timings                   | Training + inference time profiling         |

---

## 📁 How to Use

### 🔧 Setup

```bash
git clone https://github.com/your-username/sige.git
cd sige
pip install -r requirements.txt
jupyter notebook outlays.ipynb
```
___

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

---

## 🧪 How to Run

```bash
git clone https://github.com/your-username/balance-projector.git
cd balance-projector
pip install -r requirements.txt
jupyter notebook balances.ipynb
```

---

### 📁 Customize Dataset

Replace dataset ingestion cell with:

```python
import pandas as pd
df = pd.read_csv("your_dataset.csv")
X = df.drop("target_column", axis=1)
y = df["target_column"]
```

---

### 📊 Outputs

- R², MAE, MSE for each model
- Bar plots of performance scores
- Visual predicted vs. actual scatter charts
- Residual error analysis

---

## 🔮 Roadmap

- [ ] Add time series models (Prophet, ARIMA)
- [ ] Integrate GridSearchCV for model tuning
- [ ] SHAP-based interpretability
- [ ] Flask/FastAPI API for deploying forecasts
- [ ] LLM summarization of forecast outcomes

---

## 🤝 Contributing

1. 🍴 Fork the project
2. 🔧 Create a branch: `git checkout -b feat/new-feature`
3. ✅ Commit and push changes
4. 📬 Submit a pull request

---

## 📜 License

This project is licensed under the **MIT License**.

---
