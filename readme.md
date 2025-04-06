– Federal Outlay Projections with Python
![](https://github.com/is-leeroy-jenkins/Sige/blob/master/resources/assets/img/git/SigeProject.png)


**Sige** is a specialized machine learning and time series forecasting tool built for **federal financial analysts**, **budget officers**, and **data scientists** working with **Treasury Account Symbols (TAS)** and **federal outlays**. It combines classical regression, decision trees, gradient boosting, support vector machines, and time series modeling to project **monthly or quarterly outlays** at the Treasury Account level.


---




## 🚀 Features

### 📈 Full-Funnel Forecasting Pipeline
- Load and structure TAS-level outlay data from SF-133-style or custom tabular reports
- Extract features from **monthly or quarterly outlay data**
- Train and compare multiple models across time slices

### 🔢 Models Included

| Model                     | Type         | Module                              |
|--------------------------|--------------|-------------------------------------|
| Linear Regression         | Classical ML | `sklearn.linear_model.LinearRegression` |
| Decision Tree Regressor  | Tree-Based   | `sklearn.tree.DecisionTreeRegressor` |
| Random Forest Regressor  | Ensemble     | `sklearn.ensemble.RandomForestRegressor` |
| Gradient Boosting Regressor | Ensemble | `sklearn.ensemble.GradientBoostingRegressor` |
| Support Vector Regressor (SVR) | Kernel-Based | `sklearn.svm.SVR`                |
| ARIMA                    | Time Series  | `statsmodels.tsa.arima.model.ARIMA` |

### 📊 Forecast Targets
- **Monthly/quarterly outlays**
- **End-of-year projections**
- **Year-over-year outlay trends**

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

## 📁 How to Use

### 🔧 Setup

```bash
git clone https://github.com/your-username/sige.git
cd sige
pip install -r requirements.txt
jupyter notebook outlays.ipynb
