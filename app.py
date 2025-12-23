# ==================================================================================================
# Assembly:            Sige
# Filename:            app.py
# Created:             2025-12-17
# License:             MIT
# --------------------------------------------------------------------------------------------------
# Purpose:
#     Streamlit application for federal outlay analysis with:
#       • Upload OR fallback loader
#       • Agency × Main Account × TAS slicing
#       • Data-derived fiscal-year bounds
#       • Multi-model regression
#       • ARIMA + Holt-Winters forecasting
# ==================================================================================================

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    BayesianRidge,
    ElasticNet,
)
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing


# --------------------------------------------------------------------------------------------------
# Guard
# --------------------------------------------------------------------------------------------------

def throw_if(name: str, value: Any) -> None:
    if value is None:
        raise ValueError(f"Parameter '{name}' is required.")
    if isinstance(value, (list, tuple, dict, set)) and len(value) == 0:
        raise ValueError(f"Parameter '{name}' is empty.")


# --------------------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class ForecastConfig:
    min_fy: int
    max_fy: int
    test_size: float = 0.25
    random_state: int = 42
    future_years: Tuple[int, ...] = (2025, 2026)


# --------------------------------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------------------------------

def load_outlays_excel(file_bytes: bytes, sheet_name: str) -> pd.DataFrame:
    return pd.read_excel(io=file_bytes, sheet_name=sheet_name)


def load_outlays_with_fallback(
    upload: Optional[st.runtime.uploaded_file_manager.UploadedFile],
    sheet_name: str,
    fallback_path: str = r"data/Budget Outlays.xlsx",
) -> pd.DataFrame:
    if upload is not None:
        return load_outlays_excel(upload.getvalue(), sheet_name)

    path = Path(fallback_path)
    if not path.exists():
        raise FileNotFoundError(f"Fallback file not found: {path.resolve()}")

    return pd.read_excel(path, sheet_name=sheet_name)


# --------------------------------------------------------------------------------------------------
# Transformations
# --------------------------------------------------------------------------------------------------

def _extract_fy_columns(df: pd.DataFrame) -> List[str]:
    fy_cols = []
    for c in df.columns:
        if isinstance(c, str) and c.startswith("FY"):
            try:
                int(c.replace("FY", ""))
                fy_cols.append(c)
            except ValueError:
                pass
    return sorted(fy_cols, key=lambda c: int(c.replace("FY", "")))


def to_long_format(df_raw: pd.DataFrame) -> pd.DataFrame:
    fy_cols = _extract_fy_columns(df_raw)
    if not fy_cols:
        raise ValueError("No fiscal-year columns (FYxxxx) found.")

    id_vars = [c for c in df_raw.columns if c not in fy_cols]

    df = df_raw.melt(
        id_vars=id_vars,
        value_vars=fy_cols,
        var_name="FY",
        value_name="Outlays",
    )

    df["FiscalYear"] = (
        df["FY"].astype(str).str.replace("FY", "", regex=False).astype(int)
    )
    df["Outlays"] = pd.to_numeric(df["Outlays"], errors="coerce")

    return df.dropna(subset=["Outlays"]).drop(columns=["FY"])


# --------------------------------------------------------------------------------------------------
# Slicing + Aggregation
# --------------------------------------------------------------------------------------------------

def apply_filters(
    df: pd.DataFrame,
    agency: Optional[str],
    main_account: Optional[str],
    tas: Optional[str],
) -> pd.DataFrame:
    if agency and "AgencyName" in df.columns:
        df = df[df["AgencyName"] == agency]

    if main_account:
        for col in ("MainAccountCode", "MainAccount"):
            if col in df.columns:
                df = df[df[col] == main_account]
                break

    if tas:
        for col in ("TreasuryAccountSymbol", "TAS"):
            if col in df.columns:
                df = df[df[col] == tas]
                break

    return df


def aggregate_outlays(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("FiscalYear", as_index=False)["Outlays"]
        .sum()
        .sort_values("FiscalYear")
        .reset_index(drop=True)
    )


# --------------------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------------------

def build_models() -> Dict[str, Any]:
    return {
        "Linear Regression": LinearRegression(),
        "Polynomial Regression (deg=2)": "POLY_2",
        "Ridge": Ridge(alpha=10.0),
        "Lasso": Lasso(alpha=0.1),
        "Bayesian Ridge": BayesianRidge(),
        "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=200),
        "Random Forest": RandomForestRegressor(n_estimators=300),
    }


def fit_predict_regressions(
    df: pd.DataFrame, cfg: ForecastConfig
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    hist = df[
        (df["FiscalYear"] >= cfg.min_fy)
        & (df["FiscalYear"] <= cfg.max_fy)
    ]

    if hist.shape[0] < 6:
        raise ValueError("Not enough historical data for regression.")

    X = hist["FiscalYear"].to_numpy().reshape(-1, 1)
    y = hist["Outlays"].to_numpy()

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state
    )

    future_X = np.array(cfg.future_years).reshape(-1, 1)

    metrics, forecasts = [], {}

    for name, model in build_models().items():
        if model == "POLY_2":
            poly = PolynomialFeatures(2, include_bias=False)
            reg = LinearRegression()
            reg.fit(poly.fit_transform(X_tr), y_tr)
            pred = reg.predict(poly.transform(X_te))
            fut = reg.predict(poly.transform(future_X))
        else:
            model.fit(X_tr, y_tr)
            pred = model.predict(X_te)
            fut = model.predict(future_X)

        metrics.append(
            {
                "Model": name,
                "R2": r2_score(y_te, pred),
                "MAE": mean_absolute_error(y_te, pred),
                "RMSE": np.sqrt(mean_squared_error(y_te, pred)),
            }
        )
        forecasts[name] = fut

    return (
        pd.DataFrame(metrics).sort_values("R2", ascending=False),
        pd.DataFrame(forecasts, index=cfg.future_years),
    )


def fit_predict_time_series(
    df: pd.DataFrame,
    cfg: ForecastConfig,
    order: Tuple[int, int, int],
    season: int,
) -> pd.DataFrame:

    y = df["Outlays"].to_numpy()
    steps = len(cfg.future_years)

    arima = ARIMA(y, order=order).fit().forecast(steps)

    try:
        hw = ExponentialSmoothing(
            y, trend="add", seasonal="add", seasonal_periods=season
        ).fit().forecast(steps)
    except Exception:
        hw = ExponentialSmoothing(y, trend="add").fit().forecast(steps)

    return pd.DataFrame(
        {"ARIMA": arima, "Holt-Winters": hw},
        index=cfg.future_years,
    )


# --------------------------------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="Sige", layout="wide", page_icon=r'resources/assets/favicon.ico' )
    st.title("Federal Outlay Projector")

    with st.sidebar:
        upload = st.file_uploader("Upload Excel (.xlsx)", type=["xlsx"])
        sheet = st.text_input("Sheet name", value="Data")

    try:
        df_raw = load_outlays_with_fallback(upload, sheet)
        df_long = to_long_format(df_raw)
    except Exception as ex:
        st.error(f"Data load failed: {ex}")
        st.stop()

    # ---------------------------
    # Fiscal-year bounds (derived)
    # ---------------------------

    fy_min = int(df_long["FiscalYear"].min())
    fy_max = int(df_long["FiscalYear"].max())

    with st.sidebar:
        st.markdown("---")
        min_fy, max_fy = st.slider(
            "Training window (Fiscal Years)",
            min_value=fy_min,
            max_value=fy_max,
            value=(max(fy_min, 2012), fy_max),
            step=1,
        )

        f1 = st.number_input("Future FY #1", value=fy_max + 1, step=1)
        f2 = st.number_input("Future FY #2", value=fy_max + 2, step=1)

        p = st.number_input("ARIMA p", 0, 10, 5)
        d = st.number_input("ARIMA d", 0, 5, 1)
        q = st.number_input("ARIMA q", 0, 10, 0)
        season = st.number_input("HW seasonal_periods", 1, 20, 5)

    # ---------------------------
    # Hierarchical slicing
    # ---------------------------

    agency = None
    if "AgencyName" in df_long.columns:
        agency = st.selectbox(
            "Agency",
            ["(All)"] + sorted(df_long["AgencyName"].dropna().unique()),
        )
        if agency == "(All)":
            agency = None

    main_account = None
    if agency:
        for col in ("MainAccountCode", "MainAccount"):
            if col in df_long.columns:
                opts = sorted(
                    df_long[df_long["AgencyName"] == agency][col].dropna().unique()
                )
                main_account = st.selectbox("Main Account", ["(All)"] + opts)
                if main_account == "(All)":
                    main_account = None
                break

    tas = None
    if main_account:
        for col in ("TreasuryAccountSymbol", "TAS"):
            if col in df_long.columns:
                opts = sorted(
                    df_long[
                        (df_long["AgencyName"] == agency)
                        & (df_long[col.replace("TreasuryAccountSymbol", "MainAccountCode")] == main_account)
                    ][col]
                    .dropna()
                    .unique()
                )
                tas = st.selectbox("TAS", ["(All)"] + opts)
                if tas == "(All)":
                    tas = None
                break

    df_filtered = apply_filters(df_long, agency, main_account, tas)
    df_grouped = aggregate_outlays(df_filtered)

    cfg = ForecastConfig(
        min_fy=min_fy,
        max_fy=max_fy,
        future_years=(int(f1), int(f2)),
    )

    st.subheader("Aggregated Outlays")
    st.dataframe(df_grouped, use_container_width=True)

    metrics, forecasts = fit_predict_regressions(df_grouped, cfg)
    ts = fit_predict_time_series(df_grouped, cfg, (p, d, q), season)

    st.subheader("Regression Performance")
    st.dataframe(metrics, use_container_width=True)

    st.subheader("Forecasts")
    st.dataframe(pd.concat([forecasts, ts], axis=1), use_container_width=True)


if __name__ == "__main__":
    main()
