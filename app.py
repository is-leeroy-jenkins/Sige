# ==================================================================================================
# Assembly:            Sige
# Filename:            app.py
# Author:              ChatGPT
# Created:             2025-12-16
# Last Modified:       2025-12-16
#
# License:             MIT (project-level)
# Contact:             N/A
# --------------------------------------------------------------------------------------------------
# Purpose:
#     Streamlit application implementing the core workflow from Sige's outlays notebook:
#         1) Upload an Excel workbook containing TAS outlay history (FY columns).
#         2) Reshape and aggregate outlays by FiscalYear (optionally by AgencyName).
#         3) Train/evaluate multiple regression models and generate FY forecasts.
#         4) Run ARIMA and Holt-Winters forecasts for comparison.
#
# Notes:
#     - The notebook hard-codes a Colab path (e.g., /content/Budget Outlays.xlsx) and expects a
#       worksheet named "Data". This app supports upload and lets the user select the sheet.
#     - The notebook includes a broad set of libraries; this app sticks to the core ones needed
#       for outlay regression + basic time series forecasting.
# ==================================================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import streamlit as st

from sklearn.linear_model import BayesianRidge, ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# --------------------------------------------------------------------------------------------------
# Guard / Validation
# --------------------------------------------------------------------------------------------------

def throw_if( name: str, value: Any ) -> None:
	"""
		
		Purpose:
		--------
		Raise ValueError if the value is None, empty, or otherwise invalid for a required
		parameter.
	
		Parameters:
		--------
		name: Parameter name.
		value: Parameter value.
	
		Returns:
		--------
		None
		
	"""
	if name is None or str( name ).strip( ) == "":
		raise ValueError( "Parameter 'name' is required." )
	
	if value is None:
		raise ValueError( f"Parameter '{name}' is required (None)." )
	
	if isinstance( value, str ) and value.strip( ) == "":
		raise ValueError( f"Parameter '{name}' is required (empty string)." )
	
	if isinstance( value, (list, tuple, dict, set) ) and len( value ) == 0:
		raise ValueError( f"Parameter '{name}' is required (empty collection)." )

# --------------------------------------------------------------------------------------------------
# Data Model
# --------------------------------------------------------------------------------------------------

@dataclass( frozen=True )
class ForecastConfig:
	"""
	Purpose:
		Central configuration for model training + forecasting.

	Parameters:
		min_fy: Minimum fiscal year to include.
		max_fy: Maximum fiscal year to include.
		test_size: Proportion of years reserved for test set (random split).
		random_state: Random seed.
		future_years: List of fiscal years to forecast (e.g., [2025, 2026]).

	Returns:
		ForecastConfig
	"""
	min_fy: int = 2012
	max_fy: int = 2024
	test_size: float = 0.25
	random_state: int = 42
	future_years: Tuple[ int, ... ] = (2025, 2026)

# --------------------------------------------------------------------------------------------------
# Core Helpers
# --------------------------------------------------------------------------------------------------

def _extract_fy_columns( df: pd.DataFrame ) -> List[ str ]:
	"""
	Purpose:
		Identify FY columns such as FY1962 ... FY2024.

	Parameters:
		df: Input dataframe.

	Returns:
		List[str]: Fiscal-year columns in ascending order by year.
	"""
	throw_if( 'df', df )
	
	fy_cols = [ c for c in df.columns if isinstance( c, str ) and c.startswith( 'FY' ) ]
	years: List[ Tuple[ int, str ] ] = [ ]
	
	for c in fy_cols:
		try:
			y = int( c.replace( 'FY', "" ).strip( ) )
			years.append( (y, c) )
		except ValueError:
			continue
	
	years.sort( key=lambda t: t[ 0 ] )
	return [ c for _, c in years ]

def load_outlays_excel( file_bytes: bytes, sheet_name: str ) -> pd.DataFrame:
	"""
	Purpose:
		Load the outlays dataset from an uploaded Excel file.

	Parameters:
		file_bytes: Uploaded file bytes.
		sheet_name: Sheet name containing the data (typically "Data").

	Returns:
		pd.DataFrame: Raw dataset.
	"""
	throw_if( 'file_bytes', file_bytes )
	throw_if( 'sheet_name', sheet_name )
	
	df = pd.read_excel( io=file_bytes, sheet_name=sheet_name )
	return df

def to_long_format( df_raw: pd.DataFrame ) -> pd.DataFrame:
	"""
	Purpose:
		Convert wide FY columns into a long format: (AgencyName, FiscalYear, Outlays).

	Parameters:
		df_raw: Raw dataset containing FY columns and metadata columns.

	Returns:
		pd.DataFrame: Long-form dataframe.
	"""
	throw_if( 'df_raw', df_raw )
	
	fy_cols = _extract_fy_columns( df_raw )
	if len( fy_cols ) == 0:
		raise ValueError( 'No FY columns found (expected columns like FY2012, FY2013, ...).' )
	
	# Try to preserve common identifiers if present.
	id_candidates = [ 'AgencyName',
	                  'BureauName',
	                  'AccountName',
	                  'FiscalYear' ]
	id_vars = [ c for c in id_candidates if c in df_raw.columns ]
	
	if 'AgencyName' not in id_vars:
		# Still allow; but the app will be more limited.
		id_vars = [ c for c in id_vars if c != 'AgencyName' ]
	
	df_long = df_raw.melt(
		id_vars=id_vars,
		value_vars=fy_cols,
		var_name='FY',
		value_name='Outlays',
	)
	
	df_long[ 'FY' ] = df_long[
		'FY' ].astype( str ).str.replace( 'FY', "", regex=False ).astype( int )
	df_long[ 'Outlays' ] = pd.to_numeric( df_long[ 'Outlays' ], errors='coerce' )
	
	# Clean: drop NaNs; replace zeros with NaN then fill where appropriate later per aggregation.
	df_long = df_long.dropna( subset=[ 'Outlays' ] )
	return df_long

def aggregate_outlays_by_year( df_long: pd.DataFrame, agency: Optional[ str ] ) -> pd.DataFrame:
	"""
	Purpose:
		Aggregate Outlays by FiscalYear, optionally filtering by AgencyName.

	Parameters:
		df_long: Long-form outlays dataframe.
		agency: Optional agency filter.

	Returns:
		pd.DataFrame: Aggregated totals by fiscal year.
	"""
	throw_if( 'df_long', df_long )
	
	df_work = df_long.copy( )
	
	if agency is not None and 'AgencyName' in df_work.columns:
		df_work = df_work[ df_work[ 'AgencyName' ] == agency ]
	
	df_grouped = (
			df_work.groupby( 'FiscalYear', as_index=False )[ 'Outlays' ]
			.sum( )
			.sort_values( 'FiscalYear' )
			.reset_index( drop=True )
	)
	return df_grouped

def build_models( ) -> Dict[ str, Any ]:
	"""
	Purpose:
		Build the set of regression models used for comparison.

	Parameters:
		None

	Returns:
		Dict[str, Any]: Name -> model instance (or a marker for polynomial).
	"""
	return \
		{
				'Linear Regression': LinearRegression( ),
				'Polynomial Regression (deg=2)': 'POLY_2',
				'Ridge Regression': Ridge( alpha=10.0, max_iter=10000 ),
				'Lasso Regression': Lasso( alpha=0.1 ),
				'Bayesian Ridge Regression': BayesianRidge( ),
				'Gradient Boosting': GradientBoostingRegressor( n_estimators=200,
					random_state=42 ),
				'Random Forest': RandomForestRegressor( n_estimators=300, random_state=42 ),
				'ElasticNet': ElasticNet( alpha=0.1, l1_ratio=0.5, random_state=42 ),
		}

def fit_predict_regressions(
		df_grouped: pd.DataFrame,
		cfg: ForecastConfig,
) -> Tuple[ pd.DataFrame, pd.DataFrame ]:
	"""
		Purpose:
		Train multiple regression models on (FiscalYear -> Outlays)
		and forecast future years.
	
		Parameters:
		df_grouped: Aggregated outlays by FiscalYear.
		cfg: Forecast configuration.
	
		Returns:
		Tuple[pd.DataFrame, pd.DataFrame]:
		metrics_df: per-model metrics ( R2, MAE, RMSE ).
		forecast_df: per-model forecasts for cfg.future_years.
	"""
	throw_if( 'df_grouped', df_grouped )
	throw_if( 'cfg', cfg )
	
	df_hist = df_grouped[ (df_grouped[ 'FiscalYear' ] >= cfg.min_fy) & (
				df_grouped[ 'FiscalYear' ] <= cfg.max_fy) ].copy( )
	if df_hist.shape[ 0 ] < 6:
		raise ValueError( 'Not enough historical years to train (need at least ~6 rows).' )
	
	X = df_hist[ 'FiscalYear' ].to_numpy( ).reshape( -1, 1 )
	y = df_hist[ 'Outlays' ].to_numpy( )
	
	X_train, X_test, y_train, y_test = train_test_split( X, y,
		test_size=cfg.test_size, random_state=cfg.random_state )
	
	models = build_models( )
	rows_metrics: List[ Dict[ str, Any ] ] = [ ]
	forecasts: Dict[ str, List[ float ] ] = { }
	
	future_X = np.array( list( cfg.future_years ), dtype=int ).reshape( -1, 1 )
	
	for name, model in models.items( ):
		if model == "POLY_2":
			poly = PolynomialFeatures( degree=2, include_bias=False )
			X_tr = poly.fit_transform( X_train )
			X_te = poly.transform( X_test )
			reg = LinearRegression( )
			reg.fit( X_tr, y_train )
			y_pred = reg.predict( X_te )
			
			fut = reg.predict( poly.transform( future_X ) )
			forecasts[ name ] = [ float( v ) for v in fut ]
		else:
			model.fit( X_train, y_train )
			y_pred = model.predict( X_test )
			
			fut = model.predict( future_X )
			forecasts[ name ] = [ float( v ) for v in fut ]
		
		mae = float( mean_absolute_error( y_test, y_pred ) )
		rmse = float( np.sqrt( mean_squared_error( y_test, y_pred ) ) )
		r2 = float( r2_score( y_test, y_pred ) )
		
		rows_metrics.append(
			{
					'Model': name,
					'R2': r2,
					'MAE': mae,
					'RMSE': rmse,
			}
		)
	
	metrics_df = pd.DataFrame( rows_metrics ).sort_values( 'R2', ascending=False ).reset_index(
		drop=True )
	
	forecast_df = pd.DataFrame( forecasts, index=list( cfg.future_years ) )
	forecast_df.index.name = 'FiscalYear'
	
	return metrics_df, forecast_df

def fit_predict_time_series(
		df_grouped: pd.DataFrame,
		cfg: ForecastConfig,
		arima_order: Tuple[ int, int, int ],
		seasonal_periods: int,
) -> pd.DataFrame:
	"""
	Purpose:
		Fit ARIMA and Holt-Winters on the historical aggregated outlays series and forecast future
		years.

	Parameters:
		df_grouped: Aggregated outlays by FiscalYear.
		cfg: Forecast configuration.
		arima_order: ARIMA(p,d,q).
		seasonal_periods: Holt-Winters seasonal period.

	Returns:
		pd.DataFrame: Forecasts for ARIMA and Holt-Winters indexed by future fiscal years.
	"""
	throw_if( 'df_grouped', df_grouped )
	throw_if( 'cfg', cfg )
	throw_if( 'arima_order', arima_order )
	throw_if( 'seasonal_periods', seasonal_periods )
	
	df_hist = df_grouped[ (df_grouped[ 'FiscalYear' ] >= cfg.min_fy) & (
				df_grouped[ 'FiscalYear' ] <= cfg.max_fy) ].copy( )
	y = df_hist[ 'Outlays' ].astype( float ).to_numpy( )
	
	steps = len( cfg.future_years )
	
	# ARIMA
	arima_fit = ARIMA( y, order=arima_order ).fit( )
	arima_forecast = arima_fit.forecast( steps=steps )
	
	# Holt-Winters (additive seasonal; if it fails, fall back to non-seasonal)
	try:
		hw_fit = ExponentialSmoothing(
			y,
			seasonal='add',
			seasonal_periods=int( seasonal_periods ),
			trend='add',
		).fit( )
		hw_forecast = hw_fit.forecast( steps )
	except Exception:
		hw_fit = ExponentialSmoothing( y, trend='add' ).fit( )
		hw_forecast = hw_fit.forecast( steps )
	
	df_ts = pd.DataFrame(
		{
				'ARIMA': np.asarray( arima_forecast, dtype=float ),
				'Holt-Winters': np.asarray( hw_forecast, dtype=float ),
		},
		index=list( cfg.future_years ),
	)
	df_ts.index.name = 'FiscalYear'
	return df_ts

def plot_series( df_grouped: pd.DataFrame, title: str ) -> plt.Figure:
	"""
	Purpose:
		Create a Matplotlib figure for outlays over time.

	Parameters:
		df_grouped: Aggregated outlays by FiscalYear.
		title: Plot title.

	Returns:
		matplotlib.figure.Figure
	"""
	throw_if( 'df_grouped', df_grouped )
	throw_if( 'title', title )
	
	fig = plt.figure( )
	plt.plot( df_grouped[ 'FiscalYear' ], df_grouped[ 'Outlays' ] )
	plt.title( title )
	plt.xlabel( 'Fiscal Year' )
	plt.ylabel( 'Outlays' )
	plt.grid( True )
	return fig

# --------------------------------------------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------------------------------------------

def main( ) -> None:
	"""
	Purpose:
		Entry point for Streamlit application.

	Parameters:
		None

	Returns:
		None
	"""
	st.set_page_config( page_title='Sige | Federal Outlay Projections', layout='wide' )
	
	st.title( 'Sige — Federal Outlay Projections' )
	st.caption( 'Upload historical outlays (FY columns), explore trends, and compare forecasting '
	            'models.' )
	
	with st.sidebar:
		st.header( '1) Data' )
		upload = st.file_uploader( 'Upload Excel (.xlsx)', type=[ 'xlsx' ] )
		sheet_name = st.text_input( 'Sheet name', value='Data' )
		
		st.header( '2) Training Window' )
		min_fy = st.number_input( 'Min FY', min_value=1900, max_value=2100, value=2012, step=1 )
		max_fy = st.number_input( 'Max FY', min_value=1900, max_value=2100, value=2024, step=1 )
		
		st.header( '3) Forecast Years' )
		fut_1 = st.number_input( 'Future FY #1', min_value=1900, max_value=2100, value=2025,
			step=1 )
		fut_2 = st.number_input( 'Future FY #2', min_value=1900, max_value=2100, value=2026,
			step=1 )
		
		st.header( '4) Time Series Settings' )
		arima_p = st.number_input( 'ARIMA p', min_value=0, max_value=10, value=5, step=1 )
		arima_d = st.number_input( 'ARIMA d', min_value=0, max_value=5, value=1, step=1 )
		arima_q = st.number_input( 'ARIMA q', min_value=0, max_value=10, value=0, step=1 )
		seasonal_periods = st.number_input( 'Holt-Winters seasonal_periods', min_value=1,
			max_value=20, value=5, step=1 )
		
		st.header( '5) Split' )
		test_size = st.slider( 'Test size', min_value=0.10, max_value=0.50, value=0.25, step=0.05 )
		random_state = st.number_input( 'Random state', min_value=0, max_value=10_000, value=42,
			step=1 )
	
	if upload is None:
		st.info( 'Upload an Excel workbook to begin.' )
		st.stop( )
	
	try:
		df_raw = load_outlays_excel( upload.getvalue( ), sheet_name=sheet_name )
		df_long = to_long_format( df_raw )
	except Exception as ex:
		st.error( f"Failed to read/parse the uploaded file: {ex}" )
		st.stop( )
	
	# Agency filter (if present)
	agency: Optional[ str ] = None
	if 'AgencyName' in df_long.columns:
		agencies = sorted( [ a for a in df_long[ 'AgencyName' ].dropna( ).unique( ).tolist( ) if
		                     str( a ).strip( ) != "" ] )
		agency = st.selectbox( 'Agency filter (optional)', options=[ '(All)' ] + agencies )
		if agency == '(All)':
			agency = None
	
	df_grouped = aggregate_outlays_by_year( df_long, agency=agency )
	
	cfg = ForecastConfig(
		min_fy=int( min_fy ),
		max_fy=int( max_fy ),
		test_size=float( test_size ),
		random_state=int( random_state ),
		future_years=(int( fut_1 ), int( fut_2 )),
	)
	
	tab_overview, tab_models, tab_download = st.tabs( [ 'Overview',
	                                                    'Models',
	                                                    'Export' ] )
	
	with tab_overview:
		st.subheader( 'Outlays by Fiscal Year' )
		left, right = st.columns( [ 1,
		                            1 ] )
		
		with left:
			st.dataframe( df_grouped, use_container_width=True, height=420 )
		
		with right:
			fig = plot_series(
				df_grouped=df_grouped,
				title='Total Outlays by Fiscal Year' if agency is None else f'Outlays by Fiscal '
				                                                            f'Year — {agency}',
			)
			st.pyplot( fig, clear_figure=True )
		
		st.subheader( 'Raw Preview' )
		st.dataframe( df_raw.head( 25 ), use_container_width=True )
	
	with tab_models:
		st.subheader( 'Regression Models (FiscalYear → Outlays)' )
		
		try:
			metrics_df, forecast_df = fit_predict_regressions( df_grouped=df_grouped, cfg=cfg )
		except Exception as ex:
			st.error( f'Regression training failed: {ex}' )
			st.stop( )
		
		col_a, col_b = st.columns( [ 1,
		                             1 ] )
		with col_a:
			st.markdown( '**Model Performance (Holdout Split)**' )
			st.dataframe( metrics_df, use_container_width=True, height=340 )
		
		with col_b:
			st.markdown( '**Forecasts (Regression Models)**' )
			st.dataframe( forecast_df.round( 2 ), use_container_width=True, height=340 )
		
		st.subheader( 'Time Series Models' )
		
		try:
			df_ts = fit_predict_time_series(
				df_grouped=df_grouped,
				cfg=cfg,
				arima_order=(int( arima_p ), int( arima_d ), int( arima_q )),
				seasonal_periods=int( seasonal_periods ),
			)
		except Exception as ex:
			st.error( f'Time series forecasting failed: {ex}' )
			st.stop( )
		
		st.dataframe( df_ts.round( 2 ), use_container_width=True )
		
		# Combined plot: historical + forecasts
		st.subheader( 'Historical vs Forecast (Top Regression + ARIMA + Holt-Winters)' )
		
		top_model = str( metrics_df.iloc[ 0 ][ 'Model' ] )
		hist = df_grouped[ (df_grouped[ 'FiscalYear' ] >= cfg.min_fy) & (
					df_grouped[ 'FiscalYear' ] <= cfg.max_fy) ].copy( )
		
		fig2 = plt.figure( )
		plt.plot( hist[ 'FiscalYear' ], hist[ 'Outlays' ], label='Historical' )
		
		fut_years = list( cfg.future_years )
		plt.plot( fut_years,
			forecast_df[ top_model ].to_numpy( ), marker='o', label=f'Top Regression: {
			top_model}' )
		plt.plot( fut_years, df_ts[ 'ARIMA' ].to_numpy( ), marker='o', label='ARIMA' )
		plt.plot( fut_years, df_ts[ 'Holt-Winters' ].to_numpy( ), marker='o',
			label='Holt-Winters' )
		
		plt.title( 'Forecast Comparison' )
		plt.xlabel( 'Fiscal Year' )
		plt.ylabel( 'Outlays' )
		plt.grid( True )
		plt.legend( )
		st.pyplot( fig2, clear_figure=True )
	
	with tab_download:
		st.subheader( 'Export Results' )
		
		# Recompute (cheap) so this tab works independently.
		metrics_df, forecast_df = fit_predict_regressions( df_grouped=df_grouped, cfg=cfg )
		df_ts = fit_predict_time_series(
			df_grouped=df_grouped,
			cfg=cfg,
			arima_order=(int( arima_p ), int( arima_d ), int( arima_q )),
			seasonal_periods=int( seasonal_periods ),
		)
		
		st.download_button(
			label='Download regression metrics (CSV)',
			data=metrics_df.to_csv( index=False ).encode( 'utf-8' ),
			file_name='sige_regression_metrics.csv',
			mime='text/csv',
		)
		
		st.download_button(
			label='Download regression forecasts (CSV)',
			data=forecast_df.to_csv( index=True ).encode( 'utf-8' ),
			file_name='sige_regression_forecasts.csv',
			mime='text/csv',
		)
		
		st.download_button(
			label='Download time series forecasts (CSV)',
			data=df_ts.to_csv( index=True ).encode( 'utf-8' ),
			file_name='sige_time_series_forecasts.csv',
			mime='text/csv',
		)

if __name__ == "__main__":
	main( )
