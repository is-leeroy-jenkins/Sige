'''
	******************************************************************************************
	    Assembly:                Sige
	    Filename:                app.py
	    Author:                  Terry D. Eppler
	    Created:                 05-31-2022
	
	    Last Modified By:        Terry D. Eppler
	    Last Modified On:        05-01-2025
	******************************************************************************************
	<copyright file="app.py" company="Terry D. Eppler">
	

	           Copyright ©  2022  Terry Eppler
	
	   Permission is hereby granted, free of charge, to any person obtaining a copy
	   of this software and associated documentation files (the “Software”),
	   to deal in the Software without restriction,
	   including without limitation the rights to use,
	   copy, modify, merge, publish, distribute, sublicense,
	   and/or sell copies of the Software,
	   and to permit persons to whom the Software is furnished to do so,
	   subject to the following conditions:
	
	   The above copyright notice and this permission notice shall be included in all
	   copies or substantial portions of the Software.
	
	   THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
	   INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
	   FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT.
	   IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
	   DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
	   ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
	   DEALINGS IN THE SOFTWARE.
	
	   You can contact me at:  terryeppler@gmail.com or eppler.terry@epa.gov
	
	</copyright>
	<summary>
	  app.py
	</summary>
	******************************************************************************************
'''
from __future__ import annotations

import config as cfg
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
# CONSTANTS
# --------------------------------------------------------------------------------------------------
LOGO = r'resources/sige_logo.ico'
FAVICON = r'resources/assets/favicon.ico'
BLUE_DIVIDER = "<div style='height:2px;align:left;background:#0078FC;margin:6px 0 10px 0;'></div>"

# =====================================================
# Utilities
# =====================================================

def throw_if(name: str, value: Any) -> None:
    if value is None:
        raise ValueError(f"Parameter '{name}' is required.")
    if isinstance(value, (list, tuple, dict, set)) and len(value) == 0:
        raise ValueError(f"Parameter '{name}' is empty.")

def style_subheaders( ) -> None:
	"""
	
		Purpose:
		_________
		Sets the style of subheaders in the main UI
		
	"""
	st.markdown(
		"""
		<style>
		div[data-testid="stMarkdownContainer"] h2,
		div[data-testid="stMarkdownContainer"] h3,
		div[data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] h2,
		div[data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] h3 {
			color: rgb(0, 120, 252) !important;
		}
		</style>
		""",
		unsafe_allow_html=True,
	)

def fmt_num( x: float ) -> str:
	"""
	
		Purpose:
		Format a number with comma and 3 decimal places where applicable.
		
		Returns:
			str
	
	"""
	try:
		if abs( x ) >= 1:
			return f"{x:,.2f}"
		return f"{x:.6f}"
	except Exception:
		return str( x )

def numeric_columns( df: pd.DataFrame ) -> list[ str ]:
	"""
	
		Purpose:
		--------
		Return a list of numeric column names from the provided DataFrame.
	
		Parameters:
		-----------
		df: pd.DataFrame
			The dataframe to inspect.
	
		Returns:
		--------
		list[str]: Column names whose dtype is numeric (excludes booleans).
	
	"""
	if df is None or df.empty:
		return [ ]
	num_cols = df.select_dtypes( include=[ np.number ] ).columns.tolist( )
	return [ c for c in num_cols if not pd.api.types.is_bool_dtype( df[ c ] ) ]

def categorical_columns( df: pd.DataFrame ) -> list[ str ]:
	"""
	
		Purpose:
		--------
		Return a list of categorical / object column names from the DataFrame.
	
		Parameters:
		-----------
		df: pd.DataFrame
			The dataframe to inspect.
	
		Returns:
		--------
		list[str]: Column names with object or category dtypes.
		
	"""
	if df is None or df.empty:
		return [ ]
	return df.select_dtypes( include=[ 'object', 'category' ] ).columns.tolist( )

# ---------------------- Config
@dataclass(frozen=True)
class ForecastConfig:
    min_fy: int
    max_fy: int
    test_size: float = 0.25
    random_state: int = 42
    future_years: Tuple[int, ...] = (2025, 2026)

# ---------------------- Loaders

def load_outlays_excel( file_bytes: bytes, sheet_name: str ) -> pd.DataFrame:
	return pd.read_excel( io=file_bytes, sheet_name=sheet_name )

def load_outlays_with_fallback( upload: Optional[ st.runtime.uploaded_file_manager.UploadedFile ],
		use_fallback: bool, sheet_name: str, fallback_path: str=cfg.DATA ) -> pd.DataFrame:
	"""
	
		Load data based on explicit user intent.
		Upload always takes precedence if present.
		
	"""
	if upload is not None:
		return load_outlays_excel( upload.getvalue( ), sheet_name )
	
	if not use_fallback:
		raise RuntimeError( 'No upload provided and fallback loading is disabled.' )
	
	path = Path( fallback_path )
	if not path.exists( ):
		raise FileNotFoundError( f'Fallback file not found: {path.resolve( )}' )
	
	return pd.read_excel( path, sheet_name=sheet_name )

def _extract_fy_columns( df: pd.DataFrame ) -> List[ str ]:
	fy_cols = [ ]
	for c in df.columns:
		if isinstance( c, str ) and c.startswith( 'FY' ):
			try:
				int( c.replace( 'FY', "" ) )
				fy_cols.append( c )
			except ValueError:
				pass
	return sorted( fy_cols, key=lambda c: int( c.replace( 'FY', "" ) ) )

def to_long_format( df_raw: pd.DataFrame ) -> pd.DataFrame:
	fy_cols = _extract_fy_columns( df_raw )
	if not fy_cols:
		raise ValueError( 'No fiscal-year columns (FYxxxx) found.' )
	
	id_vars = [ c for c in df_raw.columns if c not in fy_cols ]
	df = df_raw.melt( id_vars=id_vars, value_vars=fy_cols, var_name='FY', value_name='Outlays' )
	df[ 'FiscalYear' ] = df[ 'FY' ].astype( str ).str.replace( 'FY', '', regex=False ).astype( int )
	df[ 'Outlays' ] = pd.to_numeric( df[ 'Outlays' ], errors='coerce' )
	
	return df.dropna( subset=[ 'Outlays' ] ).drop( columns=[ 'FY' ] )

def apply_filters( df: pd.DataFrame, agency: Optional[ str ], main_account: Optional[ str ],
		tas: Optional[ str ], ) -> pd.DataFrame:
	if agency and 'AgencyName' in df.columns:
		df = df[ df[ 'AgencyName' ] == agency ]
	
	if main_account:
		for col in ('MainAccountCode', 'MainAccount'):
			if col in df.columns:
				df = df[ df[ col ] == main_account ]
				break
	
	if tas:
		for col in ('TreasuryAccountSymbol', 'TAS'):
			if col in df.columns:
				df = df[ df[ col ] == tas ]
				break
	
	return df

def aggregate_outlays( df: pd.DataFrame ) -> pd.DataFrame:
	return (df.groupby( 'FiscalYear', as_index=False )[ 'Outlays' ].sum( )
	        .sort_values( 'FiscalYear' ).reset_index( drop=True ))

def build_models( ) -> Dict[ str, Any ]:
	return {
			'Linear Regression': LinearRegression( ),
			'Polynomial Regression (deg=2)': 'POLY_2',
			'Ridge': Ridge( alpha=10.0 ),
			'Lasso': Lasso( alpha=0.1 ),
			'Bayesian Ridge': BayesianRidge( ),
			'ElasticNet': ElasticNet( alpha=0.1, l1_ratio=0.5 ),
			'Gradient Boosting': GradientBoostingRegressor( n_estimators=200 ),
			'Random Forest': RandomForestRegressor( n_estimators=300 ),
	}

def fit_predict_regressions( df: pd.DataFrame,
		cfg: ForecastConfig ) -> Tuple[ pd.DataFrame, pd.DataFrame ]:
	hist = df[ (df[ 'FiscalYear' ] >= cfg.min_fy) & (df[ 'FiscalYear' ] <= cfg.max_fy) ]
	if hist.shape[ 0 ] < 6:
		raise ValueError( 'Not enough historical data for regression.' )
	
	X = hist[ 'FiscalYear' ].to_numpy( ).reshape( -1, 1 )
	y = hist[ 'Outlays' ].to_numpy( )
	X_tr, X_te, y_tr, y_te = train_test_split( X, y, test_size=cfg.test_size,
		random_state=cfg.random_state )
	
	future_X = np.array( cfg.future_years ).reshape( -1, 1 )
	metrics, forecasts = [ ], { }
	for name, model in build_models( ).items( ):
		if model == 'POLY_2':
			poly = PolynomialFeatures( 2, include_bias=False )
			reg = LinearRegression( )
			reg.fit( poly.fit_transform( X_tr ), y_tr )
			pred = reg.predict( poly.transform( X_te ) )
			fut = reg.predict( poly.transform( future_X ) )
		else:
			model.fit( X_tr, y_tr )
			pred = model.predict( X_te )
			fut = model.predict( future_X )
		
		metrics.append( { 'Model': name, 'R2': r2_score( y_te, pred ),
		                  'MAE': mean_absolute_error( y_te, pred ),
		                  'RMSE': np.sqrt( mean_squared_error( y_te, pred ) ), } )
		
		forecasts[ name ] = fut
	
	return ( pd.DataFrame( metrics ).sort_values( 'R2', ascending=False ),
			pd.DataFrame( forecasts, index=cfg.future_years ), )

def fit_predict_time_series( df: pd.DataFrame, cfg: ForecastConfig,
		order: Tuple[ int, int, int ], season: int, ) -> pd.DataFrame:
	y = df[ 'Outlays' ].to_numpy( )
	steps = len( cfg.future_years )
	
	arima = ARIMA( y, order=order ).fit( ).forecast( steps )
	
	try:
		hw = ExponentialSmoothing(
			y, trend='add', seasonal='add', seasonal_periods=season
		).fit( ).forecast( steps )
	except Exception:
		hw = ExponentialSmoothing( y, trend='add' ).fit( ).forecast( steps )
	
	return pd.DataFrame(
		{ 'ARIMA': arima, 'Holt-Winters': hw },
		index=cfg.future_years,
	)

def humanize_number( x: Any, decimals: int=2 ) -> str:
	"""
	
		Purpose:
		--------
		Convert a number to a human-readable string with suffixes K/M/B/T, keeping decimals reasonable.
	
		Parameters:
		-----------
		x: Any
			Input value.
		decimals: int
			Decimal places for the mantissa.
	
		Returns:
		--------
		str
			Human-readable string.
			
	"""
	try:
		if x is None or (isinstance( x, float ) and np.isnan( x )):
			return ""
		v = float( x )
	except Exception:
		return str( x )
	
	av = abs( v )
	if av < 1_000:
		# Small number: keep a modest decimal policy
		if av < 10:
			return f"{v:,.{min( 4, max( 0, decimals + 2 ) )}f}"
		return f"{v:,.{decimals}f}"
	
	suffixes = [
			(1_000_000_000_000, "T"),
			(1_000_000_000, "B"),
			(1_000_000, "M"),
			(1_000, "K"),
	]
	for base, suf in suffixes:
		if av >= base:
			return f"{v / base:,.{decimals}f}{suf}"
	return f"{v:,.{decimals}f}"

def render_table( df: pd.DataFrame, title=None, caption=None, precision=4,
		dark_mode=True, max_rows=500, humanize_large=True ) -> None:
	"""
	
		Purpose:
		--------
		Render a DataFrame as a readable HTML table with sane decimals
		and optional humanized magnitudes.
	
		Parameters:
		-----------
		df: pd.DataFrame
			Table to render.
		title: str | None
			Optional heading.
		caption: str | None
			Optional caption.
		precision: int
			Default numeric precision.
		dark_mode: bool
			Dark palette if True.
		max_rows: int
			Max rows to render.
		humanize_large: bool
			If True, large values are shown with suffixes K/M/B/T.
			
	"""
	if title:
		st.markdown( f"#### {title}" )
	if df is None or df.empty:
		st.info( "No data to display." )
		return
	
	df_show = df.copy( )
	if len( df_show ) > max_rows:
		df_show = df_show.head( max_rows )
	
	num_cols = df_show.select_dtypes( include=[ np.number ] ).columns.tolist( )
	
	def _fmt_cell( v: Any ) -> str:
		if humanize_large:
			return humanize_number( v, decimals=min( 2, max( 0, precision - 2 ) ) )
		# fallback: fixed precision
		try:
			if v is None or (isinstance( v, float ) and np.isnan( v )):
				return ""
			return f"{float( v ):,.{precision}f}"
		except Exception:
			return str( v )
	
	for c in num_cols:
		df_show[ c ] = df_show[ c ].map( _fmt_cell )
	
	if dark_mode:
		text = "#F9FAFB"
		header_bg = "#1F2937"
		row_even = "#0B1220"
		row_odd = "#111827"
		border = "#374151"
	else:
		text = "#111111"
		header_bg = "#F3F5F7"
		row_even = "#FCFCFD"
		row_odd = "#FFFFFF"
		border = "#D0D0D0"
	
	styler = (
			df_show.style
			.set_table_styles( [
					{ "selector": "table",
					  "props": [ ("border-collapse", "collapse"), ("width", "100%") ] },
					{ "selector": "th", "props": [
							("background-color", header_bg),
							("color", text),
							("border", f"1px solid {border}"),
							("padding", "6px"),
							("font-weight", "600"),
							("text-align", "left"),
							("white-space", "nowrap"),
					] },
					{ "selector": "td", "props": [
							("color", text),
							("border", f"1px solid {border}"),
							("padding", "6px"),
							("white-space", "nowrap"),
					] },
					{ "selector": "tr:nth-child(even) td",
					  "props": [ ("background-color", row_even) ] },
					{ "selector": "tr:nth-child(odd) td",
					  "props": [ ("background-color", row_odd) ] },
			] )
	)
	
	st.markdown( styler.to_html( ), unsafe_allow_html=True )
	
	if caption:
		st.caption( caption )
		
# =====================================================
# UI Setup
# =====================================================
def main() -> None:
    st.set_page_config( page_title='Sige', layout='wide', page_icon=FAVICON, )
    
    style_subheaders( )
    st.logo( cfg.LOGO, size='large' )
    st.header( 'Outlay Projector' )
    st.divider( )
    st.markdown( '##### Parameters' )
    
    # =====================================================
    # SIDEBAR
    # =====================================================
    with st.sidebar:
        use_fallback = st.checkbox( 'Use fallback data', value=True )
        upload = st.file_uploader( 'Upload Excel (.xlsx)', type=['xlsx'] )
        sheet = st.text_input( 'Sheet name', value='Data' )

        st.caption(  'Data source: '  +
                     ('Upload' if upload is not None else 'Fallback' if use_fallback else 'None') )

        try:
            df_raw = load_outlays_with_fallback( upload=upload,  use_fallback=use_fallback,
                sheet_name=sheet, )
            df_long = to_long_format( df_raw )
        except Exception as ex:
            st.error(f'Data load failed: { ex }')
            st.stop()

        fy_min = int( df_long[ 'FiscalYear' ].min( ) )
        fy_max = int( df_long[ 'FiscalYear' ].max( ) )

        st.divider( )
        min_fy, max_fy = st.slider( 'Training window (Fiscal Years)', min_value=fy_min,
	        max_value=fy_max, value=(max( fy_min, 2012 ), fy_max), step=1, )
    
    # ---------------------- Model controls
    ana_c1, ana_c2, ana_c3, ana_c4, ana_c5 = st.columns(
	    [ 0.20, 0.20, 0.20, 0.20, 0.20, ], border=True )
    with ana_c1:
	    f1 = st.number_input( 'Future FY #1', value=fy_max + 1, step=1 )
    
    with ana_c2:
	    f2 = st.number_input( 'Future FY #2', value=fy_max + 2, step=1 )
    
    with ana_c3:
	    p = st.number_input( 'ARIMA P', 0, 10, 5 )
    
    with ana_c4:
	    d = st.number_input( 'ARIMA D', 0, 5, 1 )
    
    with ana_c5:
	    q = st.number_input( 'ARIMA Q', 0, 10, 0 )

    # ---------------------- Filters
    st.divider( )
    st.markdown( '##### Filters' )
    ftr_c1, ftr_c2, ftr_c3 = st.columns( [ 0.33, 0.33, 0.33 ], border=True )
    with ftr_c1:
	    agency = None
	    if 'AgencyName' in df_long.columns:
		    agency = st.selectbox( 'Agency',
			    [ '(All)' ] + sorted( df_long[ 'AgencyName' ].dropna( ).unique( ) ), )
		    if agency == '(All)':
			    agency = None
    
    with ftr_c2:
	    main_account = None
	    if agency:
		    for col in ('MainAccountCode', 'MainAccount'):
			    if col in df_long.columns:
				    opts = sorted(
					    df_long[ df_long[ 'AgencyName' ] == agency ][ col ].dropna( ).unique( ) )
				    main_account = st.selectbox( 'Main Account', [ '(All)' ] + opts )
				    if main_account == '(All)':
					    main_account = None
			    break
			    
	    tas = None
	    if main_account:
		    for col in ('TreasuryAccountSymbol', 'TAS'):
			    if col in df_long.columns:
				    opts = sorted(
					    df_long[ (df_long[ 'AgencyName' ] == agency)
						    & ( df_long[ col.replace( 'TreasuryAccountSymbol', 'MainAccountCode' ) ]
								    == main_account) ][ col ].dropna( ).unique( ) )
				    tas = st.selectbox( 'TAS', [ '(All)' ] + opts )
				    if tas == '(All)':
					    tas = None
				    break
    
    with ftr_c3:
	    season = st.number_input( 'HW Seasonal Periods', 1, 20, 5 )
	    
    # ---------------------- Analysis
    df_filtered = apply_filters( df_long, agency, main_account, tas )
    df_grouped = aggregate_outlays( df_filtered )
    
    cfg = ForecastConfig( min_fy=min_fy, max_fy=max_fy, future_years=(int( f1 ), int( f2 )), )
    
    st.divider( )
    st.markdown( '##### Aggregated Outlays' )
    st.data_editor( df_grouped, use_container_width=True )
    
    metrics, forecasts = fit_predict_regressions( df_grouped, cfg )
    ts = fit_predict_time_series( df_grouped, cfg, (p, d, q), season )
    
    st.divider( )
    st.markdown( '##### Regression Performance' )
    st.data_editor( metrics, use_container_width=True )
    
    st.divider( )
    st.markdown( '##### Forecasts')
    st.data_editor( pd.concat( [ forecasts, ts ], axis=1 ),  use_container_width=True, )

if __name__ == "__main__":
    main()
