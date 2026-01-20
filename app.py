import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet

# Constants
REQUIRED_COLUMNS = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']

@st.cache_data
def load_and_validate_data(file_path='bitcoin.csv'):
    """Robust data ingestion with comprehensive validation."""
    try:
        df = pd.read_csv(file_path)
        
        # Schema inference and validation
        missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
            st.stop()
        
        # Parse dates and set index
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        if df['Date'].isnull().any():
            st.error("Invalid date formats found.")
            st.stop()
        df = df.sort_values('Date').set_index('Date')
        
        # Handle duplicates
        initial_len = len(df)
        df = df[~df.index.duplicated(keep='first')]
        if len(df) < initial_len:
            st.warning(f"Removed {initial_len - len(df)} duplicate dates.")
        
        # Handle missing values
        numeric_cols = ['Close', 'High', 'Low', 'Open', 'Volume']
        df[numeric_cols] = df[numeric_cols].fillna(method='ffill').fillna(method='bfill')
        if df[numeric_cols].isnull().any().any():
            st.error("Unable to fill all missing values.")
            st.stop()
        
        # Validate numeric integrity
        df[numeric_cols] = df[numeric_cols].astype(float)
        if (df[numeric_cols] < 0).any().any():
            st.warning("Negative values found in numeric columns.")
        
        # Ensure logical consistency (High >= Low, etc.)
        if not (df['High'] >= df['Low']).all():
            st.warning("Inconsistent High/Low values detected.")
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

def compute_advanced_kpis(df):
    """Compute comprehensive financial KPIs."""
    latest = df['Close'].iloc[-1]
    prev_day = df['Close'].iloc[-2] if len(df) > 1 else latest
    pct_change_day = (latest - prev_day) / prev_day * 100
    
    # Week/month change
    week_ago = df['Close'].iloc[-8] if len(df) > 7 else df['Close'].iloc[0]
    pct_change_week = (latest - week_ago) / week_ago * 100
    month_ago = df['Close'].iloc[-31] if len(df) > 30 else df['Close'].iloc[0]
    pct_change_month = (latest - month_ago) / month_ago * 100
    
    ath = df['High'].max()
    ath_date = df['High'].idxmax().date()
    atl = df['Low'].min()
    atl_date = df['Low'].idxmin().date()
    
    returns = df['Close'].pct_change().dropna()
    avg_daily_return = returns.mean() * 100
    annualized_vol = returns.std() * np.sqrt(365) * 100
    
    # Additional risk metrics
    sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else np.nan
    downside_returns = returns[returns < 0]
    sortino_ratio = (returns.mean() / downside_returns.std()) * np.sqrt(252) if len(downside_returns) > 0 and downside_returns.std() > 0 else np.nan
    var_95 = -np.percentile(returns, 5) * 100
    
    return {
        'current_price': latest,
        'pct_change_day': pct_change_day,
        'pct_change_week': pct_change_week,
        'pct_change_month': pct_change_month,
        'ath': ath,
        'ath_date': ath_date,
        'atl': atl,
        'atl_date': atl_date,
        'avg_daily_return': avg_daily_return,
        'annualized_vol': annualized_vol,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'var_95': var_95
    }

def compute_deep_eda(df):
    """Deep exploratory data analysis."""
    returns = df['Close'].pct_change().dropna()
    log_returns = np.log(1 + returns)
    cum_returns = (1 + returns).cumprod()
    
    # Distribution stats
    skewness = stats.skew(returns)
    kurtosis = stats.kurtosis(returns)
    
    # Trend analysis
    x = np.arange(len(df))
    slope, intercept = np.polyfit(x, df['Close'], 1)
    
    # Seasonality
    monthly_returns = returns.groupby(returns.index.month).mean()
    yearly_vol = returns.groupby(returns.index.year).std()
    
    # Drawdown analysis
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    max_drawdown_date = drawdown.idxmin().date()
    # Recovery period (simplified)
    peak_after_dd = running_max[drawdown.idxmin():].idxmax()
    recovery_days = (peak_after_dd - drawdown.idxmin()).days if peak_after_dd > drawdown.idxmin() else None
    
    # Volatility regimes
    rolling_vol = returns.rolling(30).std() * np.sqrt(365)
    vol_regime = pd.cut(rolling_vol, bins=[0, rolling_vol.quantile(0.33), rolling_vol.quantile(0.67), rolling_vol.max()], labels=['Low', 'Medium', 'High'])
    
    # Correlation
    price_vol_corr = df['Close'].pct_change().corr(df['Volume'].pct_change())
    
    # Seasonal decomposition
    seasonal = seasonal_decompose(df['Close'], model='additive', period=30)
    
    return {
        'returns': returns,
        'log_returns': log_returns,
        'cum_returns': cum_returns,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'slope': slope,
        'monthly_returns': monthly_returns,
        'yearly_vol': yearly_vol,
        'max_drawdown': max_drawdown,
        'max_drawdown_date': max_drawdown_date,
        'recovery_days': recovery_days,
        'vol_regime': vol_regime,
        'price_vol_corr': price_vol_corr,
        'seasonal': seasonal
    }

def compute_technical_indicators(df):
    """Compute technical indicators: RSI, MACD, Bollinger Bands."""
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = df['Close'].ewm(span=12).mean()
    ema26 = df['Close'].ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    
    # Bollinger Bands
    sma20 = df['Close'].rolling(20).mean()
    std20 = df['Close'].rolling(20).std()
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    
    # Stochastic Oscillator
    low_min = df['Low'].rolling(14).min()
    high_max = df['High'].rolling(14).max()
    stoch_k = 100 * (df['Close'] - low_min) / (high_max - low_min)
    stoch_d = stoch_k.rolling(3).mean()
    
    return {
        'rsi': rsi,
        'macd': macd,
        'signal': signal,
        'upper_bb': upper,
        'lower_bb': lower,
        'sma_bb': sma20,
        'stoch_k': stoch_k,
        'stoch_d': stoch_d
    }

def create_advanced_price_chart(df, show_ma=True, show_trend=True, show_bb=False, chart_type='line'):
    """Advanced price chart with overlays."""
    fig = go.Figure()
    
    if chart_type == 'candlestick':
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Candlestick'))
    else:
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close Price', line=dict(color='blue')))
    
    if show_ma:
        df['MA20'] = df['Close'].rolling(20).mean()
        df['MA50'] = df['Close'].rolling(50).mean()
        df['MA200'] = df['Close'].rolling(200).mean()
        fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], name='MA20', line=dict(color='orange')))
        fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], name='MA50', line=dict(color='green')))
        fig.add_trace(go.Scatter(x=df.index, y=df['MA200'], name='MA200', line=dict(color='red')))
    
    if show_trend:
        x = np.arange(len(df))
        slope, intercept = np.polyfit(x, df['Close'], 1)
        trend_line = intercept + slope * x
        fig.add_trace(go.Scatter(x=df.index, y=trend_line, name='Trend', line=dict(color='purple', dash='dash')))
    
    if show_bb:
        sma20 = df['Close'].rolling(20).mean()
        std20 = df['Close'].rolling(20).std()
        upper = sma20 + 2 * std20
        lower = sma20 - 2 * std20
        fig.add_trace(go.Scatter(x=df.index, y=upper, name='Upper BB', line=dict(color='gray', dash='dot')))
        fig.add_trace(go.Scatter(x=df.index, y=lower, name='Lower BB', line=dict(color='gray', dash='dot')))
        fig.add_trace(go.Scatter(x=df.index, y=sma20, name='SMA20 BB', line=dict(color='black', dash='dash')))
    
    fig.update_layout(title='Bitcoin Price Chart', xaxis_title='Date', yaxis_title='Price ($)')
    return fig

def create_returns_analysis_chart(returns):
    """Returns distribution with stats."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=returns, nbinsx=50, name='Returns'))
    fig.add_vline(x=returns.mean(), line_dash="dash", line_color="red", annotation_text="Mean")
    fig.add_vline(x=returns.median(), line_dash="dash", line_color="green", annotation_text="Median")
    fig.update_layout(title='Daily Returns Distribution', xaxis_title='Return (%)', yaxis_title='Frequency')
    return fig

def create_volatility_regime_chart(df, vol_regime):
    """Volatility regimes over time."""
    fig = px.scatter(df, x=df.index, y=df['Close'], color=vol_regime, title='Price Colored by Volatility Regime')
    return fig

def create_rsi_chart(df, indicators):
    """RSI chart."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=indicators['rsi'], name='RSI', line=dict(color='blue')))
    fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
    fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
    fig.update_layout(title='Relative Strength Index (RSI)', xaxis_title='Date', yaxis_title='RSI')
    return fig

def create_macd_chart(df, indicators):
    """MACD chart."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=indicators['macd'], name='MACD', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df.index, y=indicators['signal'], name='Signal', line=dict(color='red')))
    fig.update_layout(title='MACD Indicator', xaxis_title='Date', yaxis_title='Value')
    return fig

def create_stochastic_chart(df, indicators):
    """Stochastic Oscillator chart."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=indicators['stoch_k'], name='%K', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df.index, y=indicators['stoch_d'], name='%D', line=dict(color='red')))
    fig.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="Overbought")
    fig.add_hline(y=20, line_dash="dash", line_color="green", annotation_text="Oversold")
    fig.update_layout(title='Stochastic Oscillator', xaxis_title='Date', yaxis_title='Value')
    return fig

def create_seasonal_decomp_chart(df, seasonal):
    """Seasonal decomposition chart."""
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, subplot_titles=['Observed', 'Trend', 'Seasonal', 'Residual'])
    fig.add_trace(go.Scatter(x=df.index, y=seasonal.observed, name='Observed'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=seasonal.trend, name='Trend'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=seasonal.seasonal, name='Seasonal'), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=seasonal.resid, name='Residual'), row=4, col=1)
    fig.update_layout(height=800, title_text="Seasonal Decomposition")
    return fig

def forecast_arima(df, periods=30):
    """Forecast using ARIMA."""
    try:
        model = ARIMA(df['Close'], order=(5,1,0))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=periods)
        forecast_index = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=periods, freq='D')
        return pd.Series(forecast.values, index=forecast_index)
    except:
        return pd.Series(dtype=float)

def forecast_prophet(df, periods=30):
    """Forecast using Prophet."""
    try:
        prophet_df = df.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
        model = Prophet()
        model.fit(prophet_df)
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)
        forecast_series = forecast.set_index('ds')['yhat'][-periods:]
        return forecast_series
    except:
        return pd.Series(dtype=float)

def create_forecast_chart(df, forecast, model='ARIMA'):
    """Forecast chart."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Historical Price', line=dict(color='blue')))
    if not forecast.empty:
        fig.add_trace(go.Scatter(x=forecast.index, y=forecast, name='Forecast', line=dict(color='orange', dash='dash')))
    fig.update_layout(title=f'Price Forecast ({model})', xaxis_title='Date', yaxis_title='Price ($)')
    return fig

def generate_advanced_insights(kpis, eda, df):
    """Generate analytical, data-driven insights."""
    insights = []
    insights.append(f"Bitcoin is currently trading at ${kpis['current_price']:.2f}, reflecting a {kpis['pct_change_day']:+.2f}% daily change, {kpis['pct_change_week']:+.2f}% weekly, and {kpis['pct_change_month']:+.2f}% monthly.")
    insights.append(f"All-time high of ${kpis['ath']:.2f} was achieved on {kpis['ath_date']}, while the all-time low of ${kpis['atl']:.2f} occurred on {kpis['atl_date']}.")
    insights.append(f"Average daily return stands at {kpis['avg_daily_return']:.2f}%, with annualized volatility at {kpis['annualized_vol']:.2f}%, indicating {'high' if kpis['annualized_vol'] > 50 else 'moderate'} risk.")
    insights.append(f"Return distribution shows skewness of {eda['skewness']:.2f} and kurtosis of {eda['kurtosis']:.2f}, suggesting {'right-skewed' if eda['skewness'] > 0 else 'left-skewed'} returns with {'heavy tails' if eda['kurtosis'] > 3 else 'normal tails'}.")
    insights.append(f"The maximum drawdown of {eda['max_drawdown']:.2f} occurred on {eda['max_drawdown_date']}, with recovery taking {eda['recovery_days']} days." if eda['recovery_days'] else f"Maximum drawdown of {eda['max_drawdown']:.2f} on {eda['max_drawdown_date']}, recovery period undetermined.")
    insights.append(f"Price-volume correlation is {eda['price_vol_corr']:.2f}, indicating {'strong' if abs(eda['price_vol_corr']) > 0.5 else 'weak'} relationship between price movements and trading volume.")
    insights.append(f"Overall trend slope is {eda['slope']:.4f}, confirming {'an upward' if eda['slope'] > 0 else 'a downward'} trajectory in Bitcoin's price.")
    insights.append(f"Sharpe ratio of {kpis['sharpe_ratio']:.2f} indicates {'excellent' if kpis['sharpe_ratio'] > 1 else 'good' if kpis['sharpe_ratio'] > 0.5 else 'poor'} risk-adjusted returns.")
    insights.append(f"Sortino ratio of {kpis['sortino_ratio']:.2f} focuses on downside risk, showing {'strong' if kpis['sortino_ratio'] > 1 else 'moderate'} performance.")
    insights.append(f"Value at Risk (95%) is {kpis['var_95']:.2f}%, meaning there's a 5% chance of losing more than this in a day.")
    z_scores = (eda['returns'] - eda['returns'].mean()) / eda['returns'].std()
    num_anomalies = (z_scores.abs() > 3).sum()
    insights.append(f"Detected {num_anomalies} anomalous return days (Z-score > 3), indicating potential market shocks or data irregularities.")
    return insights

# Main app
st.set_page_config(page_title="Institutional Bitcoin Analytics", layout="wide", page_icon="📈")
st.title("Institutional Bitcoin Market Analytics Platform")
st.markdown("### 📊 Advanced analytics for Bitcoin price data with institutional-grade insights and machine learning forecasting.")

# Load data
df = load_and_validate_data()

# Sidebar controls
st.sidebar.header("Analysis Controls")
date_range = st.sidebar.slider(
    "Select Date Range",
    min_value=df.index.min().date(),
    max_value=df.index.max().date(),
    value=(df.index.min().date(), df.index.max().date())
)

st.sidebar.subheader("Visualization Options")
show_ma = st.sidebar.checkbox("Show Moving Averages", value=True)
show_trend = st.sidebar.checkbox("Show Trend Line", value=True)
show_vol_regime = st.sidebar.checkbox("Color by Volatility Regime", value=False)

chart_selector = st.sidebar.multiselect(
    "Select Charts to Display",
    ['Price Analysis', 'Returns Distribution', 'Volatility Regime', 'Volume Correlation', 'Volume Chart', 'Anomaly Detection'],
    default=['Price Analysis', 'Returns Distribution']
)

st.sidebar.subheader("Technical Indicators")
show_bb = st.sidebar.checkbox("Bollinger Bands", value=False)
chart_type = st.sidebar.selectbox("Price Chart Type", ['line', 'candlestick'])
show_rsi = st.sidebar.checkbox("RSI Chart", value=False)
show_macd = st.sidebar.checkbox("MACD Chart", value=False)
show_stochastic = st.sidebar.checkbox("Stochastic Oscillator", value=False)
show_seasonal = st.sidebar.checkbox("Seasonal Decomposition", value=False)
show_forecast = st.sidebar.checkbox("Price Forecast", value=False)
if show_forecast:
    forecast_periods = st.sidebar.slider("Forecast Periods (days)", 7, 90, 30)
    forecast_model = st.sidebar.selectbox("Forecast Model", ['ARIMA', 'Prophet'])
else:
    forecast_periods = 30
    forecast_model = 'ARIMA'

# Filter data
filtered_df = df.loc[pd.to_datetime(date_range[0]):pd.to_datetime(date_range[1])]

# Compute metrics
kpis = compute_advanced_kpis(filtered_df)
eda = compute_deep_eda(filtered_df)
indicators = compute_technical_indicators(filtered_df)
if show_forecast:
    if forecast_model == 'Prophet':
        forecast = forecast_prophet(filtered_df, forecast_periods)
    else:
        forecast = forecast_arima(filtered_df, forecast_periods)
else:
    forecast = pd.Series(dtype=float)

# KPI Dashboard
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🔧 Technical Analysis", "⚠️ Risk & Forecasting", "📈 Insights & Export"])

with tab1:
    st.header("Key Financial Metrics")
    with st.expander("Core Metrics", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Current Price", f"${kpis['current_price']:.2f}", f"{kpis['pct_change_day']:+.2f}% (1D)")
        with col2:
            st.metric("Weekly Change", f"{kpis['pct_change_week']:+.2f}%")
        with col3:
            st.metric("Monthly Change", f"{kpis['pct_change_month']:+.2f}%")
        with col4:
            st.metric("Avg Daily Return", f"{kpis['avg_daily_return']:.2f}%")

    # Analytics Visualizations
    st.header("Analytics Visualizations")
    if 'Price Analysis' in chart_selector:
        with st.expander("Price Chart with Indicators", expanded=True):
            st.plotly_chart(create_advanced_price_chart(filtered_df, show_ma, show_trend, show_bb, chart_type), width='stretch')

    if 'Returns Distribution' in chart_selector:
        with st.expander("Returns Analysis"):
            st.plotly_chart(create_returns_analysis_chart(eda['returns']), width='stretch')

    if 'Volume Correlation' in chart_selector:
        with st.expander("Price-Volume Correlation"):
            corr = eda['price_vol_corr']
            st.write(f"Correlation coefficient: {corr:.3f}")
            fig = px.scatter(filtered_df, x='Close', y='Volume', trendline="ols", title="Price vs Volume Scatter")
            st.plotly_chart(fig, width='stretch')

    if 'Volume Chart' in chart_selector:
        with st.expander("Volume Chart"):
            fig = go.Figure()
            fig.add_trace(go.Bar(x=filtered_df.index, y=filtered_df['Volume'], name='Volume'))
            fig.update_layout(title='Trading Volume', xaxis_title='Date', yaxis_title='Volume')
            st.plotly_chart(fig, width='stretch')

with tab2:
    st.header("Technical Indicators")
    if show_rsi:
        with st.expander("RSI Indicator"):
            st.plotly_chart(create_rsi_chart(filtered_df, indicators), width='stretch')

    if show_macd:
        with st.expander("MACD Indicator"):
            st.plotly_chart(create_macd_chart(filtered_df, indicators), width='stretch')

    if show_stochastic:
        with st.expander("Stochastic Oscillator"):
            st.plotly_chart(create_stochastic_chart(filtered_df, indicators), width='stretch')

    if show_seasonal:
        with st.expander("Seasonal Decomposition"):
            st.plotly_chart(create_seasonal_decomp_chart(filtered_df, eda['seasonal']), width='stretch')

    if 'Anomaly Detection' in chart_selector:
        with st.expander("Anomaly Detection"):
            z_scores = (eda['returns'] - eda['returns'].mean()) / eda['returns'].std()
            anomaly_mask = z_scores.abs() > 3
            anomaly_dates = filtered_df.index[1:][anomaly_mask]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=filtered_df.index, y=filtered_df['Close'], name='Price', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=anomaly_dates, y=filtered_df.loc[anomaly_dates, 'Close'], mode='markers', name='Anomalies', marker=dict(color='red', size=8, symbol='x')))
            fig.update_layout(title='Price Chart with Anomalies (Z-score > 3)', xaxis_title='Date', yaxis_title='Price ($)')
            st.plotly_chart(fig, width='stretch')

with tab3:
    st.header("Risk & Forecasting")
    with st.expander("Risk & Extremes"):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("ATH", f"${kpis['ath']:.2f}", f"on {kpis['ath_date']}")
        with col2:
            st.metric("ATL", f"${kpis['atl']:.2f}", f"on {kpis['atl_date']}")
        with col3:
            st.metric("Annualized Vol", f"{kpis['annualized_vol']:.2f}%")
        with col4:
            st.metric("Max Drawdown", f"{eda['max_drawdown']:.2f}", f"on {eda['max_drawdown_date']}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Sharpe Ratio", f"{kpis['sharpe_ratio']:.2f}")
        with col2:
            st.metric("Sortino Ratio", f"{kpis['sortino_ratio']:.2f}")
        with col3:
            st.metric("VaR 95%", f"{kpis['var_95']:.2f}%")

    if 'Volatility Regime' in chart_selector and show_vol_regime:
        with st.expander("Volatility Regimes"):
            st.plotly_chart(create_volatility_regime_chart(filtered_df, eda['vol_regime']), width='stretch')

    if show_forecast:
        with st.expander("Price Forecast"):
            st.plotly_chart(create_forecast_chart(filtered_df, forecast, forecast_model), width='stretch')

with tab4:
    st.header("Analytical Insights")
    with st.expander("Data-Driven Analysis", expanded=True):
        insights = generate_advanced_insights(kpis, eda, filtered_df)
        for insight in insights:
            st.write(f"• {insight}")

    # Export Data
    st.header("Export Data")
    st.download_button(
        label="Download Filtered Data as CSV",
        data=filtered_df.to_csv().encode('utf-8'),
        file_name='bitcoin_filtered_data.csv',
        mime='text/csv',
        key='download-csv'
    )

    # Data Integrity Summary
    st.header("Data Integrity & Summary")
    with st.expander("Dataset Overview"):
        st.write(f"**Total Records:** {len(filtered_df)}")
        st.write(f"**Date Range:** {filtered_df.index.min().date()} to {filtered_df.index.max().date()}")
        st.write(f"**Missing Values:** {filtered_df.isnull().sum().sum()}")
        st.write(f"**Data Types:** {filtered_df.dtypes.to_dict()}")
        st.write(f"**Statistical Summary:**")
        st.dataframe(filtered_df.describe())

st.markdown("---")
st.markdown("**Built with ❤️ using Streamlit, Plotly, and advanced Python libraries.** | *Empowering data-driven decisions in cryptocurrency markets.*")
