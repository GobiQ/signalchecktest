import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Tactical Allocation",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for testfol.io-like styling
st.markdown("""
<style>
    /* Main styling */
    .main {
        padding: 0;
    }
    
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* Header styling */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a1a1a;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Signal card styling */
    .signal-card {
        background: white;
        border: 1px solid #e1e5e9;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .signal-card:hover {
        box-shadow: 0 4px 16px rgba(0,0,0,0.15);
        transform: translateY(-2px);
    }
    
    .signal-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
    }
    
    .signal-name {
        font-size: 1.2rem;
        font-weight: 600;
        color: #1a1a1a;
        margin: 0;
    }
    
    .signal-type {
        background: #e3f2fd;
        color: #1976d2;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
    }
    
    /* Metric styling */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: white;
        border: 1px solid #e1e5e9;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    }
    
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    
    .metric-label {
        font-size: 0.8rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .positive {
        color: #2e7d32;
    }
    
    .negative {
        color: #d32f2f;
    }
    
    .neutral {
        color: #666;
    }
    
    /* Allocation slider styling */
    .allocation-container {
        background: white;
        border: 1px solid #e1e5e9;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .allocation-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.5rem;
    }
    
    .allocation-value {
        font-weight: 600;
        color: #1976d2;
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    
    /* Chart styling */
    .chart-container {
        background: white;
        border: 1px solid #e1e5e9;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background-color: #1976d2;
    }
    
    /* Dataframe styling */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Custom container styling */
    .custom-container {
        background: white;
        border: 1px solid #e1e5e9;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'signals' not in st.session_state:
    st.session_state.signals = []
if 'allocations' not in st.session_state:
    st.session_state.allocations = {}
if 'backtest_results' not in st.session_state:
    st.session_state.backtest_results = None
if 'selected_signal' not in st.session_state:
    st.session_state.selected_signal = None

# Helper functions
def calculate_rsi(prices: pd.Series, window: int = 14, method: str = "wilders") -> pd.Series:
    """Calculate RSI using specified method"""
    if method == "wilders":
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
    else:  # sma method
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
    return rsi

def get_stock_data(ticker: str, start_date=None, end_date=None, exclusions=None) -> pd.Series:
    """Fetch stock data with timezone normalization"""
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date)
        
        if data.empty:
            return pd.Series(dtype=float)
        
        if data.index.tz is not None:
            data.index = data.index.tz_localize(None)
        
        return data['Close']
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return pd.Series(dtype=float)

def calculate_equity_curve(signals: pd.Series, prices: pd.Series, allocation: float = 1.0) -> pd.Series:
    """Calculate equity curve based on signals and prices"""
    equity_curve = pd.Series(1.0, index=prices.index)
    current_equity = 1.0
    in_position = False
    entry_equity = 1.0
    entry_price = None
    
    for date in prices.index:
        current_signal = signals[date] if date in signals.index else 0
        current_price = prices[date]
        
        if current_signal == 1 and not in_position:
            in_position = True
            entry_equity = current_equity
            entry_price = current_price
            
        elif current_signal == 0 and in_position:
            trade_return = (current_price - entry_price) / entry_price
            current_equity = entry_equity * (1 + trade_return * allocation)
            in_position = False
        
        if in_position:
            current_equity = entry_equity * (current_price / entry_price)
        
        equity_curve[date] = current_equity
    
    if in_position:
        final_price = prices.iloc[-1]
        trade_return = (final_price - entry_price) / entry_price
        current_equity = entry_equity * (1 + trade_return * allocation)
        equity_curve.iloc[-1] = current_equity
    
    return equity_curve

def calculate_metrics(equity_curve: pd.Series, returns: pd.Series) -> dict:
    """Calculate comprehensive performance metrics"""
    if len(returns) == 0:
        return {
            'total_return': 0.0,
            'annualized_return': 0.0,
            'volatility': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'total_trades': 0,
            'avg_trade_return': 0.0
        }
    
    total_return = (equity_curve.iloc[-1] - 1) * 100
    days = (equity_curve.index[-1] - equity_curve.index[0]).days
    annualized_return = ((equity_curve.iloc[-1] ** (365/days)) - 1) * 100 if days > 0 else 0
    
    volatility = returns.std() * np.sqrt(252) * 100
    sharpe_ratio = (annualized_return / 100) / (volatility / 100) if volatility > 0 else 0
    
    downside_returns = returns[returns < 0]
    downside_deviation = downside_returns.std() * np.sqrt(252) * 100 if len(downside_returns) > 0 else 0
    sortino_ratio = (annualized_return / 100) / (downside_deviation / 100) if downside_deviation > 0 else 0
    
    rolling_max = equity_curve.expanding().max()
    drawdown = (equity_curve - rolling_max) / rolling_max
    max_drawdown = drawdown.min() * 100
    
    trade_returns = returns[returns != 0]
    win_rate = (trade_returns > 0).mean() * 100 if len(trade_returns) > 0 else 0
    total_trades = len(trade_returns)
    avg_trade_return = trade_returns.mean() * 100 if len(trade_returns) > 0 else 0
    
    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'total_trades': total_trades,
        'avg_trade_return': avg_trade_return
    }

def calculate_sma(prices: pd.Series, window: int) -> pd.Series:
    """Calculate Simple Moving Average"""
    return prices.rolling(window=window).mean()

def calculate_ema(prices: pd.Series, window: int) -> pd.Series:
    """Calculate Exponential Moving Average"""
    return prices.ewm(span=window).mean()

def calculate_cumulative_return(prices: pd.Series, window: int) -> pd.Series:
    """Calculate cumulative return over the specified window"""
    return (prices / prices.shift(window) - 1) * 100

def calculate_max_drawdown_series(prices: pd.Series, window: int) -> pd.Series:
    """Calculate rolling max drawdown over the specified window"""
    rolling_max = prices.rolling(window=window).max()
    drawdown = (prices - rolling_max) / rolling_max * 100
    return drawdown

def calculate_indicator(prices: pd.Series, indicator_type: str, days: int = None) -> pd.Series:
    """Calculate the specified indicator"""
    if indicator_type == "Current Price":
        return prices
    elif indicator_type == "SMA":
        return calculate_sma(prices, days)
    elif indicator_type == "EMA":
        return calculate_ema(prices, days)
    elif indicator_type == "RSI":
        return calculate_rsi(prices, days)
    elif indicator_type == "Cumulative Return":
        return calculate_cumulative_return(prices, days)
    elif indicator_type == "Max Drawdown":
        return calculate_max_drawdown_series(prices, days)
    else:
        return prices

def evaluate_signal_condition(indicator1_values: pd.Series, indicator2_values: pd.Series, operator: str) -> pd.Series:
    """Evaluate the signal condition based on the operator"""
    if operator == ">":
        return (indicator1_values > indicator2_values).astype(int)
    elif operator == "<":
        return (indicator1_values < indicator2_values).astype(int)
    elif operator == ">=":
        return (indicator1_values >= indicator2_values).astype(int)
    elif operator == "<=":
        return (indicator1_values <= indicator2_values).astype(int)
    elif operator == "==":
        return (indicator1_values == indicator2_values).astype(int)
    elif operator == "!=":
        return (indicator1_values != indicator2_values).astype(int)
    else:
        return pd.Series(0, index=indicator1_values.index)

# Main app
st.markdown('<h1 class="main-header">📊 Tactical Allocation</h1>', unsafe_allow_html=True)

# Sidebar for signal management
with st.sidebar:
    st.header("🎯 Signal Management")
    
    # Signal creation
    with st.expander("➕ Add New Signal", expanded=True):
        signal_name = st.text_input("Signal Name", placeholder="e.g., QQQ RSI Oversold")
        
        # Signal type selection
        signal_type = st.selectbox("Signal Type", ["Custom Indicator", "RSI Threshold", "RSI Comparison"])
        
        if signal_type == "Custom Indicator":
            st.subheader("📊 Signal Configuration")
            
            # Target ticker
            target_ticker = st.text_input("Target Ticker", value="SPY", help="The ticker to buy/sell based on the signal")
            
            # Signal logic builder
            st.write("**Is true if:**")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # First indicator
                indicator1 = st.selectbox(
                    "Indicator 1",
                    ["SMA", "EMA", "Current Price", "Cumulative Return", "Max Drawdown", "RSI"],
                    key="indicator1"
                )
                
                # Days field for indicators that need it
                if indicator1 not in ["Current Price"]:
                    days1 = st.number_input(
                        f"# of Days for {indicator1}",
                        min_value=1,
                        max_value=252,
                        value=14,
                        key="days1"
                    )
            
            with col2:
                # Operator
                operator = st.selectbox(
                    "Operator",
                    [">", "<", ">=", "<=", "==", "!="],
                    key="operator"
                )
            
            with col3:
                # Second indicator
                indicator2 = st.selectbox(
                    "Indicator 2",
                    ["SMA", "EMA", "Current Price", "Cumulative Return", "Max Drawdown", "RSI"],
                    key="indicator2"
                )
                
                # Days field for second indicator
                if indicator2 not in ["Current Price"]:
                    days2 = st.number_input(
                        f"# of Days for {indicator2}",
                        min_value=1,
                        max_value=252,
                        value=14,
                        key="days2"
                    )
            
            # Display the signal logic
            if indicator1 not in ["Current Price"] and indicator2 not in ["Current Price"]:
                st.info(f"**Signal Logic:** {indicator1}({days1}) {operator} {indicator2}({days2})")
            elif indicator1 not in ["Current Price"]:
                st.info(f"**Signal Logic:** {indicator1}({days1}) {operator} {indicator2}")
            elif indicator2 not in ["Current Price"]:
                st.info(f"**Signal Logic:** {indicator1} {operator} {indicator2}({days2})")
            else:
                st.info(f"**Signal Logic:** {indicator1} {operator} {indicator2}")
            
            if st.button("Add Signal", type="primary"):
                signal = {
                    'name': signal_name,
                    'type': signal_type,
                    'target_ticker': target_ticker,
                    'indicator1': indicator1,
                    'indicator2': indicator2,
                    'operator': operator,
                    'days1': days1 if indicator1 not in ["Current Price"] else None,
                    'days2': days2 if indicator2 not in ["Current Price"] else None
                }
                st.session_state.signals.append(signal)
                st.success(f"Signal '{signal_name}' added!")
                st.rerun()
        
        elif signal_type == "RSI Threshold":
            col1, col2 = st.columns(2)
            with col1:
                signal_ticker = st.text_input("Signal Ticker", value="QQQ")
                target_ticker = st.text_input("Target Ticker", value="SPY")
            with col2:
                rsi_period = st.number_input("RSI Period", min_value=1, max_value=50, value=14)
                rsi_threshold = st.number_input("RSI Threshold", min_value=0.0, max_value=100.0, value=30.0, step=0.5)
            
            comparison = st.selectbox("Condition", ["less_than", "greater_than"], 
                                   format_func=lambda x: "RSI ≤ threshold" if x == "less_than" else "RSI ≥ threshold")
            
            if st.button("Add Signal", type="primary"):
                signal = {
                    'name': signal_name,
                    'type': signal_type,
                    'signal_ticker': signal_ticker,
                    'target_ticker': target_ticker,
                    'rsi_period': rsi_period,
                    'rsi_threshold': rsi_threshold,
                    'comparison': comparison
                }
                st.session_state.signals.append(signal)
                st.success(f"Signal '{signal_name}' added!")
                st.rerun()
        
        elif signal_type == "RSI Comparison":
            col1, col2 = st.columns(2)
            with col1:
                signal_ticker = st.text_input("Signal Ticker", value="QQQ")
                comparison_ticker = st.text_input("Comparison Ticker", value="SPY")
            with col2:
                target_ticker = st.text_input("Target Ticker", value="TQQQ")
                rsi_period = st.number_input("RSI Period", min_value=1, max_value=50, value=14)
            
            comparison_operator = st.selectbox("Comparison", ["less_than", "greater_than"],
                                            format_func=lambda x: "Signal RSI < Comparison RSI" if x == "less_than" else "Signal RSI > Comparison RSI")
            
            if st.button("Add Signal", type="primary"):
                signal = {
                    'name': signal_name,
                    'type': signal_type,
                    'signal_ticker': signal_ticker,
                    'comparison_ticker': comparison_ticker,
                    'target_ticker': target_ticker,
                    'rsi_period': rsi_period,
                    'comparison_operator': comparison_operator
                }
                st.session_state.signals.append(signal)
                st.success(f"Signal '{signal_name}' added!")
                st.rerun()
    
    # Signal list
    if st.session_state.signals:
        st.subheader("📋 Active Signals")
        for i, signal in enumerate(st.session_state.signals):
            with st.container():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**{signal['name']}**")
                    if signal['type'] == "Custom Indicator":
                        indicator1_text = f"{signal['indicator1']}({signal['days1']})" if signal['days1'] else signal['indicator1']
                        indicator2_text = f"{signal['indicator2']}({signal['days2']})" if signal['days2'] else signal['indicator2']
                        st.caption(f"{indicator1_text} {signal['operator']} {indicator2_text} → {signal['target_ticker']}")
                    elif signal['type'] == "RSI Threshold":
                        st.caption(f"{signal['signal_ticker']} RSI {signal['rsi_period']}-day {signal['comparison']} {signal['rsi_threshold']} → {signal['target_ticker']}")
                    else:
                        st.caption(f"{signal['signal_ticker']} vs {signal['comparison_ticker']} RSI {signal['comparison_operator']} → {signal['target_ticker']}")
                with col2:
                    if st.button("🗑️", key=f"delete_{i}"):
                        st.session_state.signals.pop(i)
                        st.rerun()

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📈 Backtest Configuration")
    
    # Date range selection
    col1a, col1b = st.columns(2)
    with col1a:
        start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=365*2))
    with col1b:
        end_date = st.date_input("End Date", value=datetime.now())
    
    # Benchmark selection
    benchmark_ticker = st.selectbox("Benchmark", ["SPY", "QQQ", "VTI", "BIL"], 
                                  help="Benchmark for comparison")
    
    # Run backtest button
    if st.button("🚀 Run Backtest", type="primary", use_container_width=True):
        if not st.session_state.signals:
            st.error("Please add at least one signal before running backtest.")
        else:
            with st.spinner("Running backtest..."):
                # Fetch data for all tickers
                all_tickers = set()
                for signal in st.session_state.signals:
                    if signal['type'] == "Custom Indicator":
                        all_tickers.add(signal['target_ticker'])
                    elif signal['type'] == "RSI Threshold":
                        all_tickers.add(signal['signal_ticker'])
                        all_tickers.add(signal['target_ticker'])
                    elif signal['type'] == "RSI Comparison":
                        all_tickers.add(signal['signal_ticker'])
                        all_tickers.add(signal['comparison_ticker'])
                        all_tickers.add(signal['target_ticker'])
                all_tickers.add(benchmark_ticker)
                
                # Fetch data
                data = {}
                for ticker in all_tickers:
                    data[ticker] = get_stock_data(ticker, start_date, end_date)
                
                # Calculate signals and equity curves
                results = {}
                combined_equity = pd.Series(1.0, index=data[benchmark_ticker].index)
                
                for signal in st.session_state.signals:
                    if signal['type'] == "Custom Indicator":
                        # For custom indicators, we need to determine which ticker to use for each indicator
                        # For now, we'll use the target ticker for both indicators
                        # In a more sophisticated version, we'd allow users to specify different tickers
                        target_ticker = signal['target_ticker']
                        
                        # Calculate indicators for the signal
                        indicator1_values = calculate_indicator(data[target_ticker], signal['indicator1'], signal['days1'])
                        indicator2_values = calculate_indicator(data[target_ticker], signal['indicator2'], signal['days2'])
                        
                        # Evaluate the signal condition
                        signals = evaluate_signal_condition(indicator1_values, indicator2_values, signal['operator'])
                        
                        # Calculate equity curve
                        equity_curve = calculate_equity_curve(signals, data[target_ticker], 
                                                           st.session_state.allocations.get(signal['name'], 1.0))
                        
                        # Calculate returns and metrics
                        returns = equity_curve.pct_change().dropna()
                        metrics = calculate_metrics(equity_curve, returns)
                        
                        results[signal['name']] = {
                            'equity_curve': equity_curve,
                            'signals': signals,
                            'metrics': metrics
                        }
                        
                        combined_equity = combined_equity * (1 + (equity_curve - 1) * st.session_state.allocations.get(signal['name'], 1.0))
                    
                    elif signal['type'] == "RSI Threshold":
                        rsi = calculate_rsi(data[signal['signal_ticker']], signal['rsi_period'])
                        
                        if signal['comparison'] == "less_than":
                            signals = (rsi <= signal['rsi_threshold']).astype(int)
                        else:
                            signals = (rsi >= signal['rsi_threshold']).astype(int)
                        
                        equity_curve = calculate_equity_curve(signals, data[signal['target_ticker']], 
                                                           st.session_state.allocations.get(signal['name'], 1.0))
                        
                        returns = equity_curve.pct_change().dropna()
                        metrics = calculate_metrics(equity_curve, returns)
                        
                        results[signal['name']] = {
                            'equity_curve': equity_curve,
                            'signals': signals,
                            'metrics': metrics
                        }
                        
                        combined_equity = combined_equity * (1 + (equity_curve - 1) * st.session_state.allocations.get(signal['name'], 1.0))
                    
                    elif signal['type'] == "RSI Comparison":
                        signal_rsi = calculate_rsi(data[signal['signal_ticker']], signal['rsi_period'])
                        comparison_rsi = calculate_rsi(data[signal['comparison_ticker']], signal['rsi_period'])
                        
                        if signal['comparison_operator'] == "less_than":
                            signals = (signal_rsi < comparison_rsi).astype(int)
                        else:
                            signals = (signal_rsi > comparison_rsi).astype(int)
                        
                        equity_curve = calculate_equity_curve(signals, data[signal['target_ticker']], 
                                                           st.session_state.allocations.get(signal['name'], 1.0))
                        
                        returns = equity_curve.pct_change().dropna()
                        metrics = calculate_metrics(equity_curve, returns)
                        
                        results[signal['name']] = {
                            'equity_curve': equity_curve,
                            'signals': signals,
                            'metrics': metrics
                        }
                        
                        combined_equity = combined_equity * (1 + (equity_curve - 1) * st.session_state.allocations.get(signal['name'], 1.0))
                
                # Calculate benchmark
                benchmark_equity = data[benchmark_ticker] / data[benchmark_ticker].iloc[0]
                benchmark_returns = benchmark_equity.pct_change().dropna()
                benchmark_metrics = calculate_metrics(benchmark_equity, benchmark_returns)
                
                # Store results
                st.session_state.backtest_results = {
                    'signals': results,
                    'combined_equity': combined_equity,
                    'benchmark_equity': benchmark_equity,
                    'benchmark_metrics': benchmark_metrics,
                    'data': data
                }
                
                st.success("Backtest completed successfully!")

with col2:
    st.header("📊 Quick Stats")
    if st.session_state.backtest_results:
        results = st.session_state.backtest_results
        
        # Combined portfolio metrics
        combined_returns = results['combined_equity'].pct_change().dropna()
        combined_metrics = calculate_metrics(results['combined_equity'], combined_returns)
        
        st.metric("Portfolio Return", f"{combined_metrics['total_return']:.2f}%")
        st.metric("Benchmark Return", f"{results['benchmark_metrics']['total_return']:.2f}%")
        st.metric("Excess Return", f"{combined_metrics['total_return'] - results['benchmark_metrics']['total_return']:.2f}%")
        st.metric("Sharpe Ratio", f"{combined_metrics['sharpe_ratio']:.2f}")
        st.metric("Max Drawdown", f"{combined_metrics['max_drawdown']:.2f}%")

# Signal display and allocation management
if st.session_state.signals:
    st.header("🎯 Signal Allocation")
    
    # Display signals with allocation controls
    for signal in st.session_state.signals:
        with st.container():
            st.markdown(f"""
            <div class="signal-card">
                <div class="signal-header">
                    <h3 class="signal-name">{signal['name']}</h3>
                    <span class="signal-type">{signal['type']}</span>
                </div>
                <div class="allocation-container">
                    <div class="allocation-header">
                        <span>Allocation</span>
                        <span class="allocation-value">{st.session_state.allocations.get(signal['name'], 0):.1f}%</span>
                    </div>
                    <st-slider
                        min="0"
                        max="100"
                        value="{st.session_state.allocations.get(signal['name'], 0)}"
                        step="1"
                        key="alloc_{signal['name']}"
                        on_change="update_allocation"
                    />
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Allocation slider
            allocation = st.slider(
                f"{signal['name']} Allocation (%)", 
                min_value=0, max_value=100, value=st.session_state.allocations.get(signal['name'], 20), 
                key=f"alloc_{signal['name']}"
            )
            st.session_state.allocations[signal['name']] = allocation / 100
    
    # Total allocation summary
    total_allocation = sum(st.session_state.allocations.values()) * 100
    if total_allocation > 100:
        st.error(f"⚠️ Total allocation exceeds 100% ({total_allocation:.1f}%)")
    elif total_allocation < 100:
        st.warning(f"ℹ️ Total allocation: {total_allocation:.1f}% ({(100-total_allocation):.1f}% in cash)")
    else:
        st.success(f"✅ Total allocation: {total_allocation:.1f}%")

# Results display
if st.session_state.backtest_results:
    st.header("📈 Backtest Results")
    
    # Portfolio overview
    with st.container():
        st.markdown('<div class="custom-container">', unsafe_allow_html=True)
        st.subheader("📊 Portfolio Overview")
        
        results = st.session_state.backtest_results
        combined_returns = results['combined_equity'].pct_change().dropna()
        combined_metrics = calculate_metrics(results['combined_equity'], combined_returns)
        
        # Create metric grid
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value positive">{combined_metrics['total_return']:.2f}%</div>
                <div class="metric-label">Total Return</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value positive">{combined_metrics['annualized_return']:.2f}%</div>
                <div class="metric-label">Annualized Return</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value neutral">{combined_metrics['sharpe_ratio']:.2f}</div>
                <div class="metric-label">Sharpe Ratio</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value negative">{combined_metrics['max_drawdown']:.2f}%</div>
                <div class="metric-label">Max Drawdown</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Equity curve comparison
    with st.container():
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("📈 Performance Chart")
        
        fig = go.Figure()
        
        # Add benchmark
        fig.add_trace(go.Scatter(
            x=st.session_state.backtest_results['benchmark_equity'].index,
            y=st.session_state.backtest_results['benchmark_equity'].values,
            mode='lines',
            name='Benchmark',
            line=dict(color='#d32f2f', width=2, dash='dash')
        ))
        
        # Add combined portfolio
        fig.add_trace(go.Scatter(
            x=st.session_state.backtest_results['combined_equity'].index,
            y=st.session_state.backtest_results['combined_equity'].values,
            mode='lines',
            name='Portfolio',
            line=dict(color='#1976d2', width=3)
        ))
        
        # Add individual signals
        colors = ['#2e7d32', '#7b1fa2', '#f57c00', '#5d4037', '#c2185b', '#424242', '#388e3c']
        for i, (signal_name, result) in enumerate(st.session_state.backtest_results['signals'].items()):
            fig.add_trace(go.Scatter(
                x=result['equity_curve'].index,
                y=result['equity_curve'].values,
                mode='lines',
                name=f"{signal_name}",
                line=dict(color=colors[i % len(colors)], width=1),
                visible='legendonly'
            ))
        
        fig.update_layout(
            title="Portfolio Performance vs Benchmark",
            xaxis_title="Date",
            yaxis_title="Equity Value",
            hovermode='x unified',
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(size=12)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Signal performance table
    with st.container():
        st.markdown('<div class="custom-container">', unsafe_allow_html=True)
        st.subheader("📊 Signal Performance")
        
        # Create performance table
        performance_data = []
        for signal_name, result in st.session_state.backtest_results['signals'].items():
            metrics = result['metrics']
            performance_data.append({
                'Signal': signal_name,
                'Total Return (%)': f"{metrics['total_return']:.2f}",
                'Annualized Return (%)': f"{metrics['annualized_return']:.2f}",
                'Sharpe Ratio': f"{metrics['sharpe_ratio']:.2f}",
                'Sortino Ratio': f"{metrics['sortino_ratio']:.2f}",
                'Max Drawdown (%)': f"{metrics['max_drawdown']:.2f}",
                'Win Rate (%)': f"{metrics['win_rate']:.1f}",
                'Total Trades': metrics['total_trades'],
                'Allocation (%)': f"{st.session_state.allocations[signal_name] * 100:.1f}"
            })
        
        # Add benchmark row
        benchmark_metrics = st.session_state.backtest_results['benchmark_metrics']
        performance_data.append({
            'Signal': 'Benchmark',
            'Total Return (%)': f"{benchmark_metrics['total_return']:.2f}",
            'Annualized Return (%)': f"{benchmark_metrics['annualized_return']:.2f}",
            'Sharpe Ratio': f"{benchmark_metrics['sharpe_ratio']:.2f}",
            'Sortino Ratio': f"{benchmark_metrics['sortino_ratio']:.2f}",
            'Max Drawdown (%)': f"{benchmark_metrics['max_drawdown']:.2f}",
            'Win Rate (%)': f"{benchmark_metrics['win_rate']:.1f}",
            'Total Trades': benchmark_metrics['total_trades'],
            'Allocation (%)': '100.0'
        })
        
        df = pd.DataFrame(performance_data)
        st.dataframe(df, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Individual signal analysis
    if len(st.session_state.backtest_results['signals']) > 0:
        with st.container():
            st.markdown('<div class="custom-container">', unsafe_allow_html=True)
            st.subheader("🔍 Individual Signal Analysis")
            
            signal_names = list(st.session_state.backtest_results['signals'].keys())
            selected_signal = st.selectbox("Select Signal for Detailed Analysis", signal_names)
            
            if selected_signal:
                signal_result = st.session_state.backtest_results['signals'][selected_signal]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader(f"📈 {selected_signal} Performance")
                    
                    # Key metrics
                    metrics = signal_result['metrics']
                    col1a, col1b = st.columns(2)
                    
                    with col1a:
                        st.metric("Total Return", f"{metrics['total_return']:.2f}%", 
                                 delta=f"{metrics['total_return'] - benchmark_metrics['total_return']:.2f}%")
                        st.metric("Annualized Return", f"{metrics['annualized_return']:.2f}%")
                        st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
                        st.metric("Max Drawdown", f"{metrics['max_drawdown']:.2f}%")
                    
                    with col1b:
                        st.metric("Win Rate", f"{metrics['win_rate']:.1f}%")
                        st.metric("Total Trades", metrics['total_trades'])
                        st.metric("Avg Trade Return", f"{metrics['avg_trade_return']:.2f}%")
                        st.metric("Volatility", f"{metrics['volatility']:.2f}%")
                
                with col2:
                    st.subheader("📊 Signal Distribution")
                    
                    # Signal frequency analysis
                    signals = signal_result['signals']
                    signal_frequency = signals.value_counts()
                    
                    fig_pie = go.Figure(data=[go.Pie(
                        labels=['No Signal', 'Signal'],
                        values=[signal_frequency.get(0, 0), signal_frequency.get(1, 0)],
                        hole=0.3,
                        marker_colors=['#e3f2fd', '#1976d2']
                    )])
                    fig_pie.update_layout(
                        title="Signal Distribution",
                        plot_bgcolor='white',
                        paper_bgcolor='white'
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                # Signal equity curve
                st.subheader("📈 Signal Equity Curve")
                
                fig_signal = go.Figure()
                
                # Add signal equity curve
                fig_signal.add_trace(go.Scatter(
                    x=signal_result['equity_curve'].index,
                    y=signal_result['equity_curve'].values,
                    mode='lines',
                    name=f"{selected_signal}",
                    line=dict(color='#1976d2', width=2)
                ))
                
                # Add benchmark
                fig_signal.add_trace(go.Scatter(
                    x=st.session_state.backtest_results['benchmark_equity'].index,
                    y=st.session_state.backtest_results['benchmark_equity'].values,
                    mode='lines',
                    name='Benchmark',
                    line=dict(color='#d32f2f', width=2, dash='dash')
                ))
                
                fig_signal.update_layout(
                    title=f"{selected_signal} vs Benchmark",
                    xaxis_title="Date",
                    yaxis_title="Equity Value",
                    hovermode='x unified',
                    plot_bgcolor='white',
                    paper_bgcolor='white'
                )
                
                st.plotly_chart(fig_signal, use_container_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.write("---")
st.markdown("""
<div style='text-align: center; padding: 20px; color: #666;'>
    <strong>Tactical Allocation Platform</strong><br>
    Professional signal analysis and portfolio optimization
</div>
""", unsafe_allow_html=True)
