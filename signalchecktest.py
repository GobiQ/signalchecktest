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
    page_title="Strategy Validation Tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    /* Main styling */
    .main {
        padding: 2rem;
        background: #f5f5f5;
        min-height: 100vh;
    }
    
    .stApp {
        background: #f5f5f5;
    }
    
    /* Header styling */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #333;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem 0;
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
        color: #333;
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
        color: #333;
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
        color: #333;
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
        background: linear-gradient(135deg, #1976d2 0%, #1565c0 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #1565c0 0%, #0d47a1 100%);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
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
        background-color: white;
        color: #333;
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background-color: #1976d2;
    }
    
    /* Dataframe styling */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
        background: white;
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
    
    /* Fix text colors */
    .stMarkdown {
        color: #333 !important;
    }
    
    .stDataFrame {
        color: #333 !important;
    }
    
    .stPlotlyChart {
        color: #333 !important;
    }
    
    /* Fix sidebar text colors */
    .sidebar .stMarkdown {
        color: #333 !important;
    }
    
    .sidebar .stSelectbox > div > div {
        background: white !important;
        color: #333 !important;
    }
    
    .sidebar .stTextInput > div > div > input {
        background: white !important;
        color: #333 !important;
    }
    
    .sidebar .stSlider > div > div > div > div {
        background: #1976d2 !important;
    }
    
    /* Fix main content text colors */
    .main .stMarkdown {
        color: #333 !important;
    }
    
    .main .stSubheader {
        color: #333 !important;
    }
    
    .main .stHeader {
        color: #333 !important;
    }
    
    .main .stText {
        color: #333 !important;
    }
    
    .main .stCaption {
        color: #333 !important;
    }
    
    .main .stInfo {
        color: #333 !important;
    }
    
    .main .stSuccess {
        color: #333 !important;
    }
    
    .main .stWarning {
        color: #333 !important;
    }
    
    .main .stError {
        color: #333 !important;
    }
    
    /* Fix input field colors */
    .main .stTextInput > div > div > input {
        background: white !important;
        color: #333 !important;
        border: 1px solid #e1e5e9 !important;
    }
    
    .main .stSelectbox > div > div {
        background: white !important;
        color: #333 !important;
        border: 1px solid #e1e5e9 !important;
    }
    
    .main .stNumberInput > div > div > input {
        background: white !important;
        color: #333 !important;
        border: 1px solid #e1e5e9 !important;
    }
    
    .main .stDateInput > div > div > input {
        background: white !important;
        color: #333 !important;
        border: 1px solid #e1e5e9 !important;
    }
    
    /* Fix label colors */
    .main label {
        color: #333 !important;
    }
    
    .main .stLabel {
        color: #333 !important;
    }
    
    /* Fix metric colors */
    .main .stMetric {
        color: #333 !important;
    }
    
    .main .stMetric > div > div {
        color: #333 !important;
    }
    
    /* Fix container backgrounds */
    .main .stContainer {
        background: white !important;
        color: #333 !important;
    }
    
    .main .stExpander {
        background: white !important;
        color: #333 !important;
    }
    
    /* Fix tab text colors */
    .stTabs [data-baseweb="tab-list"] {
        background: white !important;
        color: #333 !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: white !important;
        color: #333 !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: #f5f5f5 !important;
        color: #333 !important;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: #1976d2 !important;
        color: white !important;
    }
    
    /* Fix all text elements */
    * {
        color: #333 !important;
    }
    
    /* Override for specific elements that should be white */
    .stButton > button {
        color: white !important;
    }
    
    .signal-type {
        color: #1976d2 !important;
    }
    
    .allocation-value {
        color: #1976d2 !important;
    }
    
    .positive {
        color: #2e7d32 !important;
    }
    
    .negative {
        color: #d32f2f !important;
    }
    
    .neutral {
        color: #666 !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'signals' not in st.session_state:
    st.session_state.signals = []
if 'allocations' not in st.session_state:
    st.session_state.allocations = {}
if 'output_allocations' not in st.session_state:
    st.session_state.output_allocations = {}
if 'strategies' not in st.session_state:
    st.session_state.strategies = []
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

def calculate_multi_ticker_equity_curve(signals: pd.Series, allocation: dict, data: dict) -> pd.Series:
    """Calculate equity curve for an allocation with multiple tickers"""
    # Calculate weighted equity curve for multiple tickers
    weighted_equity_curves = []
    for ticker_component in allocation['tickers']:
        ticker = ticker_component['ticker']
        weight = ticker_component['weight'] / 100
        
        # Calculate equity curve for this ticker
        ticker_equity = calculate_equity_curve(signals, data[ticker], weight)
        weighted_equity_curves.append(ticker_equity)
    
    # Combine weighted equity curves
    equity_curve = pd.Series(1.0, index=data[list(data.keys())[0]].index)
    for ticker_equity in weighted_equity_curves:
        equity_curve = equity_curve * ticker_equity
    
    return equity_curve

# Main app
st.markdown('<h1 class="main-header">📊 Strategy Validation Tool</h1>', unsafe_allow_html=True)

# Create tabs for different sections
tab1, tab2, tab3, tab4 = st.tabs(["💰 Allocations", "📊 Signals", "🎯 Strategies", "📈 Backtest"])

# Tab 1: Allocations
with tab1:
    st.header("💰 Output Allocations")
    
    # Create allocation
    with st.expander("➕ Create Output Allocation", expanded=True):
        allocation_name = st.text_input("Allocation Name", placeholder="e.g., Aggressive Growth")
        
        st.subheader("📊 Ticker Components")
        
        # Initialize ticker components in session state
        if 'current_allocation_tickers' not in st.session_state:
            st.session_state.current_allocation_tickers = []
        
        # Add ticker component
        col1, col2 = st.columns(2)
        with col1:
            new_ticker = st.text_input("Ticker", placeholder="e.g., SPY", key="new_ticker")
        with col2:
            ticker_weight = st.number_input("Weight (%)", min_value=0, max_value=100, value=50, key="ticker_weight")
        
        if st.button("➕ Add Ticker", key="add_ticker"):
            if new_ticker and ticker_weight > 0:
                ticker_component = {
                    'ticker': new_ticker.upper(),
                    'weight': ticker_weight
                }
                st.session_state.current_allocation_tickers.append(ticker_component)
                st.rerun()
        
        # Display current ticker components
        if st.session_state.current_allocation_tickers:
            st.subheader("📋 Current Tickers")
            total_weight = sum([tc['weight'] for tc in st.session_state.current_allocation_tickers])
            
            for i, ticker_component in enumerate(st.session_state.current_allocation_tickers):
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.write(f"**{ticker_component['ticker']}**")
                with col2:
                    st.write(f"{ticker_component['weight']}%")
                with col3:
                    if st.button("🗑️", key=f"remove_ticker_{i}"):
                        st.session_state.current_allocation_tickers.pop(i)
                        st.rerun()
            
            st.write(f"**Total Weight: {total_weight}%**")
            
            if total_weight != 100:
                if total_weight > 100:
                    st.error(f"⚠️ Total weight exceeds 100% ({total_weight}%)")
                else:
                    st.warning(f"ℹ️ Total weight: {total_weight}% ({(100-total_weight):.1f}% unallocated)")
            else:
                st.success(f"✅ Total weight: {total_weight}%")
        
        # Create allocation button
        if st.button("Create Allocation", type="primary"):
            if allocation_name and st.session_state.current_allocation_tickers:
                total_weight = sum([tc['weight'] for tc in st.session_state.current_allocation_tickers])
                
                if total_weight == 100:
                    allocation = {
                        'name': allocation_name,
                        'tickers': st.session_state.current_allocation_tickers.copy(),
                        'total_weight': total_weight
                    }
                    st.session_state.output_allocations[allocation_name] = allocation
                    st.session_state.current_allocation_tickers = []  # Reset for next allocation
                    st.success(f"Allocation '{allocation_name}' created successfully!")
                    st.rerun()
                else:
                    st.error("Total weight must equal 100% to create allocation.")
            else:
                st.error("Please provide an allocation name and at least one ticker component.")
    
    # Display existing allocations
    if st.session_state.output_allocations:
        st.subheader("📋 Active Allocations")
        for name, allocation in st.session_state.output_allocations.items():
            with st.container():
                st.markdown(f"""
                <div class="signal-card">
                    <div class="signal-header">
                        <h3 class="signal-name">{name}</h3>
                        <span class="signal-type">Allocation</span>
                    </div>
                    <div class="allocation-container">
                        <div class="allocation-header">
                            <span>Components</span>
                        </div>
                        <div style="margin-top: 0.5rem;">
                """, unsafe_allow_html=True)
                
                for ticker_component in allocation['tickers']:
                    st.write(f"• **{ticker_component['ticker']}**: {ticker_component['weight']}%")
                
                st.markdown("""
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("🗑️", key=f"delete_allocation_{name}"):
                    del st.session_state.output_allocations[name]
                    st.rerun()
    else:
        st.info("No allocations created yet. Create your first allocation above.")

# Tab 2: Signals
with tab2:
    st.header("📊 Signal Management")
    
    # Signal creation
    with st.expander("➕ Create Signal", expanded=True):
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
    
    # Display existing signals
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
                    if st.button("🗑️", key=f"delete_signal_{i}"):
                        st.session_state.signals.pop(i)
                        st.rerun()
    else:
        st.info("No signals created yet. Create your first signal above.")

# Tab 3: Strategies
with tab3:
    st.header("🎯 Strategy Builder")
    
    # Strategy builder
    with st.expander("➕ Create Strategy", expanded=True):
        strategy_name = st.text_input("Strategy Name", placeholder="e.g., Multi-Signal Strategy")
        
        if st.session_state.signals and st.session_state.output_allocations:
            st.subheader("📋 Conditional Logic")
            
            # First condition
            col1, col2, col3 = st.columns(3)
            with col1:
                signal1 = st.selectbox("Signal 1", [s['name'] for s in st.session_state.signals], key="signal1")
                signal1_negated = st.checkbox("NOT", key="signal1_not")
            with col2:
                logic_operator = st.selectbox("Logic", ["AND", "OR"], key="logic_op")
            with col3:
                signal2 = st.selectbox("Signal 2", [s['name'] for s in st.session_state.signals], key="signal2")
                signal2_negated = st.checkbox("NOT", key="signal2_not")
            
            # Output allocation
            output_allocation = st.selectbox("Then Allocate To", list(st.session_state.output_allocations.keys()))
            
            if st.button("Add Strategy", type="primary"):
                strategy = {
                    'name': strategy_name,
                    'signal1': signal1,
                    'signal1_negated': signal1_negated,
                    'logic_operator': logic_operator,
                    'signal2': signal2,
                    'signal2_negated': signal2_negated,
                    'output_allocation': output_allocation
                }
                st.session_state.strategies.append(strategy)
                st.success(f"Strategy '{strategy_name}' added!")
                st.rerun()
        else:
            st.warning("Please create signals and allocations first.")
    
    # Display existing strategies
    if st.session_state.strategies:
        st.subheader("📋 Active Strategies")
        for i, strategy in enumerate(st.session_state.strategies):
            with st.container():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**{strategy['name']}**")
                    signal1_text = f"NOT {strategy['signal1']}" if strategy['signal1_negated'] else strategy['signal1']
                    signal2_text = f"NOT {strategy['signal2']}" if strategy['signal2_negated'] else strategy['signal2']
                    st.caption(f"IF {signal1_text} {strategy['logic_operator']} {signal2_text} THEN {strategy['output_allocation']}")
                with col2:
                    if st.button("🗑️", key=f"delete_strategy_{i}"):
                        st.session_state.strategies.pop(i)
                        st.rerun()
    else:
        st.info("No strategies created yet. Create your first strategy above.")

# Tab 4: Backtest
with tab4:
    st.header("📈 Backtest Configuration")
    
    # Backtest configuration
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=365))
        benchmark_ticker = st.text_input("Benchmark Ticker", value="SPY")
    with col2:
        end_date = st.date_input("End Date", value=datetime.now())
    
    # Quick stats
    if st.session_state.output_allocations:
        st.subheader("📊 Quick Stats")
        total_allocation = sum([alloc['total_weight'] for alloc in st.session_state.output_allocations.values()])
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Allocations", f"{total_allocation}%")
        with col2:
            st.metric("Active Signals", len(st.session_state.signals))
        with col3:
            st.metric("Active Strategies", len(st.session_state.strategies))
        
        if total_allocation > 100:
            st.error(f"⚠️ Total allocation exceeds 100% ({total_allocation:.1f}%)")
        elif total_allocation < 100:
            st.warning(f"ℹ️ Total allocation: {total_allocation:.1f}% ({(100-total_allocation):.1f}% in cash)")
        else:
            st.success(f"✅ Total allocation: {total_allocation:.1f}%")
    
    # Strategy overview
    if st.session_state.strategies:
        st.subheader("🎯 Strategy Overview")
        
        # Display strategies
        for strategy in st.session_state.strategies:
            with st.container():
                st.markdown(f"""
                <div class="signal-card">
                    <div class="signal-header">
                        <h3 class="signal-name">{strategy['name']}</h3>
                        <span class="signal-type">Strategy</span>
                    </div>
                    <div class="allocation-container">
                        <div class="allocation-header">
                            <span>Condition</span>
                        </div>
                        <p>
                            IF {f"NOT {strategy['signal1']}" if strategy['signal1_negated'] else strategy['signal1']} 
                            {strategy['logic_operator']} 
                            {f"NOT {strategy['signal2']}" if strategy['signal2_negated'] else strategy['signal2']} 
                            THEN {strategy['output_allocation']}
                        </p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    # Run backtest button
    if st.button("🚀 Run Backtest", type="primary", use_container_width=True):
        if not st.session_state.signals:
            st.error("Please add at least one signal before running backtest.")
        elif not st.session_state.strategies:
            st.error("Please create at least one strategy before running backtest.")
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
                
                # Add allocation tickers
                for allocation in st.session_state.output_allocations.values():
                    all_tickers.add(allocation['ticker'])
                
                all_tickers.add(benchmark_ticker)
                
                # Fetch data
                data = {}
                for ticker in all_tickers:
                    data[ticker] = get_stock_data(ticker, start_date, end_date)
                
                # Calculate individual signals
                signal_results = {}
                for signal in st.session_state.signals:
                    if signal['type'] == "Custom Indicator":
                        target_ticker = signal['target_ticker']
                        
                        indicator1_values = calculate_indicator(data[target_ticker], signal['indicator1'], signal['days1'])
                        indicator2_values = calculate_indicator(data[target_ticker], signal['indicator2'], signal['days2'])
                        
                        signals = evaluate_signal_condition(indicator1_values, indicator2_values, signal['operator'])
                        
                        signal_results[signal['name']] = signals
                    
                    elif signal['type'] == "RSI Threshold":
                        rsi = calculate_rsi(data[signal['signal_ticker']], signal['rsi_period'])
                        
                        if signal['comparison'] == "less_than":
                            signals = (rsi <= signal['rsi_threshold']).astype(int)
                        else:
                            signals = (rsi >= signal['rsi_threshold']).astype(int)
                        
                        signal_results[signal['name']] = signals
                    
                    elif signal['type'] == "RSI Comparison":
                        signal_rsi = calculate_rsi(data[signal['signal_ticker']], signal['rsi_period'])
                        comparison_rsi = calculate_rsi(data[signal['comparison_ticker']], signal['rsi_period'])
                        
                        if signal['comparison_operator'] == "less_than":
                            signals = (signal_rsi < comparison_rsi).astype(int)
                        else:
                            signals = (signal_rsi > comparison_rsi).astype(int)
                        
                        signal_results[signal['name']] = signals
                
                # Calculate strategy allocations
                strategy_results = {}
                combined_equity = pd.Series(1.0, index=data[benchmark_ticker].index)
                
                for strategy in st.session_state.strategies:
                    # Get signal values
                    signal1_values = signal_results[strategy['signal1']]
                    signal2_values = signal_results[strategy['signal2']]
                    
                    # Apply negation if needed
                    if strategy['signal1_negated']:
                        signal1_values = (~signal1_values.astype(bool)).astype(int)
                    if strategy['signal2_negated']:
                        signal2_values = (~signal2_values.astype(bool)).astype(int)
                    
                    # Apply logic operator
                    if strategy['logic_operator'] == "AND":
                        strategy_signals = (signal1_values & signal2_values).astype(int)
                    else:  # OR
                        strategy_signals = (signal1_values | signal2_values).astype(int)
                    
                    # Get allocation details
                    allocation = st.session_state.output_allocations[strategy['output_allocation']]
                    allocation_ticker = allocation['ticker']
                    allocation_percentage = allocation['percentage'] / 100
                    
                    # Calculate equity curve for this strategy
                    equity_curve = calculate_equity_curve(strategy_signals, data[allocation_ticker], allocation_percentage)
                    
                    # Calculate metrics
                    returns = equity_curve.pct_change().dropna()
                    metrics = calculate_metrics(equity_curve, returns)
                    
                    strategy_results[strategy['name']] = {
                        'equity_curve': equity_curve,
                        'signals': strategy_signals,
                        'metrics': metrics,
                        'allocation': allocation
                    }
                    
                    # Add to combined portfolio
                    combined_equity = combined_equity * (1 + (equity_curve - 1))
                
                # Calculate benchmark
                benchmark_equity = data[benchmark_ticker] / data[benchmark_ticker].iloc[0]
                benchmark_returns = benchmark_equity.pct_change().dropna()
                benchmark_metrics = calculate_metrics(benchmark_equity, benchmark_returns)
                
                # Store results
                st.session_state.backtest_results = {
                    'signals': signal_results,
                    'strategies': strategy_results,
                    'combined_equity': combined_equity,
                    'benchmark_equity': benchmark_equity,
                    'benchmark_metrics': benchmark_metrics,
                    'data': data
                }
                
                st.success("Backtest completed successfully!")

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
        elif not st.session_state.strategies:
            st.error("Please create at least one strategy before running backtest.")
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
                
                # Add allocation tickers
                for allocation in st.session_state.output_allocations.values():
                    all_tickers.add(allocation['ticker'])
                
                all_tickers.add(benchmark_ticker)
                
                # Fetch data
                data = {}
                for ticker in all_tickers:
                    data[ticker] = get_stock_data(ticker, start_date, end_date)
                
                # Calculate individual signals
                signal_results = {}
                for signal in st.session_state.signals:
                    if signal['type'] == "Custom Indicator":
                        target_ticker = signal['target_ticker']
                        
                        indicator1_values = calculate_indicator(data[target_ticker], signal['indicator1'], signal['days1'])
                        indicator2_values = calculate_indicator(data[target_ticker], signal['indicator2'], signal['days2'])
                        
                        signals = evaluate_signal_condition(indicator1_values, indicator2_values, signal['operator'])
                        
                        signal_results[signal['name']] = signals
                    
                    elif signal['type'] == "RSI Threshold":
                        rsi = calculate_rsi(data[signal['signal_ticker']], signal['rsi_period'])
                        
                        if signal['comparison'] == "less_than":
                            signals = (rsi <= signal['rsi_threshold']).astype(int)
                        else:
                            signals = (rsi >= signal['rsi_threshold']).astype(int)
                        
                        signal_results[signal['name']] = signals
                    
                    elif signal['type'] == "RSI Comparison":
                        signal_rsi = calculate_rsi(data[signal['signal_ticker']], signal['rsi_period'])
                        comparison_rsi = calculate_rsi(data[signal['comparison_ticker']], signal['rsi_period'])
                        
                        if signal['comparison_operator'] == "less_than":
                            signals = (signal_rsi < comparison_rsi).astype(int)
                        else:
                            signals = (signal_rsi > comparison_rsi).astype(int)
                        
                        signal_results[signal['name']] = signals
                
                # Calculate strategy allocations
                strategy_results = {}
                combined_equity = pd.Series(1.0, index=data[benchmark_ticker].index)
                
                for strategy in st.session_state.strategies:
                    # Get signal values
                    signal1_values = signal_results[strategy['signal1']]
                    signal2_values = signal_results[strategy['signal2']]
                    
                    # Apply negation if needed
                    if strategy['signal1_negated']:
                        signal1_values = (~signal1_values.astype(bool)).astype(int)
                    if strategy['signal2_negated']:
                        signal2_values = (~signal2_values.astype(bool)).astype(int)
                    
                    # Apply logic operator
                    if strategy['logic_operator'] == "AND":
                        strategy_signals = (signal1_values & signal2_values).astype(int)
                    else:  # OR
                        strategy_signals = (signal1_values | signal2_values).astype(int)
                    
                    # Get allocation details
                    allocation = st.session_state.output_allocations[strategy['output_allocation']]
                    allocation_ticker = allocation['ticker']
                    allocation_percentage = allocation['percentage'] / 100
                    
                    # Calculate equity curve for this strategy
                    equity_curve = calculate_equity_curve(strategy_signals, data[allocation_ticker], allocation_percentage)
                    
                    # Calculate metrics
                    returns = equity_curve.pct_change().dropna()
                    metrics = calculate_metrics(equity_curve, returns)
                    
                    strategy_results[strategy['name']] = {
                        'equity_curve': equity_curve,
                        'signals': strategy_signals,
                        'metrics': metrics,
                        'allocation': allocation
                    }
                    
                    # Add to combined portfolio
                    combined_equity = combined_equity * (1 + (equity_curve - 1))
                
                # Calculate benchmark
                benchmark_equity = data[benchmark_ticker] / data[benchmark_ticker].iloc[0]
                benchmark_returns = benchmark_equity.pct_change().dropna()
                benchmark_metrics = calculate_metrics(benchmark_equity, benchmark_returns)
                
                # Store results
                st.session_state.backtest_results = {
                    'signals': signal_results,
                    'strategies': strategy_results,
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

# Strategy overview
if st.session_state.strategies:
    st.header("🎯 Strategy Overview")
    
    # Display strategies
    for strategy in st.session_state.strategies:
        with st.container():
            st.markdown(f"""
            <div class="signal-card">
                <div class="signal-header">
                    <h3 class="signal-name">{strategy['name']}</h3>
                    <span class="signal-type">Strategy</span>
                </div>
                <div class="allocation-container">
                    <div class="allocation-header">
                        <span>Condition</span>
                    </div>
                    <p>
                        IF {f"NOT {strategy['signal1']}" if strategy['signal1_negated'] else strategy['signal1']} 
                        {strategy['logic_operator']} 
                        {f"NOT {strategy['signal2']}" if strategy['signal2_negated'] else strategy['signal2']} 
                        THEN {strategy['output_allocation']}
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Total allocation summary
    total_allocation = sum([alloc['percentage'] for alloc in st.session_state.output_allocations.values()])
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
    
        # Strategy performance table
    with st.container():
        st.markdown('<div class="custom-container">', unsafe_allow_html=True)
        st.subheader("📊 Strategy Performance")
        
        # Create performance table
        performance_data = []
        for strategy_name, result in st.session_state.backtest_results['strategies'].items():
            metrics = result['metrics']
            allocation = result['allocation']
            performance_data.append({
                'Strategy': strategy_name,
                'Total Return (%)': f"{metrics['total_return']:.2f}",
                'Annualized Return (%)': f"{metrics['annualized_return']:.2f}",
                'Sharpe Ratio': f"{metrics['sharpe_ratio']:.2f}",
                'Sortino Ratio': f"{metrics['sortino_ratio']:.2f}",
                'Max Drawdown (%)': f"{metrics['max_drawdown']:.2f}",
                'Win Rate (%)': f"{metrics['win_rate']:.1f}",
                'Total Trades': metrics['total_trades'],
                'Allocation': ', '.join([f"{tc['ticker']}({tc['weight']}%)" for tc in allocation['tickers']])
            })
        
        # Add benchmark row
        benchmark_metrics = st.session_state.backtest_results['benchmark_metrics']
        performance_data.append({
            'Strategy': 'Benchmark',
            'Total Return (%)': f"{benchmark_metrics['total_return']:.2f}",
            'Annualized Return (%)': f"{benchmark_metrics['annualized_return']:.2f}",
            'Sharpe Ratio': f"{benchmark_metrics['sharpe_ratio']:.2f}",
            'Sortino Ratio': f"{benchmark_metrics['sortino_ratio']:.2f}",
            'Max Drawdown (%)': f"{benchmark_metrics['max_drawdown']:.2f}",
            'Win Rate (%)': f"{benchmark_metrics['win_rate']:.1f}",
            'Total Trades': benchmark_metrics['total_trades'],
            'Allocation': '100% → SPY'
        })
        
        df = pd.DataFrame(performance_data)
        st.dataframe(df, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Individual strategy analysis
    if len(st.session_state.backtest_results['strategies']) > 0:
        with st.container():
            st.markdown('<div class="custom-container">', unsafe_allow_html=True)
            st.subheader("🔍 Individual Strategy Analysis")
            
            strategy_names = list(st.session_state.backtest_results['strategies'].keys())
            selected_strategy = st.selectbox("Select Strategy for Detailed Analysis", strategy_names)
            
            if selected_strategy:
                strategy_result = st.session_state.backtest_results['strategies'][selected_strategy]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader(f"📈 {selected_strategy} Performance")
                    
                    # Key metrics
                    metrics = strategy_result['metrics']
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
                    st.subheader("📊 Strategy Distribution")
                    
                    # Signal frequency analysis
                    signals = strategy_result['signals']
                    signal_frequency = signals.value_counts()
                    
                    fig_pie = go.Figure(data=[go.Pie(
                        labels=['No Signal', 'Signal'],
                        values=[signal_frequency.get(0, 0), signal_frequency.get(1, 0)],
                        hole=0.3,
                        marker_colors=['#e3f2fd', '#1976d2']
                    )])
                    fig_pie.update_layout(
                        title="Strategy Signal Distribution",
                        plot_bgcolor='white',
                        paper_bgcolor='white'
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                # Strategy equity curve
                st.subheader("📈 Strategy Equity Curve")
                
                fig_strategy = go.Figure()
                
                # Add strategy equity curve
                fig_strategy.add_trace(go.Scatter(
                    x=strategy_result['equity_curve'].index,
                    y=strategy_result['equity_curve'].values,
                    mode='lines',
                    name=f"{selected_strategy}",
                    line=dict(color='#1976d2', width=2)
                ))
                
                # Add benchmark
                fig_strategy.add_trace(go.Scatter(
                    x=st.session_state.backtest_results['benchmark_equity'].index,
                    y=st.session_state.backtest_results['benchmark_equity'].values,
                    mode='lines',
                    name='Benchmark',
                    line=dict(color='#d32f2f', width=2, dash='dash')
                ))
                
                fig_strategy.update_layout(
                    title=f"{selected_strategy} vs Benchmark",
                    xaxis_title="Date",
                    yaxis_title="Equity Value",
                    hovermode='x unified',
                    plot_bgcolor='white',
                    paper_bgcolor='white'
                )
                
                st.plotly_chart(fig_strategy, use_container_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.write("---")
st.markdown("""
<div style='text-align: center; padding: 20px; color: #666;'>
    <strong>Strategy Testing Tool</strong><br>
    Professional signal analysis and portfolio optimization
</div>
""", unsafe_allow_html=True)
