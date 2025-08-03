import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import warnings
import json
import copy
import hashlib
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Strategy Validation Tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for modern slate grey theme
st.markdown("""
<style>
    .main {
        background: #475569;
        color: white;
    }
    .stApp {
        background: #475569;
    }
    .stExpander {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 10px;
        backdrop-filter: blur(10px);
    }
    .strategy-builder {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        backdrop-filter: blur(10px);
    }
    .branch-container {
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        backdrop-filter: blur(8px);
    }
    .condition-block {
        background: rgba(59, 130, 246, 0.2);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 6px;
        padding: 10px;
        margin: 5px 0;
        position: relative;
    }
    .condition-block::before {
        content: "IF";
        position: absolute;
        top: -8px;
        left: 10px;
        background: #3b82f6;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: bold;
    }
    .then-block {
        background: rgba(34, 197, 94, 0.2);
        border: 1px solid rgba(34, 197, 94, 0.3);
        border-radius: 6px;
        padding: 10px;
        margin: 5px 0;
        position: relative;
    }
    .then-block::before {
        content: "THEN";
        position: absolute;
        top: -8px;
        left: 10px;
        background: #22c55e;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: bold;
    }
    .else-block {
        background: rgba(239, 68, 68, 0.2);
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-radius: 6px;
        padding: 10px;
        margin: 5px 0;
        position: relative;
    }
    .else-block::before {
        content: "ELSE";
        position: absolute;
        top: -8px;
        left: 10px;
        background: #ef4444;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: bold;
    }
    .nested-else-block {
        background: rgba(168, 85, 247, 0.2);
        border: 1px solid rgba(168, 85, 247, 0.3);
        border-radius: 6px;
        padding: 10px;
        margin: 5px 0;
        position: relative;
    }
    .nested-else-block::before {
        content: "ELSE IF";
        position: absolute;
        top: -8px;
        left: 10px;
        background: #a855f7;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: bold;
    }
    .allocation-display {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 6px;
        padding: 8px;
        margin: 5px 0;
    }
    .if-else-block {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        backdrop-filter: blur(8px);
    }
    .if-else-header {
        background: rgba(59, 130, 246, 0.3);
        border-radius: 6px;
        padding: 8px;
        margin-bottom: 10px;
        font-weight: bold;
    }
    .stButton > button {
        background: rgba(59, 130, 246, 0.8);
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 6px;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.3);
    }
    .stButton > button:hover {
        background: rgba(59, 130, 246, 1);
        border-color: rgba(255, 255, 255, 0.4);
    }
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: white;
    }
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: white;
    }
    .stNumberInput > div > div > input {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'signals' not in st.session_state:
    st.session_state.signals = []
    # Add pre-built signals automatically
    prebuilt_signals = [
        {
            'name': 'If 10d RSI QQQ > 80',
            'type': 'Custom Indicator',
            'signal_ticker1': 'QQQ',
            'signal_ticker2': 'QQQ',
            'indicator1': 'RSI',
            'indicator2': 'Static Value',
            'operator': '>',
            'days1': 10,
            'days2': None,
            'static_value': 80.0
        },
        {
            'name': 'If 10d RSI QQQ < 30',
            'type': 'Custom Indicator',
            'signal_ticker1': 'QQQ',
            'signal_ticker2': 'QQQ',
            'indicator1': 'RSI',
            'indicator2': 'Static Value',
            'operator': '<',
            'days1': 10,
            'days2': None,
            'static_value': 30.0
        },
        {
            'name': 'If Current Price SPY > 200d SMA SPY',
            'type': 'Custom Indicator',
            'signal_ticker1': 'SPY',
            'signal_ticker2': 'SPY',
            'indicator1': 'Current Price',
            'indicator2': 'SMA',
            'operator': '>',
            'days1': None,
            'days2': 200,
            'static_value': None
        },
        {
            'name': 'If Current Price SPY > 20d SMA SPY',
            'type': 'Custom Indicator',
            'signal_ticker1': 'SPY',
            'signal_ticker2': 'SPY',
            'indicator1': 'Current Price',
            'indicator2': 'SMA',
            'operator': '>',
            'days1': None,
            'days2': 20,
            'static_value': None
        }
    ]
    st.session_state.signals = prebuilt_signals

if 'output_allocations' not in st.session_state:
    st.session_state.output_allocations = {}
    # Add pre-built allocations automatically
    prebuilt_allocations = {
        'TQQQ': {
            'name': 'TQQQ',
            'tickers': [{'ticker': 'TQQQ', 'weight': 100}]
        },
        'QLD': {
            'name': 'QLD',
            'tickers': [{'ticker': 'QLD', 'weight': 100}]
        },
        'QQQ': {
            'name': 'QQQ',
            'tickers': [{'ticker': 'QQQ', 'weight': 100}]
        },
        'SPY': {
            'name': 'SPY',
            'tickers': [{'ticker': 'SPY', 'weight': 100}]
        },
        'XLP': {
            'name': 'XLP',
            'tickers': [{'ticker': 'XLP', 'weight': 100}]
        },
        'XLU': {
            'name': 'XLU',
            'tickers': [{'ticker': 'XLU', 'weight': 100}]
        },
        'BIL': {
            'name': 'BIL',
            'tickers': [{'ticker': 'BIL', 'weight': 100}]
        },
        'UVXY': {
            'name': 'UVXY',
            'tickers': [{'ticker': 'UVXY', 'weight': 100}]
        },
        'VIXY': {
            'name': 'VIXY',
            'tickers': [{'ticker': 'VIXY', 'weight': 100}]
        },
        'BIL VIXY Defensive': {
            'name': 'BIL VIXY Defensive',
            'tickers': [
                {'ticker': 'BIL', 'weight': 50},
                {'ticker': 'VIXY', 'weight': 50}
            ]
        },
        'XLP XLU Defensive': {
            'name': 'XLP XLU Defensive',
            'tickers': [
                {'ticker': 'XLP', 'weight': 50},
                {'ticker': 'XLU', 'weight': 50}
            ]
        }
    }
    st.session_state.output_allocations = prebuilt_allocations

if 'strategy_branches' not in st.session_state:
    st.session_state.strategy_branches = []

if 'copied_block' not in st.session_state:
    st.session_state.copied_block = None

# Reference Blocks System
if 'reference_blocks' not in st.session_state:
    st.session_state.reference_blocks = {}

if 'saved_blocks' not in st.session_state:
    st.session_state.saved_blocks = {}

# Logic Block Caching System
if 'block_cache' not in st.session_state:
    st.session_state.block_cache = {}

if 'computed_signals' not in st.session_state:
    st.session_state.computed_signals = {}

if 'cache_hits' not in st.session_state:
    st.session_state.cache_hits = 0

if 'cache_misses' not in st.session_state:
    st.session_state.cache_misses = 0

# Helper Functions
def generate_block_signature(block_data):
    """Generate a unique signature for a logic block"""
    signature_data = {
        'signals': block_data.get('signals', []),
        'allocations': block_data.get('allocations', []),
        'else_allocations': block_data.get('else_allocations', []),
        'type': block_data.get('type', 'custom')
    }
    return hashlib.md5(json.dumps(signature_data, sort_keys=True).encode()).hexdigest()

def compute_logic_block(block_data, signal_results, data, benchmark_ticker):
    """Compute logic block with caching"""
    signature = generate_block_signature(block_data)
    
    if signature in st.session_state.block_cache:
        st.session_state.cache_hits += 1
        return st.session_state.block_cache[signature]
    
    st.session_state.cache_misses += 1
    
    # Compute the logic block
    if block_data.get('type') == 'if_else':
        # Handle If/Else logic
        if_signals = block_data.get('signals', [])
        if_allocations = block_data.get('allocations', [])
        else_allocations = block_data.get('else_allocations', [])
        
        # Evaluate IF conditions
        if_result = True
        for signal_config in if_signals:
            signal_name = signal_config.get('signal', '')
            if signal_name and signal_name in signal_results:
                signal_result = signal_results[signal_name]
                if signal_config.get('negated', False):
                    signal_result = ~signal_result
                
                if signal_config.get('operator', 'AND') == 'AND':
                    if_result = if_result & signal_result
                else:
                    if_result = if_result | signal_result
        
        # Return appropriate allocation based on IF result
        if if_result.any():
            result = if_allocations
        else:
            result = else_allocations
        
        st.session_state.block_cache[signature] = result
        return result
    
    return []

def get_cache_stats():
    """Get cache statistics"""
    total_requests = st.session_state.cache_hits + st.session_state.cache_misses
    hit_rate = (st.session_state.cache_hits / total_requests * 100) if total_requests > 0 else 0
    return {
        'hits': st.session_state.cache_hits,
        'misses': st.session_state.cache_misses,
        'total': total_requests,
        'hit_rate': hit_rate
    }

def clear_cache():
    """Clear the block cache"""
    st.session_state.block_cache = {}
    st.session_state.cache_hits = 0
    st.session_state.cache_misses = 0

# Reference Block Management Functions
def save_reference_block(block_data, block_name, block_type="custom"):
    """Save a logic block as a reference block for reuse"""
    reference_block = {
        'name': block_name,
        'type': block_type,
        'data': copy.deepcopy(block_data),
        'created_at': datetime.now().isoformat(),
        'usage_count': 0
    }
    st.session_state.reference_blocks[block_name] = reference_block
    return block_name

def load_reference_block(block_name):
    """Load a reference block by name"""
    if block_name in st.session_state.reference_blocks:
        block = st.session_state.reference_blocks[block_name]
        block['usage_count'] += 1
        return copy.deepcopy(block['data'])
    return None

def get_reference_block_preview(block_name):
    """Get a preview of a reference block"""
    if block_name in st.session_state.reference_blocks:
        block = st.session_state.reference_blocks[block_name]
        signals_count = len(block['data'].get('signals', []))
        allocations_count = len(block['data'].get('allocations', []))
        usage_count = block['usage_count']
        return f"📋 {signals_count} signals, {allocations_count} allocations (used {usage_count} times)"
    return "Block not found"

def delete_reference_block(block_name):
    """Delete a reference block"""
    if block_name in st.session_state.reference_blocks:
        del st.session_state.reference_blocks[block_name]
        return True
    return False

def get_all_reference_blocks():
    """Get all available reference blocks"""
    return list(st.session_state.reference_blocks.keys())

# Technical Analysis Functions
def calculate_rsi(prices: pd.Series, window: int = 14, method: str = "wilders") -> pd.Series:
    """Calculate RSI with TradingView/Composer.trade compatibility"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    
    if method == "wilders":
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
    else:
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
        
        # Normalize timezone
        data.index = data.index.tz_localize(None)
        
        # Handle exclusions (holidays, etc.)
        if exclusions:
            data = data[~data.index.isin(exclusions)]
        
        return data['Close']
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return pd.Series(dtype=float)

def calculate_equity_curve(signals: pd.Series, prices: pd.Series, allocation: float = 1.0) -> pd.Series:
    """Calculate equity curve from signals and prices"""
    if signals.empty or prices.empty:
        return pd.Series(dtype=float)
    
    # Align signals and prices
    aligned_data = pd.concat([signals, prices], axis=1).dropna()
    if aligned_data.empty:
        return pd.Series(dtype=float)
    
    signals_aligned = aligned_data.iloc[:, 0]
    prices_aligned = aligned_data.iloc[:, 1]
    
    # Calculate returns
    returns = prices_aligned.pct_change()
    
    # Apply signals
    strategy_returns = returns * signals_aligned * allocation
    
    # Calculate cumulative equity curve
    equity_curve = (1 + strategy_returns).cumprod()
    
    return equity_curve

def calculate_metrics(equity_curve: pd.Series, returns: pd.Series) -> dict:
    """Calculate comprehensive performance metrics"""
    if equity_curve.empty:
        return {}
    
    # Basic metrics
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
    annualized_return = ((equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (252 / len(equity_curve))) - 1
    
    # Volatility
    strategy_returns = equity_curve.pct_change().dropna()
    volatility = strategy_returns.std() * np.sqrt(252)
    
    # Sharpe Ratio
    risk_free_rate = 0.02  # 2% annual risk-free rate
    sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0
    
    # Sortino Ratio
    downside_returns = strategy_returns[strategy_returns < 0]
    downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
    sortino_ratio = (annualized_return - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
    
    # Maximum Drawdown
    rolling_max = equity_curve.expanding().max()
    drawdown = (equity_curve - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    # Calmar Ratio
    calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    # Win Rate and Trade Statistics
    trades = strategy_returns[strategy_returns != 0]
    win_rate = (trades > 0).mean() if len(trades) > 0 else 0
    total_trades = len(trades)
    avg_trade_return = trades.mean() if len(trades) > 0 else 0
    
    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown,
        'calmar_ratio': calmar_ratio,
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
    """Calculate cumulative return over window"""
    return (prices / prices.shift(window)) - 1

def calculate_max_drawdown_series(prices: pd.Series, window: int) -> pd.Series:
    """Calculate rolling maximum drawdown"""
    rolling_max = prices.rolling(window=window).max()
    return (prices - rolling_max) / rolling_max

def calculate_indicator(prices: pd.Series, indicator_type: str, days: int = None) -> pd.Series:
    """Calculate various technical indicators"""
    if indicator_type == "RSI":
        return calculate_rsi(prices, days or 14)
    elif indicator_type == "SMA":
        return calculate_sma(prices, days or 200)
    elif indicator_type == "EMA":
        return calculate_ema(prices, days or 20)
    elif indicator_type == "Current Price":
        return prices
    elif indicator_type == "Cumulative Return":
        return calculate_cumulative_return(prices, days or 252)
    elif indicator_type == "Max Drawdown":
        return calculate_max_drawdown_series(prices, days or 252)
    else:
        return pd.Series(dtype=float)

def evaluate_signal_condition(indicator1_values: pd.Series, indicator2_values: pd.Series, operator: str) -> pd.Series:
    """Evaluate signal condition between two indicators"""
    if indicator1_values.empty or indicator2_values.empty:
        return pd.Series(dtype=bool)
    
    # Align the series
    aligned_data = pd.concat([indicator1_values, indicator2_values], axis=1).dropna()
    if aligned_data.empty:
        return pd.Series(dtype=bool)
    
    indicator1_aligned = aligned_data.iloc[:, 0]
    indicator2_aligned = aligned_data.iloc[:, 1]
    
    if operator == ">":
        return indicator1_aligned > indicator2_aligned
    elif operator == "<":
        return indicator1_aligned < indicator2_aligned
    elif operator == ">=":
        return indicator1_aligned >= indicator2_aligned
    elif operator == "<=":
        return indicator1_aligned <= indicator2_aligned
    elif operator == "==":
        return indicator1_aligned == indicator2_aligned
    elif operator == "!=":
        return indicator1_aligned != indicator2_aligned
    else:
        return pd.Series(dtype=bool)

def calculate_multi_ticker_equity_curve(signals: pd.Series, allocation: dict, data: dict) -> pd.Series:
    """Calculate equity curve for multi-ticker allocation"""
    if signals.empty or not allocation or not data:
        return pd.Series(dtype=float)
    
    # Get tickers and weights
    tickers = allocation.get('tickers', [])
    if not tickers:
        return pd.Series(dtype=float)
    
    # Calculate weighted returns
    weighted_returns = pd.Series(0.0, index=signals.index)
    
    for ticker_config in tickers:
        ticker = ticker_config.get('ticker', '')
        weight = ticker_config.get('weight', 0) / 100.0
        
        if ticker in data:
            ticker_prices = data[ticker]
            ticker_returns = ticker_prices.pct_change()
            weighted_returns += ticker_returns * weight * signals
    
    # Calculate cumulative equity curve
    equity_curve = (1 + weighted_returns).cumprod()
    
    return equity_curve 

# Main Application Interface
st.title("✨ Strategy Validation Tool")
st.caption("Build, test, and validate trading strategies with advanced conditional logic")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Signal Blocks", "💰 Allocation Blocks", "🎯 Strategy Builder", "📈 Backtest"])

# Tab 1: Signal Blocks
with tab1:
    st.header("📊 Signal Blocks")
    
    # Pre-built signal blocks
    st.subheader("🚀 Pre-built Signals")
    st.write("The following signals are automatically available:")
    
    # Display pre-built signals as a clean list
    prebuilt_signal_names = [
        "If 10d RSI QQQ > 80",
        "If 10d RSI QQQ < 30", 
        "If Current Price SPY > 200d SMA SPY",
        "If Current Price SPY > 20d SMA SPY"
    ]
    
    for signal_name in prebuilt_signal_names:
        st.write(f"• **{signal_name}**")
    
    st.markdown("---")
    
    # Create custom signal
    with st.expander("➕ Create Custom Signal", expanded=False):
        signal_name = st.text_input("Signal Name", placeholder="e.g., RSI Oversold")
        
        col1, col2 = st.columns(2)
        
        with col1:
            signal_ticker1 = st.text_input("Signal Ticker", value="SPY", help="Ticker to analyze")
            indicator1 = st.selectbox(
                "Indicator 1",
                ["RSI", "SMA", "EMA", "Current Price", "Cumulative Return", "Max Drawdown", "Static RSI", "RSI Comparison"],
                key="indicator1"
            )
            
            # Days field for first indicator
            if indicator1 not in ["Current Price", "Static RSI", "RSI Comparison"]:
                # Set smart defaults based on indicator type
                default_days1 = 10 if indicator1 == "RSI" else 200
                days1 = st.number_input(
                    f"# of Days for {indicator1}",
                    min_value=1,
                    max_value=252,
                    value=default_days1,
                    key="days1"
                )
            else:
                days1 = None
        
        with col2:
            # Operator selection
            if indicator1 == "Static RSI":
                operator = st.selectbox(
                    "Comparison",
                    ["less_than", "greater_than"],
                    key="operator"
                )
            else:
                operator = st.selectbox(
                    "Operator",
                    [">", "<", ">=", "<=", "==", "!="],
                    key="operator"
                )
            
            # Handle RSI-specific logic
            if indicator1 == "Static RSI":
                # For Static RSI, show threshold input instead of indicator2
                with col2:
                    # Set smart RSI threshold defaults based on operator
                    default_threshold = 32.5 if operator == "less_than" else 78.5
                    rsi_threshold = st.number_input(
                        "RSI Threshold",
                        min_value=0.0,
                        max_value=100.0,
                        value=default_threshold,
                        step=0.5,
                        key="rsi_threshold"
                    )
                signal_ticker2 = signal_ticker1  # Not used for Static RSI
                indicator2 = "Static Value"
                days2 = None
                static_value = rsi_threshold
                
            elif indicator1 == "RSI Comparison":
                # For RSI Comparison, automatically set indicator2 to RSI
                signal_ticker2 = st.text_input("Signal Ticker 2", value="QQQ", help="Second ticker to analyze")
                indicator2 = "RSI"
                days2 = st.number_input(
                    "RSI Period",
                    min_value=1,
                    max_value=50,
                    value=14,
                    key="days2"
                )
                static_value = None
                
            else:
                # Regular Custom Indicator logic
                signal_ticker2 = st.text_input("Signal Ticker 2", value="QQQ", help="Second ticker to analyze")
                indicator2 = st.selectbox(
                    "Indicator 2",
                    ["SMA", "EMA", "Current Price", "Cumulative Return", "Max Drawdown", "Static Value"],
                    key="indicator2"
                )
                
                # Days field for second indicator or static value
                if indicator2 not in ["Current Price", "Static Value"]:
                    # Set smart defaults based on indicator type
                    default_days2 = 200 if indicator2 == "SMA" else 14
                    days2 = st.number_input(
                        f"# of Days for {indicator2}",
                        min_value=1,
                        max_value=252,
                        value=default_days2,
                        key="days2"
                    )
                elif indicator2 == "Static Value":
                    static_value = st.number_input(
                        "Static Value",
                        min_value=0.0,
                        max_value=1000.0,
                        value=50.0,
                        step=0.1,
                        key="static_value"
                    )
                
                # Handle Signal Ticker 2 logic
                if indicator2 == "Static Value":
                    signal_ticker2 = signal_ticker1  # Use same ticker for static value comparisons
            
            # Display the signal logic
            if indicator1 == "Static RSI":
                st.info(f"**Signal Logic:** {signal_ticker1} RSI({days1}) {operator} {static_value}")
            elif indicator1 == "RSI Comparison":
                st.info(f"**Signal Logic:** {signal_ticker1} RSI({days1}) vs {signal_ticker2} RSI({days2})")
            elif indicator1 not in ["Current Price"] and indicator2 not in ["Current Price", "Static Value"]:
                st.info(f"**Signal Logic:** {signal_ticker1} {indicator1}({days1}) {operator} {signal_ticker2} {indicator2}({days2})")
            elif indicator1 not in ["Current Price"] and indicator2 == "Static Value":
                st.info(f"**Signal Logic:** {signal_ticker1} {indicator1}({days1}) {operator} {static_value}")
            elif indicator1 not in ["Current Price"]:
                st.info(f"**Signal Logic:** {signal_ticker1} {indicator1} {operator} {signal_ticker2} {indicator2}")
            elif indicator2 not in ["Current Price", "Static Value"]:
                st.info(f"**Signal Logic:** {signal_ticker1} {indicator1} {operator} {signal_ticker2} {indicator2}({days2})")
            elif indicator2 == "Static Value":
                st.info(f"**Signal Logic:** {signal_ticker1} {indicator1} {operator} {static_value}")
            else:
                st.info(f"**Signal Logic:** {signal_ticker1} {indicator1} {operator} {signal_ticker2} {indicator2}")
            
            if st.button("Add Signal", type="primary"):
                if indicator1 == "Static RSI":
                    signal = {
                        'name': signal_name,
                        'type': 'Static RSI',
                        'signal_ticker': signal_ticker1,
                        'target_ticker': signal_ticker1,  # Default to same ticker
                        'rsi_period': days1,
                        'rsi_threshold': static_value,
                        'comparison': operator
                    }
                elif indicator1 == "RSI Comparison":
                    signal = {
                        'name': signal_name,
                        'type': 'RSI Comparison',
                        'signal_ticker': signal_ticker1,
                        'comparison_ticker': signal_ticker2,
                        'target_ticker': signal_ticker1,  # Default to same ticker
                        'rsi_period': days1,
                        'comparison_operator': 'less_than' if operator == '<' else 'greater_than'
                    }
                else:
                    signal = {
                        'name': signal_name,
                        'type': 'Custom Indicator',
                        'signal_ticker1': signal_ticker1,
                        'signal_ticker2': signal_ticker2,
                        'indicator1': indicator1,
                        'indicator2': indicator2,
                        'operator': operator,
                        'days1': days1 if indicator1 not in ["Current Price"] else None,
                        'days2': days2 if indicator2 not in ["Current Price", "Static Value"] else None,
                        'static_value': static_value if indicator2 == "Static Value" else None
                    }
                st.session_state.signals.append(signal)
                st.success(f"Reference Signal '{signal_name}' added!")
                st.rerun()
    
    # Display existing signals
    if st.session_state.signals:
        st.subheader("📋 Active Reference Signal Blocks")
        for signal_idx, signal in enumerate(st.session_state.signals):
            with st.container():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**{signal['name']}**")
                    if signal['type'] == "Custom Indicator":
                        indicator1_text = f"{signal['indicator1']}({signal['days1']})" if signal['days1'] else signal['indicator1']
                        if signal['indicator2'] == "Static Value":
                            indicator2_text = str(signal['static_value'])
                        else:
                            indicator2_text = f"{signal['indicator2']}({signal['days2']})" if signal['days2'] else signal['indicator2']
                        st.caption(f"{signal['signal_ticker1']} {indicator1_text} {signal['operator']} {signal['signal_ticker2']} {indicator2_text}")
                    elif signal['type'] == "Static RSI":
                        st.caption(f"{signal['signal_ticker']} RSI {signal['rsi_period']}-day {signal['comparison']} {signal['rsi_threshold']}")
                    else:
                        st.caption(f"{signal['signal_ticker']} vs {signal['comparison_ticker']} RSI {signal['comparison_operator']}")
                with col2:
                    if st.button("🗑️", key=f"delete_signal_{signal_idx}"):
                        st.session_state.signals.pop(signal_idx)
                        st.rerun()
    else:
        st.info("No reference signal blocks created yet. Create your first signal above.")

# Tab 2: Allocation Blocks
with tab2:
    st.header("💰 Allocation Blocks")
    
    # Pre-built allocation blocks
    st.subheader("🚀 Pre-built Allocations")
    st.write("The following allocations are automatically available:")
    
    # Display pre-built allocations as clean lists
    st.write("**Single Ticker Allocations:**")
    single_ticker_allocations = ["TQQQ (3x QQQ)", "QLD (2x QQQ)", "QQQ (Nasdaq)", "SPY (S&P 500)", "XLP (Consumer Staples)", "XLU (Utilities)", "BIL (T-Bills)", "UVXY (Volatility)", "VIXY (VIX)"]
    for allocation in single_ticker_allocations:
        st.write(f"• **{allocation}**")
    
    st.write("**Multi-Ticker Allocations:**")
    multi_ticker_allocations = ["BIL VIXY Defensive (50/50)", "XLP XLU Defensive (50/50)"]
    for allocation in multi_ticker_allocations:
        st.write(f"• **{allocation}**")
    
    st.markdown("---")
    
    # Create allocation
    with st.expander("➕ Create Allocation Block", expanded=False):
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
            ticker_weight = st.number_input("Weight (%)", min_value=0, max_value=100, value=100, key="ticker_weight")
        
        # Info about auto-add feature
        if not st.session_state.current_allocation_tickers and new_ticker:
            st.info("💡 **Tip:** You can click 'Create Allocation Block' to automatically add this ticker with 100% weight.")
        
        # Add ticker button
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
            
            # Equal weight button for multiple tickers
            if len(st.session_state.current_allocation_tickers) > 1:
                if st.button("⚖️ Equal Weight All", key="equal_weight"):
                    equal_weight = 100 / len(st.session_state.current_allocation_tickers)
                    for tc in st.session_state.current_allocation_tickers:
                        tc['weight'] = equal_weight
                    st.rerun()
            
            if total_weight != 100:
                if total_weight > 100:
                    st.error(f"⚠️ Total weight exceeds 100% ({total_weight}%)")
                else:
                    st.warning(f"ℹ️ Total weight: {total_weight}% ({(100-total_weight):.1f}% unallocated)")
            else:
                st.success(f"✅ Total weight: {total_weight}%")
        
        # Create allocation button
        if st.button("Create Allocation Block", type="primary"):
            if allocation_name:
                # Check if we have tickers in the list or if we should auto-add the current ticker
                if st.session_state.current_allocation_tickers:
                    # Use existing tickers
                    tickers_to_use = st.session_state.current_allocation_tickers.copy()
                elif new_ticker:
                    # Auto-add the current ticker with 100% weight
                    tickers_to_use = [{
                        'ticker': new_ticker.upper(),
                        'weight': 100
                    }]
                else:
                    st.error("Please provide at least one ticker.")
                
                total_weight = sum([tc['weight'] for tc in tickers_to_use])
                
                if total_weight == 100:
                    allocation = {
                        'name': allocation_name,
                        'tickers': tickers_to_use,
                        'total_weight': total_weight
                    }
                    st.session_state.output_allocations[allocation_name] = allocation
                    st.session_state.current_allocation_tickers = []  # Reset for next allocation
                    st.success(f"Allocation Block '{allocation_name}' created successfully!")
                    st.rerun()
                else:
                    st.error("Total weight must equal 100% to create allocation.")
            else:
                st.error("Please provide an allocation name.")
    
    # Display existing allocations
    if st.session_state.output_allocations:
        st.subheader("📋 Active Allocation Blocks")
        for name, allocation in st.session_state.output_allocations.items():
            with st.container():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**{allocation['name']}**")
                    ticker_text = ", ".join([f"{tc['ticker']} ({tc['weight']}%)" for tc in allocation['tickers']])
                    st.caption(ticker_text)
                with col2:
                    if st.button("🗑️", key=f"delete_allocation_{name}"):
                        del st.session_state.output_allocations[name]
                        st.rerun()
    else:
        st.info("No allocation blocks created yet. Create your first allocation above.")

# Tab 3: Strategy Builder
with tab3:
    st.header("🎯 Strategy Builder")
    
    # Ensure pre-built signals are available
    if not st.session_state.signals:
        # Re-add pre-built signals if they're missing
        prebuilt_signals = [
            {
                'name': 'If 10d RSI QQQ > 80',
                'type': 'Custom Indicator',
                'signal_ticker1': 'QQQ',
                'signal_ticker2': 'QQQ',
                'indicator1': 'RSI',
                'indicator2': 'Static Value',
                'operator': '>',
                'days1': 10,
                'days2': None,
                'static_value': 80.0
            },
            {
                'name': 'If 10d RSI QQQ < 30',
                'type': 'Custom Indicator',
                'signal_ticker1': 'QQQ',
                'signal_ticker2': 'QQQ',
                'indicator1': 'RSI',
                'indicator2': 'Static Value',
                'operator': '<',
                'days1': 10,
                'days2': None,
                'static_value': 30.0
            },
            {
                'name': 'If Current Price SPY > 200d SMA SPY',
                'type': 'Custom Indicator',
                'signal_ticker1': 'SPY',
                'signal_ticker2': 'SPY',
                'indicator1': 'Current Price',
                'indicator2': 'SMA',
                'operator': '>',
                'days1': None,
                'days2': 200,
                'static_value': None
            },
            {
                'name': 'If Current Price SPY > 20d SMA SPY',
                'type': 'Custom Indicator',
                'signal_ticker1': 'SPY',
                'signal_ticker2': 'SPY',
                'indicator1': 'Current Price',
                'indicator2': 'SMA',
                'operator': '>',
                'days1': None,
                'days2': 20,
                'static_value': None
            }
        ]
        st.session_state.signals = prebuilt_signals
    
    # Reference Blocks Management
    with st.expander("📚 Reference Blocks Manager", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("💾 Save Current Block")
            block_name = st.text_input("Block Name", placeholder="e.g., RSI Oversold Strategy")
            if st.button("💾 Save as Reference Block"):
                if st.session_state.strategy_branches:
                    # Save the last branch as a reference block
                    last_branch = st.session_state.strategy_branches[-1]
                    if block_name:
                        save_reference_block(last_branch, block_name)
                        st.success(f"✅ Block '{block_name}' saved as reference!")
                        st.rerun()
                    else:
                        st.error("Please provide a block name.")
                else:
                    st.warning("No strategy branches to save.")
        
        with col2:
            st.subheader("📋 Saved Reference Blocks")
            reference_blocks = get_all_reference_blocks()
            if reference_blocks:
                for block_name in reference_blocks:
                    col_a, col_b, col_c = st.columns([3, 1, 1])
                    with col_a:
                        st.write(f"**{block_name}**")
                        st.caption(get_reference_block_preview(block_name))
                    with col_b:
                        if st.button("📋 Load", key=f"load_ref_{block_name}"):
                            loaded_block = load_reference_block(block_name)
                            if loaded_block:
                                st.session_state.strategy_branches.append(loaded_block)
                                st.success(f"✅ Block '{block_name}' loaded!")
                                st.rerun()
                    with col_c:
                        if st.button("🗑️", key=f"delete_ref_{block_name}"):
                            delete_reference_block(block_name)
                            st.success(f"✅ Block '{block_name}' deleted!")
                            st.rerun()
            else:
                st.info("No reference blocks saved yet.")
    
    # Cache management
    with st.expander("⚡ Cache Manager", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            cache_stats = get_cache_stats()
            st.metric("Cache Hits", cache_stats['hits'])
            st.metric("Cache Misses", cache_stats['misses'])
        with col2:
            st.metric("Hit Rate", f"{cache_stats['hit_rate']:.1f}%")
            if st.button("🗑️ Clear Cache"):
                clear_cache()
                st.success("Cache cleared!")
    
    # Strategy builder interface
    st.markdown('<div class="strategy-builder">', unsafe_allow_html=True)
    
    # Strategy creation interface
    st.subheader("🎯 Build Your Strategy")
    
    # Add new strategy component
    if st.button("➕", key="add_component"):
        new_branch = {
            'type': 'if_else',
            'signals': [],
            'allocations': [],
            'else_allocations': [],
            'collapsed': False
        }
        st.session_state.strategy_branches.append(new_branch)
        st.rerun()
    
    # Display strategy branches
    if st.session_state.strategy_branches:
        st.markdown("---")
        st.subheader("📋 Active Strategy Components")
        
        for branch_idx, branch in enumerate(st.session_state.strategy_branches):
            st.markdown("---")
            
            # Handle If/Else structure differently
            if branch.get('type') == 'if_else':
                # If/Else collapsible block
                if branch.get('is_else_if'):
                    # ELSE IF block styling
                    st.markdown("""
                    <div class="if-else-block" style="border-left: 4px solid #FF5722; background-color: #FFF3E0;">
                        <div class="if-else-header">
                            <span>🔗 ELSE IF</span>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    # Regular IF/ELSE block styling
                    st.markdown("""
                    <div class="if-else-block">
                        <div class="if-else-header">
                            <span>🔗 If/Else</span>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Block operations menu inside the If/Else box
                col_menu, col_spacer = st.columns([1, 9])
                with col_menu:
                    block_menu_action = st.selectbox(
                        '',
                        ["", "Copy", "Paste", "Delete", "Save as Reference"],
                        key=f"block_menu_{branch_idx}",
                        label_visibility="collapsed"
                    )
                    if block_menu_action == "Copy":
                        st.session_state.copied_block = copy.deepcopy(branch)
                        st.success("Block copied!")
                    elif block_menu_action == "Paste":
                        if st.session_state.copied_block:
                            st.session_state.strategy_branches.insert(branch_idx+1, copy.deepcopy(st.session_state.copied_block))
                            st.success("Block pasted!")
                            st.rerun()
                        else:
                            st.warning("⚠️ No block in clipboard")
                    elif block_menu_action == "Delete":
                        st.session_state.strategy_branches.pop(branch_idx)
                        st.rerun()
                    elif block_menu_action == "Save as Reference":
                        ref_name = st.text_input("Reference Block Name:", key=f"ref_name_{branch_idx}")
                        if st.button("💾 Save", key=f"save_ref_{branch_idx}"):
                            if ref_name:
                                save_reference_block(branch, ref_name)
                                st.success(f"✅ Block saved as '{ref_name}'!")
                                st.rerun()
                            else:
                                st.error("Please provide a name.")
                
                st.markdown("""
                    </div>
                """, unsafe_allow_html=True)
                
                # Make the entire IF/THEN/ELSE block collapsible
                with st.expander(f"🔗 IF/THEN/ELSE Block {branch_idx + 1}", expanded=True):
                    # IF section
                    st.markdown('<div class="condition-block" style="border-left: 3px solid #2196F3; padding-left: 10px; margin-left: 0px;">', unsafe_allow_html=True)
                    if branch.get('is_else_if'):
                        st.markdown("**ELSE IF:**")
                    else:
                        st.markdown("**IF:**")
                    
                    # Always visible signal dropdown
                    st.markdown("**Add Signal:**")
                    if st.session_state.signals:
                        selected_signal = st.selectbox(
                            "Select Signal:",
                            [""] + [s['name'] for s in st.session_state.signals],
                            key=f"if_signal_select_{branch_idx}"
                        )
                        if selected_signal:
                            if 'signals' not in branch:
                                branch['signals'] = []
                            # Check if signal already exists
                            if not any(s.get('signal') == selected_signal for s in branch.get('signals', [])):
                                branch['signals'].append({
                                    'signal': selected_signal, 
                                    'negated': False, 
                                    'operator': 'AND'
                                })
                                st.success(f"✅ Signal '{selected_signal}' added!")
                                st.rerun()
                    else:
                        st.warning("No signals available. Create signals in the Signal Blocks tab first.")
                    
                    # Display IF signals in collapsible expander
                    if branch.get('signals'):
                        with st.expander(f"📊 IF Signals ({len(branch['signals'])})", expanded=True):
                            for signal_idx, signal_config in enumerate(branch['signals']):
                                # Display each signal in its own row
                                st.markdown(f"**Signal {signal_idx + 1}:**")
                                
                                col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                                
                                with col1:
                                    signal_config['signal'] = st.selectbox(
                                        f"Signal {signal_idx + 1}", 
                                        [""] + [s['name'] for s in st.session_state.signals],
                                        index=0 if not signal_config.get('signal') else 
                                        [s['name'] for s in st.session_state.signals].index(signal_config['signal']) + 1,
                                        key=f"if_branch_{branch_idx}_signal_{signal_idx}"
                                    )
                                
                                with col2:
                                    signal_config['negated'] = st.checkbox("NOT", key=f"if_branch_{branch_idx}_negated_{signal_idx}")
                                
                                with col3:
                                    if len(branch['signals']) > 1 and signal_idx < len(branch['signals']) - 1:
                                        signal_config['operator'] = st.selectbox(
                                            "Operator",
                                            ["AND", "OR"],
                                            index=0 if signal_config.get('operator', 'AND') == 'AND' else 1,
                                            key=f"if_branch_{branch_idx}_operator_{signal_idx}"
                                        )
                                    else:
                                        st.write("")  # Empty space for alignment
                                
                                with col4:
                                    if len(branch['signals']) > 1:
                                        if st.button("🗑️", key=f"remove_if_branch_{branch_idx}_signal_{signal_idx}"):
                                            branch['signals'].pop(signal_idx)
                                            st.rerun()
                                    else:
                                        st.write("")  # Empty space for alignment
                                
                                # Add some spacing between signals
                                st.markdown("<br>", unsafe_allow_html=True)
                            
                            # Add button to add more signals
                            if st.button("➕ Add Another Signal", key=f"add_more_if_signal_{branch_idx}"):
                                branch['signals'].append({
                                    'signal': '', 
                                    'negated': False, 
                                    'operator': 'AND'
                                })
                                st.rerun()
                    else:
                        st.write("**No signals in IF yet**")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # THEN section
                    st.markdown('<div class="then-block" style="border-left: 3px solid #FF9800; padding-left: 10px; margin-left: 0px;">', unsafe_allow_html=True)
                    st.markdown("**THEN:**")
                    
                    # Always visible allocation dropdown
                    st.markdown("**Add Allocation:**")
                    if st.session_state.output_allocations:
                        selected_allocation = st.selectbox(
                            "Select Allocation:",
                            [""] + list(st.session_state.output_allocations.keys()),
                            key=f"then_allocation_select_{branch_idx}_{id(branch)}"
                        )
                        if selected_allocation:
                            if 'allocations' not in branch:
                                branch['allocations'] = []
                            # Check if allocation already exists
                            if not any(a.get('allocation') == selected_allocation for a in branch.get('allocations', [])):
                                branch['allocations'].append({
                                    'allocation': selected_allocation, 
                                    'weight': 100
                                })
                                st.success(f"✅ Allocation '{selected_allocation}' added!")
                                st.rerun()
                    else:
                        st.warning("No allocations available. Create allocations in the Allocation Blocks tab first.")
                    
                    # Display THEN allocations in collapsible expander
                    if branch.get('allocations'):
                        with st.expander(f"💰 THEN Allocations ({len(branch['allocations'])})", expanded=True):
                            for alloc_idx, allocation_config in enumerate(branch['allocations']):
                                col1, col2, col3 = st.columns([2, 1, 1])
                                with col1:
                                    allocation_config['allocation'] = st.selectbox(
                                        f"Allocation {alloc_idx + 1}", 
                                        list(st.session_state.output_allocations.keys()),
                                        key=f"then_branch_{branch_idx}_allocation_{alloc_idx}"
                                    )
                                with col2:
                                    allocation_config['weight'] = st.number_input(
                                        "Weight %",
                                        min_value=0,
                                        max_value=100,
                                        value=allocation_config.get('weight', 100),
                                        key=f"then_branch_{branch_idx}_weight_{alloc_idx}"
                                    )
                                with col3:
                                    if len(branch['allocations']) > 1:  # Don't allow removing the last allocation
                                        if st.button("🗑️", key=f"remove_then_{branch_idx}_{alloc_idx}_{id(allocation_config)}_delete"):
                                            branch['allocations'].pop(alloc_idx)
                                            st.rerun()
                                    else:
                                        st.write("")  # Empty space for alignment
                            
                            # Show total weight for this branch
                            total_branch_weight = sum(alloc.get('weight', 0) for alloc in branch['allocations'])
                            if total_branch_weight != 100:
                                if total_branch_weight > 100:
                                    st.error(f"⚠️ Branch total weight: {total_branch_weight}% (exceeds 100%)")
                                else:
                                    st.warning(f"ℹ️ Branch total weight: {total_branch_weight}% ({(100-total_branch_weight):.1f}% unallocated)")
                            else:
                                st.success(f"✅ Branch total weight: {total_branch_weight}%")
                            
                            # Add button to add more allocations
                            if st.button("➕ Add Another Allocation", key=f"add_more_then_allocation_{branch_idx}_{id(branch)}"):
                                branch['allocations'].append({
                                    'allocation': '', 
                                    'weight': 100
                                })
                                st.rerun()
                    else:
                        st.write("**No allocations in THEN yet**")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # ELSE section
                    st.markdown('<div class="else-block" style="border-left: 3px solid #4CAF50; padding-left: 10px; margin-left: 30px;">', unsafe_allow_html=True)
                    st.markdown("**ELSE:**")
                    
                    # Weight distribution between allocations and chains
                    st.markdown("**ELSE Weight Distribution:**")
                    col1, col2 = st.columns(2)
                    with col1:
                        if 'else_allocation_weight' not in branch:
                            branch['else_allocation_weight'] = 50
                        new_allocation_weight = st.number_input(
                            "Allocation Weight %",
                            min_value=0,
                            max_value=100,
                            value=branch.get('else_allocation_weight', 50),
                            key=f"else_allocation_weight_{branch_idx}"
                        )
                        # Only update if value actually changed to avoid unnecessary reruns
                        if new_allocation_weight != branch.get('else_allocation_weight', 50):
                            branch['else_allocation_weight'] = new_allocation_weight
                    with col2:
                        if 'else_chain_weight' not in branch:
                            branch['else_chain_weight'] = 50
                        new_chain_weight = st.number_input(
                            "Chain Weight %",
                            min_value=0,
                            max_value=100,
                            value=branch.get('else_chain_weight', 50),
                            key=f"else_chain_weight_{branch_idx}"
                        )
                        # Only update if value actually changed to avoid unnecessary reruns
                        if new_chain_weight != branch.get('else_chain_weight', 50):
                            branch['else_chain_weight'] = new_chain_weight
                    
                    # Validate total weight
                    total_else_weight = branch.get('else_allocation_weight', 0) + branch.get('else_chain_weight', 0)
                    if total_else_weight != 100:
                        if total_else_weight > 100:
                            st.error(f"⚠️ ELSE total weight: {total_else_weight}% (exceeds 100%)")
                        else:
                            st.warning(f"ℹ️ ELSE total weight: {total_else_weight}% ({(100-total_else_weight):.1f}% unallocated)")
                    else:
                        st.success(f"✅ ELSE total weight: {total_else_weight}%")
                    
                    # Always visible allocation dropdown for ELSE
                    st.markdown("**Add Allocation to ELSE:**")
                    if st.session_state.output_allocations:
                        selected_else_allocation = st.selectbox(
                            "Select Allocation:",
                            [""] + list(st.session_state.output_allocations.keys()),
                            key=f"else_allocation_select_{branch_idx}_{id(branch)}"
                        )
                        if selected_else_allocation:
                            if 'else_allocations' not in branch:
                                branch['else_allocations'] = []
                            # Check if allocation already exists
                            if not any(a.get('allocation') == selected_else_allocation for a in branch.get('else_allocations', [])):
                                branch['else_allocations'].append({
                                    'allocation': selected_else_allocation, 
                                    'weight': 100
                                })
                                st.success(f"✅ Allocation '{selected_else_allocation}' added to ELSE!")
                                # Remove rerun to prevent state conflicts
                    else:
                        st.warning("No allocations available. Create allocations in the Allocation Blocks tab first.")
                    
                    # Separate button for adding IF/THEN/ELSE chains
                    if st.button("🔗 Add IF/THEN/ELSE Chain to ELSE", key=f"add_else_nested_chain_{branch_idx}_{id(branch)}"):
                        if 'else_nested_chains' not in branch:
                            branch['else_nested_chains'] = []
                        branch['else_nested_chains'].append({
                            'type': 'nested_if_else_chain',
                            'chain_blocks': [{
                                'type': 'chain_if_else',
                                'signals': [],
                                'allocations': [],
                                'else_allocations': [],
                                'else_signals': [],
                                'else_nested_blocks': [],
                                'is_else_if': False
                            }]
                        })
                        st.success(f"✅ IF/THEN/ELSE chain added to ELSE!")
                        # Remove rerun to prevent state conflicts
                    

                    
                    # Display ELSE allocations in collapsible expander
                    if branch.get('else_allocations'):
                        with st.expander(f"💰 ELSE Allocations ({len(branch['else_allocations'])})", expanded=True):
                            for else_alloc_idx, else_allocation_config in enumerate(branch['else_allocations']):
                                col1, col2, col3 = st.columns([2, 1, 1])
                                with col1:
                                    else_allocation_config['allocation'] = st.selectbox(
                                        f"ELSE Allocation {else_alloc_idx + 1}", 
                                        list(st.session_state.output_allocations.keys()),
                                        key=f"else_branch_{branch_idx}_allocation_{else_alloc_idx}"
                                    )
                                with col2:
                                    new_weight = st.number_input(
                                        "Weight %",
                                        min_value=0,
                                        max_value=100,
                                        value=else_allocation_config.get('weight', 100),
                                        key=f"else_branch_{branch_idx}_weight_{else_alloc_idx}"
                                    )
                                    # Only update if value actually changed to avoid unnecessary state changes
                                    if new_weight != else_allocation_config.get('weight', 100):
                                        else_allocation_config['weight'] = new_weight
                                with col3:
                                    if len(branch['else_allocations']) > 1:  # Don't allow removing the last allocation
                                        if st.button("🗑️", key=f"remove_else_{branch_idx}_{else_alloc_idx}_{id(else_allocation_config)}_delete"):
                                            branch['else_allocations'].pop(else_alloc_idx)
                                            # Remove rerun to prevent state conflicts
                                    else:
                                        st.write("")  # Empty space for alignment
                            
                            # Show total ELSE allocation weight for this branch
                            total_else_allocation_weight = sum(alloc.get('weight', 0) for alloc in branch['else_allocations'])
                            if total_else_allocation_weight != 100:
                                if total_else_allocation_weight > 100:
                                    st.error(f"⚠️ ELSE allocation weight: {total_else_allocation_weight}% (exceeds 100%)")
                                else:
                                    st.warning(f"ℹ️ ELSE allocation weight: {total_else_allocation_weight}% ({(100-total_else_allocation_weight):.1f}% unallocated)")
                            else:
                                st.success(f"✅ ELSE allocation weight: {total_else_allocation_weight}%")
                            
                            # Show weight distribution info
                            allocation_weight = branch.get('else_allocation_weight', 50)
                            chain_weight = branch.get('else_chain_weight', 50)
                            st.info(f"📊 ELSE Distribution: {allocation_weight}% allocations, {chain_weight}% chains")
                            
                            # Add button to add more allocations to ELSE
                            if st.button("➕ Add Another Allocation to ELSE", key=f"add_more_else_allocation_{branch_idx}_{id(branch)}"):
                                branch['else_allocations'].append({
                                    'allocation': '', 
                                    'weight': 100
                                })
                                # Remove rerun to prevent state conflicts
                    
                    # Display nested IF/ELSE blocks in ELSE
                    if branch.get('else_nested_blocks'):
                        with st.expander(f"🔗 ELSE Nested If/Else Blocks ({len(branch['else_nested_blocks'])})", expanded=True):
                            for nested_idx, nested_block in enumerate(branch['else_nested_blocks']):
                                st.markdown(f"""
                                <div class="nested-else-block" style="border-left: 3px solid #9C27B0; padding-left: 10px; margin-left: 20px;">
                                    <div class="nested-else-header">
                                        <span>🔗 Nested If/Else Block {nested_idx + 1}</span>
                                    </div>
                                """, unsafe_allow_html=True)
                                
                                # Nested IF section
                                st.markdown('<div class="nested-condition-block" style="border-left: 3px solid #2196F3; padding-left: 10px; margin-left: 0px;">', unsafe_allow_html=True)
                                st.markdown("**NESTED IF:**")
                                
                                # Add signal button for nested IF
                                if st.button("➕", key=f"add_nested_if_{branch_idx}_{nested_idx}"):
                                    if 'signals' not in nested_block:
                                        nested_block['signals'] = []
                                    nested_block['signals'].append({
                                        'signal': '', 
                                        'negated': False, 
                                        'operator': 'AND'
                                    })
                                    st.rerun()
                                
                                # Display nested IF signals
                                if nested_block.get('signals'):
                                    with st.expander(f"📊 Nested IF Signals ({len(nested_block['signals'])})", expanded=True):
                                        for signal_idx, signal_config in enumerate(nested_block['signals']):
                                            st.markdown(f"**Nested Signal {signal_idx + 1}:**")
                                            
                                            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                                            with col1:
                                                if signal_config.get('signal'):
                                                    st.write(f"• {signal_config['signal']}")
                                                else:
                                                    signal_config['signal'] = st.selectbox(
                                                        "Select Signal:",
                                                        [""] + [s['name'] for s in st.session_state.signals],
                                                        key=f"nested_if_signal_{branch_idx}_{nested_idx}_{signal_idx}"
                                                    )
                                            with col2:
                                                signal_config['negated'] = st.checkbox("NOT", key=f"nested_if_negated_{branch_idx}_{nested_idx}_{signal_idx}")
                                            with col3:
                                                if len(nested_block['signals']) > 1 and signal_idx < len(nested_block['signals']) - 1:
                                                    signal_config['operator'] = st.selectbox(
                                                        "Operator",
                                                        ["AND", "OR"],
                                                        index=0 if signal_config.get('operator', 'AND') == 'AND' else 1,
                                                        key=f"nested_if_operator_{branch_idx}_{nested_idx}_{signal_idx}"
                                                    )
                                                else:
                                                    st.write("")
                                            with col4:
                                                if len(nested_block['signals']) > 1:
                                                    if st.button("🗑️", key=f"remove_nested_if_signal_{branch_idx}_{nested_idx}_{signal_idx}"):
                                                        nested_block['signals'].pop(signal_idx)
                                                        st.rerun()
                                                else:
                                                    st.write("")
                                            
                                            st.markdown("<br>", unsafe_allow_html=True)
                                else:
                                    st.write("**No signals in nested IF yet**")
                                
                                st.markdown('</div>', unsafe_allow_html=True)
                                
                                # Nested THEN section
                                st.markdown('<div class="nested-then-block" style="border-left: 3px solid #FF9800; padding-left: 10px; margin-left: 0px;">', unsafe_allow_html=True)
                                st.markdown("**NESTED THEN:**")
                                
                                # Add allocation button for nested THEN
                                if st.button("➕", key=f"add_nested_then_{branch_idx}_{nested_idx}"):
                                    if 'allocations' not in nested_block:
                                        nested_block['allocations'] = []
                                    nested_block['allocations'].append({
                                        'allocation': '', 
                                        'weight': 100
                                    })
                                    st.rerun()
                                
                                # Display nested THEN allocations
                                if nested_block.get('allocations'):
                                    with st.expander(f"💰 Nested THEN Allocations ({len(nested_block['allocations'])})", expanded=True):
                                        for alloc_idx, allocation_config in enumerate(nested_block['allocations']):
                                            col1, col2, col3 = st.columns([2, 1, 1])
                                            with col1:
                                                allocation_config['allocation'] = st.selectbox(
                                                    f"Nested Allocation {alloc_idx + 1}", 
                                                    list(st.session_state.output_allocations.keys()),
                                                    key=f"nested_then_allocation_{branch_idx}_{nested_idx}_{alloc_idx}"
                                                )
                                            with col2:
                                                allocation_config['weight'] = st.number_input(
                                                    "Weight %",
                                                    min_value=0,
                                                    max_value=100,
                                                    value=allocation_config.get('weight', 100),
                                                    key=f"nested_then_weight_{branch_idx}_{nested_idx}_{alloc_idx}"
                                                )
                                            with col3:
                                                if len(nested_block['allocations']) > 1:
                                                    if st.button("🗑️", key=f"remove_nested_{branch_idx}_{nested_idx}_{alloc_idx}_{id(allocation_config)}_delete"):
                                                        nested_block['allocations'].pop(alloc_idx)
                                                        st.rerun()
                                                else:
                                                    st.write("")
                                        
                                        # Show total weight for nested branch
                                        total_nested_weight = sum(alloc.get('weight', 0) for alloc in nested_block['allocations'])
                                        if total_nested_weight != 100:
                                            if total_nested_weight > 100:
                                                st.error(f"⚠️ Nested total weight: {total_nested_weight}% (exceeds 100%)")
                                            else:
                                                st.warning(f"ℹ️ Nested total weight: {total_nested_weight}% ({(100-total_nested_weight):.1f}% unallocated)")
                                        else:
                                            st.success(f"✅ Nested total weight: {total_nested_weight}%")
                                else:
                                    st.write("**No allocations in nested THEN yet**")
                                
                                st.markdown('</div>', unsafe_allow_html=True)
                                
                                # Delete nested block
                                if st.button("🗑️ Delete Nested Block", key=f"delete_nested_{branch_idx}_{nested_idx}"):
                                    branch['else_nested_blocks'].pop(nested_idx)
                                    st.rerun()
                                
                                st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Display nested IF/THEN/ELSE chains in ELSE
                    if branch.get('else_nested_chains'):
                        with st.expander(f"🔗 ELSE Nested IF/THEN/ELSE Chains ({len(branch['else_nested_chains'])})", expanded=True):
                            for chain_idx, chain in enumerate(branch['else_nested_chains']):
                                st.markdown(f"""
                                <div class="nested-chain-block" style="border-left: 3px solid #E91E63; padding-left: 10px; margin-left: 20px;">
                                    <div class="nested-chain-header">
                                        <span>🔗 Nested IF/THEN/ELSE Chain {chain_idx + 1}</span>
                                    </div>
                                """, unsafe_allow_html=True)
                                
                                # Display each block in the chain
                                for block_idx, chain_block in enumerate(chain['chain_blocks']):
                                    # Chain block styling
                                    if chain_block.get('is_else_if'):
                                        st.markdown(f"""
                                        <div class="chain-else-if-block" style="border-left: 4px solid #FF5722; background-color: #FFF3E0; margin-left: 10px;">
                                            <div class="chain-else-if-header">
                                                <span>🔗 Chain ELSE IF Block {block_idx + 1}</span>
                                            </div>
                                        """, unsafe_allow_html=True)
                                    else:
                                        st.markdown(f"""
                                        <div class="chain-if-block" style="border-left: 4px solid #2196F3; background-color: #E3F2FD; margin-left: 10px;">
                                            <div class="chain-if-header">
                                                <span>🔗 Chain IF Block {block_idx + 1}</span>
                                            </div>
                                        """, unsafe_allow_html=True)
                                    
                                    # Chain IF section - Streamlined format like primary blocks
                                    st.markdown('<div class="chain-condition-block" style="border-left: 3px solid #2196F3; padding-left: 10px; margin-left: 0px;">', unsafe_allow_html=True)
                                    if chain_block.get('is_else_if'):
                                        st.markdown("**CHAIN ELSE IF:**")
                                    else:
                                        st.markdown("**CHAIN IF:**")
                                    
                                    # Direct signal selection (no + button needed)
                                    if st.session_state.signals:
                                        selected_chain_signal = st.selectbox(
                                            "Select Signal:",
                                            [""] + [s['name'] for s in st.session_state.signals],
                                            key=f"chain_signal_select_{branch_idx}_{chain_idx}_{block_idx}"
                                        )
                                        if selected_chain_signal:
                                            if 'signals' not in chain_block:
                                                chain_block['signals'] = []
                                            # Check if signal already exists
                                            if not any(s.get('signal') == selected_chain_signal for s in chain_block.get('signals', [])):
                                                chain_block['signals'].append({
                                                    'signal': selected_chain_signal, 
                                                    'negated': False, 
                                                    'operator': 'AND'
                                                })
                                                st.success(f"✅ Signal '{selected_chain_signal}' added to chain!")
                                                # Remove rerun to prevent state conflicts
                                    else:
                                        st.warning("No signals available. Create signals in the Signal Blocks tab first.")
                                    
                                    # Display chain IF signals
                                    if chain_block.get('signals'):
                                        with st.expander(f"📊 Chain IF Signals ({len(chain_block['signals'])})", expanded=True):
                                            for signal_idx, signal_config in enumerate(chain_block['signals']):
                                                st.markdown(f"**Chain Signal {signal_idx + 1}:**")
                                                
                                                col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                                                with col1:
                                                    signal_config['signal'] = st.selectbox(
                                                        f"Chain Signal {signal_idx + 1}", 
                                                        [""] + [s['name'] for s in st.session_state.signals],
                                                        index=0 if not signal_config.get('signal') else 
                                                        [s['name'] for s in st.session_state.signals].index(signal_config['signal']) + 1,
                                                        key=f"chain_if_signal_{branch_idx}_{chain_idx}_{block_idx}_{signal_idx}"
                                                    )
                                                with col2:
                                                    signal_config['negated'] = st.checkbox("NOT", key=f"chain_if_negated_{branch_idx}_{chain_idx}_{block_idx}_{signal_idx}")
                                                with col3:
                                                    if len(chain_block['signals']) > 1 and signal_idx < len(chain_block['signals']) - 1:
                                                        signal_config['operator'] = st.selectbox(
                                                            "Operator",
                                                            ["AND", "OR"],
                                                            index=0 if signal_config.get('operator', 'AND') == 'AND' else 1,
                                                            key=f"chain_if_operator_{branch_idx}_{chain_idx}_{block_idx}_{signal_idx}"
                                                        )
                                                    else:
                                                        st.write("")
                                                with col4:
                                                    if len(chain_block['signals']) > 1:
                                                        if st.button("🗑️", key=f"remove_chain_if_signal_{branch_idx}_{chain_idx}_{block_idx}_{signal_idx}"):
                                                            chain_block['signals'].pop(signal_idx)
                                                            # Remove rerun to prevent state conflicts
                                                    else:
                                                        st.write("")
                                                
                                                st.markdown("<br>", unsafe_allow_html=True)
                                            
                                            # Add button to add more signals
                                            if st.button("➕ Add Another Signal", key=f"add_more_chain_signal_{branch_idx}_{chain_idx}_{block_idx}"):
                                                chain_block['signals'].append({
                                                    'signal': '', 
                                                    'negated': False, 
                                                    'operator': 'AND'
                                                })
                                                # Remove rerun to prevent state conflicts
                                    else:
                                        st.write("**No signals in chain IF yet**")
                                    
                                    st.markdown('</div>', unsafe_allow_html=True)
                                    
                                    # Chain THEN section - Streamlined format like primary blocks
                                    st.markdown('<div class="chain-then-block" style="border-left: 3px solid #FF9800; padding-left: 10px; margin-left: 0px;">', unsafe_allow_html=True)
                                    st.markdown("**CHAIN THEN:**")
                                    
                                    # Direct allocation selection (no + button needed)
                                    if st.session_state.output_allocations:
                                        selected_chain_allocation = st.selectbox(
                                            "Select Allocation:",
                                            [""] + list(st.session_state.output_allocations.keys()),
                                            key=f"chain_allocation_select_{branch_idx}_{chain_idx}_{block_idx}_{id(chain_block)}"
                                        )
                                        if selected_chain_allocation:
                                            if 'allocations' not in chain_block:
                                                chain_block['allocations'] = []
                                            # Check if allocation already exists
                                            if not any(a.get('allocation') == selected_chain_allocation for a in chain_block.get('allocations', [])):
                                                chain_block['allocations'].append({
                                                    'allocation': selected_chain_allocation, 
                                                    'weight': 100
                                                })
                                                st.success(f"✅ Allocation '{selected_chain_allocation}' added to chain!")
                                                # Remove rerun to prevent state conflicts
                                    else:
                                        st.warning("No allocations available. Create allocations in the Allocation Blocks tab first.")
                                    
                                    # Display chain THEN allocations
                                    if chain_block.get('allocations'):
                                        with st.expander(f"💰 Chain THEN Allocations ({len(chain_block['allocations'])})", expanded=True):
                                            for alloc_idx, allocation_config in enumerate(chain_block['allocations']):
                                                col1, col2, col3 = st.columns([2, 1, 1])
                                                with col1:
                                                    allocation_config['allocation'] = st.selectbox(
                                                        f"Chain Allocation {alloc_idx + 1}", 
                                                        list(st.session_state.output_allocations.keys()),
                                                        key=f"chain_then_allocation_{branch_idx}_{chain_idx}_{block_idx}_{alloc_idx}"
                                                    )
                                                with col2:
                                                    new_weight = st.number_input(
                                                        "Weight %",
                                                        min_value=0,
                                                        max_value=100,
                                                        value=allocation_config.get('weight', 100),
                                                        key=f"chain_then_weight_{branch_idx}_{chain_idx}_{block_idx}_{alloc_idx}"
                                                    )
                                                    # Only update if value actually changed to avoid unnecessary state changes
                                                    if new_weight != allocation_config.get('weight', 100):
                                                        allocation_config['weight'] = new_weight
                                                with col3:
                                                    if len(chain_block['allocations']) > 1:
                                                        if st.button("🗑️", key=f"remove_chain_{branch_idx}_{chain_idx}_{block_idx}_{alloc_idx}_{id(allocation_config)}_delete"):
                                                            chain_block['allocations'].pop(alloc_idx)
                                                            # Remove rerun to prevent state conflicts
                                                    else:
                                                        st.write("")
                                            
                                            # Show total weight for chain branch
                                            total_chain_weight = sum(alloc.get('weight', 0) for alloc in chain_block['allocations'])
                                            if total_chain_weight != 100:
                                                if total_chain_weight > 100:
                                                    st.error(f"⚠️ Chain total weight: {total_chain_weight}% (exceeds 100%)")
                                                else:
                                                    st.warning(f"ℹ️ Chain total weight: {total_chain_weight}% ({(100-total_chain_weight):.1f}% unallocated)")
                                            else:
                                                st.success(f"✅ Chain total weight: {total_chain_weight}%")
                                            
                                            # Add button to add more allocations
                                            if st.button("➕ Add Another Allocation", key=f"add_more_chain_allocation_{branch_idx}_{chain_idx}_{block_idx}"):
                                                chain_block['allocations'].append({
                                                    'allocation': '', 
                                                    'weight': 100
                                                })
                                                # Remove rerun to prevent state conflicts
                                    else:
                                        st.write("**No allocations in chain THEN yet**")
                                    
                                    st.markdown('</div>', unsafe_allow_html=True)
                                    
                                    # Add button to add next block in chain
                                    if st.button("➕ Add Next Block in Chain", key=f"add_next_chain_block_{branch_idx}_{chain_idx}_{block_idx}"):
                                        chain['chain_blocks'].append({
                                            'type': 'chain_if_else',
                                            'signals': [],
                                            'allocations': [],
                                            'else_allocations': [],
                                            'else_signals': [],
                                            'else_nested_blocks': [],
                                            'is_else_if': True
                                        })
                                        # Remove rerun to prevent state conflicts
                                    
                                    # Delete chain block
                                    if len(chain['chain_blocks']) > 1:
                                        if st.button("🗑️ Delete Chain Block", key=f"delete_chain_block_{branch_idx}_{chain_idx}_{block_idx}"):
                                            chain['chain_blocks'].pop(block_idx)
                                            # Remove rerun to prevent state conflicts
                                    
                                    st.markdown("</div>", unsafe_allow_html=True)
                                
                                # Add ELSE block at the end of each chain (like primary blocks)
                                st.markdown('<div class="chain-else-block" style="border-left: 3px solid #4CAF50; padding-left: 10px; margin-left: 10px;">', unsafe_allow_html=True)
                                st.markdown("**CHAIN ELSE:**")
                                
                                # Weight distribution between allocations and chains for the ELSE block
                                st.markdown("**Chain ELSE Weight Distribution:**")
                                col1, col2 = st.columns(2)
                                with col1:
                                    if 'chain_else_allocation_weight' not in chain:
                                        chain['chain_else_allocation_weight'] = 50
                                    new_chain_else_allocation_weight = st.number_input(
                                        "Allocation Weight %",
                                        min_value=0,
                                        max_value=100,
                                        value=chain.get('chain_else_allocation_weight', 50),
                                        key=f"chain_else_allocation_weight_{branch_idx}_{chain_idx}"
                                    )
                                    # Only update if value actually changed to avoid unnecessary reruns
                                    if new_chain_else_allocation_weight != chain.get('chain_else_allocation_weight', 50):
                                        chain['chain_else_allocation_weight'] = new_chain_else_allocation_weight
                                with col2:
                                    if 'chain_else_chain_weight' not in chain:
                                        chain['chain_else_chain_weight'] = 50
                                    new_chain_else_chain_weight = st.number_input(
                                        "Chain Weight %",
                                        min_value=0,
                                        max_value=100,
                                        value=chain.get('chain_else_chain_weight', 50),
                                        key=f"chain_else_chain_weight_{branch_idx}_{chain_idx}"
                                    )
                                    # Only update if value actually changed to avoid unnecessary reruns
                                    if new_chain_else_chain_weight != chain.get('chain_else_chain_weight', 50):
                                        chain['chain_else_chain_weight'] = new_chain_else_chain_weight
                                
                                # Validate total weight
                                total_chain_else_weight = chain.get('chain_else_allocation_weight', 0) + chain.get('chain_else_chain_weight', 0)
                                if total_chain_else_weight != 100:
                                    if total_chain_else_weight > 100:
                                        st.error(f"⚠️ Chain ELSE total weight: {total_chain_else_weight}% (exceeds 100%)")
                                    else:
                                        st.warning(f"ℹ️ Chain ELSE total weight: {total_chain_else_weight}% ({(100-total_chain_else_weight):.1f}% unallocated)")
                                else:
                                    st.success(f"✅ Chain ELSE total weight: {total_chain_else_weight}%")
                                
                                # Direct allocation selection for chain ELSE
                                st.markdown("**Add Allocation to Chain ELSE:**")
                                if st.session_state.output_allocations:
                                    selected_chain_else_allocation = st.selectbox(
                                        "Select Allocation:",
                                        [""] + list(st.session_state.output_allocations.keys()),
                                        key=f"chain_else_allocation_select_{branch_idx}_{chain_idx}_{id(chain)}"
                                    )
                                    if selected_chain_else_allocation:
                                        if 'chain_else_allocations' not in chain:
                                            chain['chain_else_allocations'] = []
                                        # Check if allocation already exists
                                        if not any(a.get('allocation') == selected_chain_else_allocation for a in chain.get('chain_else_allocations', [])):
                                            chain['chain_else_allocations'].append({
                                                'allocation': selected_chain_else_allocation, 
                                                'weight': 100
                                            })
                                            st.success(f"✅ Allocation '{selected_chain_else_allocation}' added to chain ELSE!")
                                            # Remove rerun to prevent state conflicts
                                else:
                                    st.warning("No allocations available. Create allocations in the Allocation Blocks tab first.")
                                
                                # Display chain ELSE allocations
                                if chain.get('chain_else_allocations'):
                                    with st.expander(f"💰 Chain ELSE Allocations ({len(chain['chain_else_allocations'])})", expanded=True):
                                        for else_alloc_idx, else_allocation_config in enumerate(chain['chain_else_allocations']):
                                            col1, col2, col3 = st.columns([2, 1, 1])
                                            with col1:
                                                else_allocation_config['allocation'] = st.selectbox(
                                                    f"Chain ELSE Allocation {else_alloc_idx + 1}", 
                                                    list(st.session_state.output_allocations.keys()),
                                                    key=f"chain_else_branch_{branch_idx}_{chain_idx}_allocation_{else_alloc_idx}"
                                                )
                                            with col2:
                                                new_weight = st.number_input(
                                                    "Weight %",
                                                    min_value=0,
                                                    max_value=100,
                                                    value=else_allocation_config.get('weight', 100),
                                                    key=f"chain_else_branch_{branch_idx}_{chain_idx}_weight_{else_alloc_idx}"
                                                )
                                                # Only update if value actually changed to avoid unnecessary state changes
                                                if new_weight != else_allocation_config.get('weight', 100):
                                                    else_allocation_config['weight'] = new_weight
                                            with col3:
                                                if len(chain['chain_else_allocations']) > 1:
                                                    if st.button("🗑️", key=f"remove_chain_else_{branch_idx}_{chain_idx}_{else_alloc_idx}_{id(else_allocation_config)}_delete"):
                                                        chain['chain_else_allocations'].pop(else_alloc_idx)
                                                        # Remove rerun to prevent state conflicts
                                                else:
                                                    st.write("")
                                        
                                        # Show total chain ELSE allocation weight
                                        total_chain_else_allocation_weight = sum(alloc.get('weight', 0) for alloc in chain['chain_else_allocations'])
                                        if total_chain_else_allocation_weight != 100:
                                            if total_chain_else_allocation_weight > 100:
                                                st.error(f"⚠️ Chain ELSE allocation weight: {total_chain_else_allocation_weight}% (exceeds 100%)")
                                            else:
                                                st.warning(f"ℹ️ Chain ELSE allocation weight: {total_chain_else_allocation_weight}% ({(100-total_chain_else_allocation_weight):.1f}% unallocated)")
                                        else:
                                            st.success(f"✅ Chain ELSE allocation weight: {total_chain_else_allocation_weight}%")
                                        
                                        # Show weight distribution info
                                        allocation_weight = chain.get('chain_else_allocation_weight', 50)
                                        chain_weight = chain.get('chain_else_chain_weight', 50)
                                        st.info(f"📊 Chain ELSE Distribution: {allocation_weight}% allocations, {chain_weight}% chains")
                                        
                                        # Add button to add more allocations to chain ELSE
                                        if st.button("➕ Add Another Allocation to Chain ELSE", key=f"add_more_chain_else_allocation_{branch_idx}_{chain_idx}_{id(chain)}"):
                                            chain['chain_else_allocations'].append({
                                                'allocation': '', 
                                                'weight': 100
                                            })
                                            # Remove rerun to prevent state conflicts
                                
                                # Add nested chain button for chain ELSE
                                if st.button("🔗 Add Nested Chain to Chain ELSE", key=f"add_nested_chain_to_chain_else_{branch_idx}_{chain_idx}"):
                                    if 'chain_else_nested_chains' not in chain:
                                        chain['chain_else_nested_chains'] = []
                                    chain['chain_else_nested_chains'].append({
                                        'type': 'nested_if_else_chain',
                                        'chain_blocks': [{
                                            'type': 'chain_if_else',
                                            'signals': [],
                                            'allocations': [],
                                            'else_allocations': [],
                                            'else_signals': [],
                                            'else_nested_blocks': [],
                                            'is_else_if': False
                                        }]
                                    })
                                    st.success(f"✅ Nested chain added to chain ELSE!")
                                    # Remove rerun to prevent state conflicts
                                
                                st.markdown('</div>', unsafe_allow_html=True)
                                
                                # Delete entire chain
                                if st.button("🗑️ Delete Entire Chain", key=f"delete_chain_{branch_idx}_{chain_idx}"):
                                    branch['else_nested_chains'].pop(chain_idx)
                                    # Remove rerun to prevent state conflicts
                                
                                st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Show status if no allocations, nested blocks, or chains
                    if not branch.get('else_allocations') and not branch.get('else_nested_blocks') and not branch.get('else_nested_chains'):
                        st.write("**No allocations, nested blocks, or chains in ELSE yet**")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
            
            else:
                # Regular branch container
                total_branch_weight = sum(alloc.get('weight', 0) for alloc in branch.get('allocations', [{'weight': 100}]))
                st.markdown(f"""
                <div class="branch-container">
                    <div class="branch-header">
                        <span>🎯 Branch {branch_idx + 1}</span>
                        <span>Total Weight: {total_branch_weight}%</span>
                    </div>
                """, unsafe_allow_html=True)
                
                # Block operations menu
                col_menu, col_spacer = st.columns([1, 9])
                with col_menu:
                    block_menu_action = st.selectbox(
                        '',
                        ["", "Copy", "Paste", "Delete", "Save as Reference"],
                        key=f"branch_menu_{branch_idx}",
                        label_visibility="collapsed"
                    )
                    if block_menu_action == "Copy":
                        st.session_state.copied_block = copy.deepcopy(branch)
                        st.success("Block copied!")
                    elif block_menu_action == "Paste":
                        if st.session_state.copied_block:
                            st.session_state.strategy_branches.insert(branch_idx+1, copy.deepcopy(st.session_state.copied_block))
                            st.success("Block pasted!")
                            st.rerun()
                        else:
                            st.warning("⚠️ No block in clipboard")
                    elif block_menu_action == "Delete":
                        st.session_state.strategy_branches.pop(branch_idx)
                        st.rerun()
                    elif block_menu_action == "Save as Reference":
                        ref_name = st.text_input("Reference Block Name:", key=f"ref_name_{branch_idx}")
                        if st.button("💾 Save", key=f"save_ref_{branch_idx}"):
                            if ref_name:
                                save_reference_block(branch, ref_name)
                                st.success(f"✅ Block saved as '{ref_name}'!")
                                st.rerun()
                            else:
                                st.error("Please provide a name.")
                
                # Simple add button for the branch - sleeker interface
                if st.button("➕", key=f"add_to_branch_{branch_idx}"):
                    # Show options in a popup-like interface
                    st.markdown("**Add to Branch:**")
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        if st.button("📊 Add Signal", key=f"add_branch_signal_{branch_idx}"):
                            if 'signals' not in branch:
                                branch['signals'] = []
                            branch['signals'].append({'signal': '', 'negated': False, 'operator': 'AND'})
                            st.rerun()
                    with col_b:
                        if st.button("💰 Add Allocation", key=f"add_branch_allocation_{branch_idx}"):
                            if 'allocations' not in branch:
                                branch['allocations'] = []
                            branch['allocations'].append({'allocation': '', 'weight': 100})
                            st.rerun()
                    with col_c:
                        if st.button("📋 Add Block", key=f"add_branch_block_{branch_idx}"):
                            st.info("Add Block functionality coming soon!")
                
                # Display existing signals
                if branch.get('signals'):
                    st.markdown('<div class="condition-block">', unsafe_allow_html=True)
                    for signal_idx, signal_config in enumerate(branch['signals']):
                        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                        
                        with col1:
                            signal_config['signal'] = st.selectbox(
                                f"Signal {signal_idx + 1}", 
                                [""] + [s['name'] for s in st.session_state.signals], 
                                key=f"branch_{branch_idx}_signal_{signal_idx}"
                            )
                        
                        with col2:
                            signal_config['negated'] = st.checkbox("NOT", key=f"branch_{branch_idx}_negated_{signal_idx}")
                        
                        with col3:
                            if signal_idx > 0:  # Don't show operator for first signal
                                signal_config['operator'] = st.selectbox(
                                    "Logic", 
                                    ["AND", "OR"], 
                                    key=f"branch_{branch_idx}_operator_{signal_idx}"
                                )
                            else:
                                st.write("")  # Empty space for alignment
                        
                        with col4:
                            if st.button("🗑️", key=f"remove_branch_{branch_idx}_signal_{signal_idx}"):
                                branch['signals'].pop(signal_idx)
                                st.rerun()
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Display existing allocations
                if branch.get('allocations'):
                    st.markdown('<div class="then-block">', unsafe_allow_html=True)
                    
                    for alloc_idx, allocation_config in enumerate(branch['allocations']):
                        col1, col2, col3 = st.columns([2, 1, 1])
                        with col1:
                            allocation_config['allocation'] = st.selectbox(
                                f"Allocation {alloc_idx + 1}", 
                                list(st.session_state.output_allocations.keys()),
                                key=f"branch_{branch_idx}_allocation_{alloc_idx}"
                            )
                        with col2:
                            allocation_config['weight'] = st.number_input(
                                "Weight %",
                                min_value=0,
                                max_value=100,
                                value=allocation_config.get('weight', 100),
                                key=f"branch_{branch_idx}_weight_{alloc_idx}"
                            )
                        with col3:
                            if len(branch['allocations']) > 1:  # Don't allow removing the last allocation
                                if st.button("🗑️", key=f"remove_regular_{branch_idx}_{alloc_idx}_{id(allocation_config)}_delete"):
                                    branch['allocations'].pop(alloc_idx)
                                    st.rerun()
                            else:
                                st.write("")  # Empty space for alignment
                    
                    # Show total weight for this branch
                    total_branch_weight = sum(alloc.get('weight', 0) for alloc in branch['allocations'])
                    if total_branch_weight != 100:
                        if total_branch_weight > 100:
                            st.error(f"⚠️ Branch total weight: {total_branch_weight}% (exceeds 100%)")
                        else:
                            st.warning(f"ℹ️ Branch total weight: {total_branch_weight}% ({(100-total_branch_weight):.1f}% unallocated)")
                    else:
                        st.success(f"✅ Branch total weight: {total_branch_weight}%")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Delete entire branch
                if st.button("🗑️ Delete Branch", key=f"delete_branch_{branch_idx}"):
                    st.session_state.strategy_branches.pop(branch_idx)
                    st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

# Tab 4: Backtest
with tab4:
    st.header("📈 Backtest")
    
    # Backtest configuration
    col1, col2, col3 = st.columns(3)
    
    with col1:
        start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=365))
    with col2:
        end_date = st.date_input("End Date", value=datetime.now())
    with col3:
        benchmark_ticker = st.text_input("Benchmark Ticker", value="SPY")
    
    # Run backtest button
    if st.button("🚀 Run Backtest", type="primary"):
        if not st.session_state.strategy_branches:
            st.error("Please create at least one strategy branch first.")
        elif not st.session_state.output_allocations:
            st.error("Please create at least one allocation block first.")
        else:
            with st.spinner("Running backtest..."):
                try:
                    # Collect all tickers needed
                    all_tickers = set()
                    
                    # Add benchmark ticker
                    all_tickers.add(benchmark_ticker)
                    
                    # Add tickers from allocations
                    for allocation in st.session_state.output_allocations.values():
                        for ticker_config in allocation['tickers']:
                            all_tickers.add(ticker_config['ticker'])
                    
                    # Add tickers from signals
                    for signal in st.session_state.signals:
                        if signal['type'] == 'Custom Indicator':
                            all_tickers.add(signal['signal_ticker1'])
                            all_tickers.add(signal['signal_ticker2'])
                        elif signal['type'] == 'Static RSI':
                            all_tickers.add(signal['signal_ticker'])
                        elif signal['type'] == 'RSI Comparison':
                            all_tickers.add(signal['signal_ticker'])
                            all_tickers.add(signal['comparison_ticker'])
                    
                    # Fetch data for all tickers
                    data = {}
                    for ticker in all_tickers:
                        ticker_data = get_stock_data(ticker, start_date, end_date)
                        if not ticker_data.empty:
                            data[ticker] = ticker_data
                    
                    if not data:
                        st.error("No data available for the specified tickers and date range.")
                        st.stop()
                    
                    # Calculate signals
                    signal_results = {}
                    for signal in st.session_state.signals:
                        signal_name = signal['name']
                        
                        if signal['type'] == 'Custom Indicator':
                            # Get indicator values
                            ticker1_data = data.get(signal['signal_ticker1'])
                            ticker2_data = data.get(signal['signal_ticker2'])
                            
                            if ticker1_data is not None and ticker2_data is not None:
                                indicator1_values = calculate_indicator(ticker1_data, signal['indicator1'], signal['days1'])
                                indicator2_values = calculate_indicator(ticker2_data, signal['indicator2'], signal['days2'])
                                
                                # Evaluate condition
                                signal_results[signal_name] = evaluate_signal_condition(
                                    indicator1_values, indicator2_values, signal['operator']
                                )
                        
                        elif signal['type'] == 'Static RSI':
                            ticker_data = data.get(signal['signal_ticker'])
                            if ticker_data is not None:
                                rsi_values = calculate_rsi(ticker_data, signal['rsi_period'])
                                if signal['comparison'] == 'less_than':
                                    signal_results[signal_name] = rsi_values < signal['rsi_threshold']
                                else:
                                    signal_results[signal_name] = rsi_values > signal['rsi_threshold']
                        
                        elif signal['type'] == 'RSI Comparison':
                            ticker1_data = data.get(signal['signal_ticker'])
                            ticker2_data = data.get(signal['comparison_ticker'])
                            
                            if ticker1_data is not None and ticker2_data is not None:
                                rsi1_values = calculate_rsi(ticker1_data, signal['rsi_period'])
                                rsi2_values = calculate_rsi(ticker2_data, signal['rsi_period'])
                                
                                if signal['comparison_operator'] == 'less_than':
                                    signal_results[signal_name] = rsi1_values < rsi2_values
                                else:
                                    signal_results[signal_name] = rsi1_values > rsi2_values
                    
                    # Calculate strategy signals
                    strategy_signals = pd.Series(False, index=list(data.values())[0].index)
                    
                    for branch in st.session_state.strategy_branches:
                        if branch.get('type') == 'if_else':
                            # Handle If/Else logic
                            if_signals = branch.get('signals', [])
                            if_allocations = branch.get('allocations', [])
                            else_allocations = branch.get('else_allocations', [])
                            
                            # Evaluate IF conditions
                            if_result = pd.Series(True, index=strategy_signals.index)
                            for signal_config in if_signals:
                                signal_name = signal_config.get('signal', '')
                                if signal_name and signal_name in signal_results:
                                    signal_result = signal_results[signal_name]
                                    if signal_config.get('negated', False):
                                        signal_result = ~signal_result
                                    
                                    if signal_config.get('operator', 'AND') == 'AND':
                                        if_result = if_result & signal_result
                                    else:
                                        if_result = if_result | signal_result
                            
                            # Apply allocations based on IF result
                            # Check if IF condition is met (any True values in the series)
                            if_condition_met = if_result.any()
                            
                            if if_condition_met:
                                # Use IF allocations when IF condition is met
                                for alloc_config in if_allocations:
                                    allocation_name = alloc_config.get('allocation', '')
                                    if allocation_name in st.session_state.output_allocations:
                                        allocation = st.session_state.output_allocations[allocation_name]
                                        weight = alloc_config.get('weight', 100) / 100.0
                                        
                                        # Calculate equity curve for this allocation
                                        alloc_equity = calculate_multi_ticker_equity_curve(
                                            if_result, allocation, data
                                        )
                                
                                # Add IF signals to strategy
                                strategy_signals = strategy_signals | if_result
                            else:
                                # Use ELSE allocations and chains only when IF condition is NOT met
                                else_allocation_weight = branch.get('else_allocation_weight', 50) / 100.0
                                else_chain_weight = branch.get('else_chain_weight', 50) / 100.0
                                
                                # Create ELSE signals (inverse of IF signals)
                                else_signals = ~if_result
                                
                                # Process ELSE allocations with proper weight isolation
                                if else_allocations and else_allocation_weight > 0:
                                    for alloc_config in else_allocations:
                                        allocation_name = alloc_config.get('allocation', '')
                                        if allocation_name in st.session_state.output_allocations:
                                            allocation = st.session_state.output_allocations[allocation_name]
                                            weight = (alloc_config.get('weight', 100) / 100.0) * else_allocation_weight
                                            
                                            # Calculate equity curve for this allocation using ELSE signals
                                            alloc_equity = calculate_multi_ticker_equity_curve(
                                                else_signals, allocation, data
                                            )
                                
                                # Process ELSE chains (nested IF/THEN/ELSE chains) with proper isolation
                                if branch.get('else_nested_chains') and else_chain_weight > 0:
                                    for chain in branch['else_nested_chains']:
                                        if chain['type'] == 'nested_if_else_chain':
                                            for chain_block in chain.get('chain_blocks', []):
                                                # Process each chain block (simplified - would need more complex logic)
                                                # This is a placeholder for the nested chain processing
                                                pass
                                
                                # Add ELSE signals to strategy (only when IF condition is not met)
                                strategy_signals = strategy_signals | else_signals
                        else:
                            # Regular branch logic
                            branch_signals = pd.Series(True, index=strategy_signals.index)
                            
                            for signal_config in branch.get('signals', []):
                                signal_name = signal_config.get('signal', '')
                                if signal_name and signal_name in signal_results:
                                    signal_result = signal_results[signal_name]
                                    if signal_config.get('negated', False):
                                        signal_result = ~signal_result
                                    
                                    if signal_config.get('operator', 'AND') == 'AND':
                                        branch_signals = branch_signals & signal_result
                                    else:
                                        branch_signals = branch_signals | signal_result
                            
                            # Apply allocations
                            for alloc_config in branch.get('allocations', []):
                                allocation_name = alloc_config.get('allocation', '')
                                if allocation_name in st.session_state.output_allocations:
                                    allocation = st.session_state.output_allocations[allocation_name]
                                    weight = alloc_config.get('weight', 100) / 100.0
                                    
                                    # Calculate equity curve for this allocation
                                    alloc_equity = calculate_multi_ticker_equity_curve(
                                        branch_signals, allocation, data
                                    )
                                    
                                    # Add to strategy signals
                                    strategy_signals = strategy_signals | branch_signals
                    
                    # Calculate benchmark equity curve
                    benchmark_data = data.get(benchmark_ticker)
                    if benchmark_data is not None:
                        benchmark_equity = calculate_equity_curve(strategy_signals, benchmark_data)
                        benchmark_returns = benchmark_data.pct_change()
                        benchmark_metrics = calculate_metrics(benchmark_equity, benchmark_returns)
                        
                        # Display results
                        st.subheader("📊 Backtest Results")
                        
                        # Performance metrics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Total Return", f"{benchmark_metrics.get('total_return', 0):.2%}")
                            st.metric("Annualized Return", f"{benchmark_metrics.get('annualized_return', 0):.2%}")
                        
                        with col2:
                            st.metric("Sharpe Ratio", f"{benchmark_metrics.get('sharpe_ratio', 0):.2f}")
                            st.metric("Sortino Ratio", f"{benchmark_metrics.get('sortino_ratio', 0):.2f}")
                        
                        with col3:
                            st.metric("Max Drawdown", f"{benchmark_metrics.get('max_drawdown', 0):.2%}")
                            st.metric("Calmar Ratio", f"{benchmark_metrics.get('calmar_ratio', 0):.2f}")
                        
                        with col4:
                            st.metric("Win Rate", f"{benchmark_metrics.get('win_rate', 0):.2%}")
                            st.metric("Total Trades", benchmark_metrics.get('total_trades', 0))
                        
                        # Equity curve chart
                        st.subheader("📈 Equity Curve")
                        
                        fig = go.Figure()
                        
                        # Strategy equity curve
                        fig.add_trace(go.Scatter(
                            x=benchmark_equity.index,
                            y=benchmark_equity.values,
                            mode='lines',
                            name=f'Strategy → {benchmark_ticker}',
                            line=dict(color='blue', width=2)
                        ))
                        
                        # Benchmark equity curve
                        benchmark_buy_hold = (1 + benchmark_returns).cumprod()
                        fig.add_trace(go.Scatter(
                            x=benchmark_buy_hold.index,
                            y=benchmark_buy_hold.values,
                            mode='lines',
                            name=f'{benchmark_ticker} Buy & Hold',
                            line=dict(color='gray', width=1, dash='dash')
                        ))
                        
                        fig.update_layout(
                            title="Strategy vs Benchmark Performance",
                            xaxis_title="Date",
                            yaxis_title="Equity",
                            hovermode='x unified',
                            template='plotly_white'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Cache statistics
                        cache_stats = get_cache_stats()
                        st.info(f"💾 Cache Performance: {cache_stats['hits']} hits, {cache_stats['misses']} misses ({cache_stats['hit_rate']:.1f}% hit rate)")
                    
                    else:
                        st.error(f"No data available for benchmark ticker: {benchmark_ticker}")
                
                except Exception as e:
                    st.error(f"Error during backtest: {str(e)}")
                    st.exception(e)
    else:
        st.info("Click 'Run Backtest' to test your strategy against historical data.") 
