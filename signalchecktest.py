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
    page_title="Tactical Signal Validator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .signal-card {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e9ecef;
        border-radius: 6px;
        padding: 0.75rem;
        margin: 0.25rem 0;
    }
    .positive-metric {
        color: #28a745;
        font-weight: bold;
    }
    .negative-metric {
        color: #dc3545;
        font-weight: bold;
    }
    .neutral-metric {
        color: #6c757d;
        font-weight: bold;
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

# Helper functions (keeping existing ones)
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
        # Fetch data
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date)
        
        if data.empty:
            return pd.Series(dtype=float)
        
        # Normalize timezone to naive timestamps
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
            # Enter position
            in_position = True
            entry_equity = current_equity
            entry_price = current_price
            
        elif current_signal == 0 and in_position:
            # Exit position
            trade_return = (current_price - entry_price) / entry_price
            current_equity = entry_equity * (1 + trade_return * allocation)
            in_position = False
        
        # Update equity curve
        if in_position:
            # Mark-to-market the position
            current_equity = entry_equity * (current_price / entry_price)
        
        equity_curve[date] = current_equity
    
    # Handle case where we're still in position at the end
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
    
    # Sortino ratio (downside deviation)
    downside_returns = returns[returns < 0]
    downside_deviation = downside_returns.std() * np.sqrt(252) * 100 if len(downside_returns) > 0 else 0
    sortino_ratio = (annualized_return / 100) / (downside_deviation / 100) if downside_deviation > 0 else 0
    
    # Max drawdown
    rolling_max = equity_curve.expanding().max()
    drawdown = (equity_curve - rolling_max) / rolling_max
    max_drawdown = drawdown.min() * 100
    
    # Trade metrics
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

# Main app
st.markdown('<h1 class="main-header">📊 Tactical Signal Validator</h1>', unsafe_allow_html=True)

# Sidebar for signal management
with st.sidebar:
    st.header("🎯 Signal Management")
    
    # Signal creation
    with st.expander("➕ Add New Signal", expanded=True):
        signal_name = st.text_input("Signal Name", placeholder="e.g., QQQ RSI Oversold")
        signal_type = st.selectbox("Signal Type", ["RSI Threshold", "RSI Comparison", "Custom"])
        
        if signal_type == "RSI Threshold":
            signal_ticker = st.text_input("Signal Ticker", value="QQQ")
            target_ticker = st.text_input("Target Ticker", value="SPY")
            rsi_period = st.number_input("RSI Period", min_value=1, max_value=50, value=14)
            rsi_threshold = st.number_input("RSI Threshold", min_value=0.0, max_value=100.0, value=30.0, step=0.5)
            comparison = st.selectbox("Condition", ["less_than", "greater_than"], 
                                   format_func=lambda x: "RSI ≤ threshold" if x == "less_than" else "RSI ≥ threshold")
            
            if st.button("Add Signal"):
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
            signal_ticker = st.text_input("Signal Ticker", value="QQQ")
            comparison_ticker = st.text_input("Comparison Ticker", value="SPY")
            target_ticker = st.text_input("Target Ticker", value="TQQQ")
            rsi_period = st.number_input("RSI Period", min_value=1, max_value=50, value=14)
            comparison_operator = st.selectbox("Comparison", ["less_than", "greater_than"],
                                            format_func=lambda x: "Signal RSI < Comparison RSI" if x == "less_than" else "Signal RSI > Comparison RSI")
            
            if st.button("Add Signal"):
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
                    if signal['type'] == "RSI Threshold":
                        st.caption(f"{signal['signal_ticker']} RSI {signal['rsi_period']}-day {signal['comparison']} {signal['rsi_threshold']} → {signal['target_ticker']}")
                    else:
                        st.caption(f"{signal['signal_ticker']} vs {signal['comparison_ticker']} RSI {signal['comparison_operator']} → {signal['target_ticker']}")
                with col2:
                    if st.button("🗑️", key=f"delete_{i}"):
                        st.session_state.signals.pop(i)
                        st.rerun()
    
    # Allocation management
    st.header("💰 Allocation Management")
    if st.session_state.signals:
        total_allocation = 0
        for signal in st.session_state.signals:
            allocation = st.slider(f"{signal['name']} Allocation (%)", 
                                 min_value=0, max_value=100, value=20, key=f"alloc_{signal['name']}")
            st.session_state.allocations[signal['name']] = allocation / 100
            total_allocation += allocation
        
        if total_allocation > 100:
            st.error(f"⚠️ Total allocation exceeds 100% ({total_allocation}%)")
        elif total_allocation < 100:
            st.warning(f"ℹ️ Total allocation: {total_allocation}% ({(100-total_allocation)}% in cash)")
        else:
            st.success(f"✅ Total allocation: {total_allocation}%")

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
                    all_tickers.add(signal['signal_ticker'])
                    all_tickers.add(signal['target_ticker'])
                    if signal['type'] == "RSI Comparison":
                        all_tickers.add(signal['comparison_ticker'])
                all_tickers.add(benchmark_ticker)
                
                # Fetch data
                data = {}
                for ticker in all_tickers:
                    data[ticker] = get_stock_data(ticker, start_date, end_date)
                
                # Calculate signals and equity curves
                results = {}
                combined_equity = pd.Series(1.0, index=data[benchmark_ticker].index)
                
                for signal in st.session_state.signals:
                    if signal['type'] == "RSI Threshold":
                        # Calculate RSI
                        rsi = calculate_rsi(data[signal['signal_ticker']], signal['rsi_period'])
                        
                        # Generate signals
                        if signal['comparison'] == "less_than":
                            signals = (rsi <= signal['rsi_threshold']).astype(int)
                        else:
                            signals = (rsi >= signal['rsi_threshold']).astype(int)
                        
                        # Calculate equity curve
                        equity_curve = calculate_equity_curve(signals, data[signal['target_ticker']], 
                                                           st.session_state.allocations[signal['name']])
                        
                        # Calculate metrics
                        returns = equity_curve.pct_change().dropna()
                        metrics = calculate_metrics(equity_curve, returns)
                        
                        results[signal['name']] = {
                            'equity_curve': equity_curve,
                            'signals': signals,
                            'metrics': metrics
                        }
                        
                        # Add to combined portfolio
                        combined_equity = combined_equity * (1 + (equity_curve - 1) * st.session_state.allocations[signal['name']])
                    
                    elif signal['type'] == "RSI Comparison":
                        # Calculate RSI for both tickers
                        signal_rsi = calculate_rsi(data[signal['signal_ticker']], signal['rsi_period'])
                        comparison_rsi = calculate_rsi(data[signal['comparison_ticker']], signal['rsi_period'])
                        
                        # Generate signals
                        if signal['comparison_operator'] == "less_than":
                            signals = (signal_rsi < comparison_rsi).astype(int)
                        else:
                            signals = (signal_rsi > comparison_rsi).astype(int)
                        
                        # Calculate equity curve
                        equity_curve = calculate_equity_curve(signals, data[signal['target_ticker']], 
                                                           st.session_state.allocations[signal['name']])
                        
                        # Calculate metrics
                        returns = equity_curve.pct_change().dropna()
                        metrics = calculate_metrics(equity_curve, returns)
                        
                        results[signal['name']] = {
                            'equity_curve': equity_curve,
                            'signals': signals,
                            'metrics': metrics
                        }
                        
                        # Add to combined portfolio
                        combined_equity = combined_equity * (1 + (equity_curve - 1) * st.session_state.allocations[signal['name']])
                
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

# Results display
if st.session_state.backtest_results:
    st.header("📈 Backtest Results")
    
    # Equity curve comparison
    fig = go.Figure()
    
    # Add benchmark
    fig.add_trace(go.Scatter(
        x=st.session_state.backtest_results['benchmark_equity'].index,
        y=st.session_state.backtest_results['benchmark_equity'].values,
        mode='lines',
        name='Benchmark',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    # Add combined portfolio
    fig.add_trace(go.Scatter(
        x=st.session_state.backtest_results['combined_equity'].index,
        y=st.session_state.backtest_results['combined_equity'].values,
        mode='lines',
        name='Portfolio',
        line=dict(color='blue', width=3)
    ))
    
    # Add individual signals
    colors = ['green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive']
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
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Signal analysis table
    st.subheader("📊 Signal Performance Analysis")
    
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
    
    # Individual signal analysis
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
                hole=0.3
            )])
            fig_pie.update_layout(title="Signal Distribution")
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
            line=dict(color='blue', width=2)
        ))
        
        # Add benchmark
        fig_signal.add_trace(go.Scatter(
            x=st.session_state.backtest_results['benchmark_equity'].index,
            y=st.session_state.backtest_results['benchmark_equity'].values,
            mode='lines',
            name='Benchmark',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        fig_signal.update_layout(
            title=f"{selected_signal} vs Benchmark",
            xaxis_title="Date",
            yaxis_title="Equity Value",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_signal, use_container_width=True)

# Footer
st.write("---")
st.markdown("""
<div style='text-align: center; padding: 20px; color: #666;'>
    <strong>Tactical Signal Validator</strong><br>
    Built for sophisticated signal analysis and portfolio optimization
</div>
""", unsafe_allow_html=True)
