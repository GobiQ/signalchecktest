import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, Tuple, List, Optional
import warnings
from scipy import stats
warnings.filterwarnings('ignore')

# Try to import QuantStats with error handling
try:
    import quantstats as qs
    # Configure QuantStats
    qs.extend_pandas()
    QUANTSTATS_AVAILABLE = True
    st.success("✅ QuantStats loaded successfully!")
except ImportError as e:
    st.warning(f"⚠️ QuantStats import error: {str(e)}. Install with: pip install quantstats>=0.0.62")
    QUANTSTATS_AVAILABLE = False
except Exception as e:
    st.warning(f"⚠️ QuantStats import failed: {str(e)}. Using fallback calculations.")
    QUANTSTATS_AVAILABLE = False

st.set_page_config(page_title="Signal Check", layout="wide")

st.title("Signal Check")
st.write("RSI Threshold Statistics")

def calculate_rsi(prices: pd.Series, window: int = 14, method: str = "wilders") -> pd.Series:
    """Calculate RSI using specified method (Wilder's smoothing or simple moving average)"""
    if len(prices) < window + 1:
        return pd.Series(index=prices.index, dtype=float)
    
    delta = prices.diff()
    
    # Separate gains and losses
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)
    
    if method == "wilders":
        # Wilder's smoothing: use exponential moving average with alpha = 1/window
        alpha = 1.0 / window
        
        # Calculate smoothed average gains and losses
        avg_gains = gains.ewm(alpha=alpha, adjust=False).mean()
        avg_losses = losses.ewm(alpha=alpha, adjust=False).mean()
    else:
        # Simple moving average method
        avg_gains = gains.rolling(window=window).mean()
        avg_losses = losses.rolling(window=window).mean()
    
    # Calculate relative strength
    rs = avg_gains / avg_losses
    
    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.02, use_quantstats: bool = True) -> float:
    """Calculate Sortino ratio using QuantStats or fallback"""
    if len(returns) == 0:
        return 0
    
    # Convert to pandas Series for QuantStats
    returns_series = pd.Series(returns)
    
    if QUANTSTATS_AVAILABLE and use_quantstats:
        try:
            # Use QuantStats sortino ratio calculation
            sortino_ratio = qs.stats.sortino(returns_series, rf=risk_free_rate)
            return sortino_ratio if not np.isnan(sortino_ratio) else 0
        except Exception:
            pass  # Fall through to original calculation
    
    # Fallback to original calculation
    rf_per_trade = risk_free_rate / 252
    excess_returns = returns - rf_per_trade
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0:
        return np.inf if excess_returns.mean() > 0 else 0
    
    downside_deviation = np.sqrt(np.mean(downside_returns**2))
    
    if downside_deviation == 0:
        return np.inf if excess_returns.mean() > 0 else 0
    
    return excess_returns.mean() / downside_deviation

def get_stock_data(ticker: str, start_date=None, end_date=None, exclusions=None) -> pd.Series:
    """Fetch stock data using yfinance with optional date range and exclusions"""
    try:
        stock = yf.Ticker(ticker)
        
        if start_date and end_date:
            data = stock.history(start=start_date, end=end_date)
        else:
            # Default to maximum available period
            data = stock.history(period="max")
        
        if data.empty:
            st.error(f"No data found for ticker: {ticker}")
            return None
        
        # Normalize timezone information to avoid alignment issues
        if data.index.tz is not None:
            try:
                data.index = data.index.tz_localize(None)
            except Exception as e:
                # If tz_localize fails, try tz_convert to UTC then remove timezone
                try:
                    data.index = data.index.tz_convert('UTC').tz_localize(None)
                except Exception:
                    # If all else fails, convert to naive timestamps
                    data.index = pd.to_datetime(data.index).tz_localize(None)
        
        # Apply exclusions if provided
        if exclusions:
            for exclusion in exclusions:
                exclusion_start = pd.Timestamp(exclusion['start'])
                exclusion_end = pd.Timestamp(exclusion['end'])
                # Remove data within exclusion period
                data = data[~((data.index >= exclusion_start) & (data.index <= exclusion_end))]
        
        return data['Close']
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None

def resolve_fallback_ticker(fallback_option: str, main_signal_ticker: str, preconditions: List[Dict] = None, precondition_index: int = None) -> str:
    """
    Resolve the actual ticker symbol for a fallback option.
    
    Args:
        fallback_option: The fallback option (e.g., "Main Signal", "Precondition 1 Signal", "BIL")
        main_signal_ticker: The main signal ticker
        preconditions: List of preconditions
        precondition_index: Index of current precondition (to avoid self-reference)
    
    Returns:
        The resolved ticker symbol
    """
    if fallback_option == "Main Signal":
        return main_signal_ticker
    elif fallback_option.startswith("Precondition"):
        # Extract precondition number (1-based)
        try:
            precondition_num = int(fallback_option.split()[1]) - 1  # Convert to 0-based index
            if preconditions and 0 <= precondition_num < len(preconditions) and precondition_num != precondition_index:
                return preconditions[precondition_num]['signal_ticker']
            else:
                return "BIL"  # Default fallback if invalid reference
        except (ValueError, IndexError):
            return "BIL"  # Default fallback if parsing fails
    else:
        return fallback_option  # Custom ticker

def resolve_fallback_signal(fallback_option: str, main_signal_output: pd.Series, precondition_outputs: Dict[str, pd.Series] = None, precondition_index: int = None) -> pd.Series:
    """
    Resolve the actual signal output for a fallback option.
    
    Args:
        fallback_option: The fallback option (e.g., "Main Signal", "Precondition 1 Signal", "BIL")
        main_signal_output: The main signal output series
        precondition_outputs: Dictionary of precondition signal outputs
        precondition_index: Index of current precondition (to avoid self-reference)
    
    Returns:
        The resolved signal output series
    """
    if fallback_option == "Main Signal":
        return main_signal_output
    elif fallback_option.startswith("Precondition"):
        # Extract precondition number (1-based)
        try:
            precondition_num = int(fallback_option.split()[1]) - 1  # Convert to 0-based index
            if precondition_outputs and f"Precondition {precondition_num + 1} Signal" in precondition_outputs and precondition_num != precondition_index:
                return precondition_outputs[f"Precondition {precondition_num + 1} Signal"]
            else:
                return pd.Series(0, index=main_signal_output.index)  # Default to no signal
        except (ValueError, IndexError):
            return pd.Series(0, index=main_signal_output.index)  # Default to no signal
    else:
        # For custom tickers, return zeros (no signal)
        return pd.Series(0, index=main_signal_output.index)

def get_signal_output(signal_ticker: str, comparison_ticker: str, signal_prices: pd.Series, comparison_prices: pd.Series, 
                     rsi_period: int, rsi_method: str, comparison_operator: str) -> pd.Series:
    """
    Get the signal output (buy/sell signals) for a given ticker comparison.
    
    Args:
        signal_ticker: The signal ticker
        comparison_ticker: The comparison ticker
        signal_prices: Price data for signal ticker
        comparison_prices: Price data for comparison ticker
        rsi_period: RSI period
        rsi_method: RSI calculation method
        comparison_operator: "less_than" or "greater_than"
    
    Returns:
        Series of buy signals (1 for buy, 0 for sell/hold)
    """
    signal_rsi = calculate_rsi(signal_prices, window=rsi_period, method=rsi_method)
    comparison_rsi = calculate_rsi(comparison_prices, window=rsi_period, method=rsi_method)
    
    if comparison_operator == "greater_than":
        return (signal_rsi > comparison_rsi).astype(int)
    else:  # less_than (default)
        return (signal_rsi < comparison_rsi).astype(int)

def analyze_rsi_comparison_signals(signal_prices: pd.Series, comparison_prices: pd.Series, target_prices: pd.Series, rsi_period: int = 14, rsi_method: str = "wilders", comparison_operator: str = "less_than", use_quantstats: bool = True, preconditions: List[Dict] = None, precondition_data: Dict[str, pd.Series] = None) -> Dict:
    """
    Analyze RSI comparison signals: when signal RSI < comparison RSI, buy target, else hold cash
    """
    # Calculate RSI for both tickers
    signal_rsi = calculate_rsi(signal_prices, window=rsi_period, method=rsi_method)
    comparison_rsi = calculate_rsi(comparison_prices, window=rsi_period, method=rsi_method)
    
    # Generate buy signals based on comparison operator
    if comparison_operator == "greater_than":
        buy_signals = (signal_rsi > comparison_rsi).astype(int)
    else:  # less_than (default)
        buy_signals = (signal_rsi < comparison_rsi).astype(int)
    
    # Apply preconditions if they exist
    if preconditions and precondition_data:
        precondition_mask = pd.Series(True, index=signal_prices.index)
        
        # Calculate main signal output for reference
        main_signal_output = buy_signals.copy()
        
        # Track precondition signal outputs
        precondition_outputs = {}
        
        for i, precondition in enumerate(preconditions):
            precondition_ticker = precondition['signal_ticker']
            if precondition_ticker in precondition_data:
                # Use custom RSI period if specified, otherwise use default
                precondition_rsi_period = precondition.get('rsi_period', rsi_period)
                precondition_rsi = calculate_rsi(precondition_data[precondition_ticker], window=precondition_rsi_period, method=rsi_method)
                
                if precondition.get('type') == 'comparison':
                    # RSI comparison precondition
                    comparison_ticker = precondition['comparison_ticker']
                    if comparison_ticker in precondition_data:
                        # Use custom RSI periods for comparison
                        signal_rsi_period = precondition.get('signal_rsi_period', rsi_period)
                        comparison_rsi_period = precondition.get('comparison_rsi_period', rsi_period)
                        precondition_comparison_operator = precondition.get('comparison_operator', 'less_than')
                        
                        signal_precondition_rsi = calculate_rsi(precondition_data[precondition_ticker], window=signal_rsi_period, method=rsi_method)
                        comparison_precondition_rsi = calculate_rsi(precondition_data[comparison_ticker], window=comparison_rsi_period, method=rsi_method)
                        
                        if precondition_comparison_operator == "greater_than":
                            precondition_condition = (signal_precondition_rsi > comparison_precondition_rsi)
                        else:  # less_than (default)
                            precondition_condition = (signal_precondition_rsi < comparison_precondition_rsi)
                        
                        # Store precondition output for reference
                        precondition_outputs[f"Precondition {i+1} Signal"] = precondition_condition.astype(int)
                    else:
                        precondition_condition = pd.Series(False, index=signal_prices.index)
                else:
                    # RSI threshold precondition
                    precondition_comparison = precondition.get('comparison', 'less_than')
                    precondition_threshold = precondition.get('threshold', 50.0)
                    
                    if precondition_comparison == "less_than":
                        precondition_condition = (precondition_rsi <= precondition_threshold)
                    else:  # greater_than
                        precondition_condition = (precondition_rsi >= precondition_threshold)
                
                precondition_mask = precondition_mask & precondition_condition
        
        buy_signals = buy_signals & precondition_mask
    
    # Calculate equity curve using signal outputs for fallbacks
    equity_curve = pd.Series(1.0, index=target_prices.index)
    current_equity = 1.0
    in_position = False
    entry_equity = 1.0
    entry_price = None
    trades = []
    
    # Get fallback signal if specified
    fallback_signal = None
    if preconditions and precondition_data:
        # For now, use the main signal as fallback
        # This will be enhanced to use actual signal outputs
        fallback_signal = buy_signals.copy()
    
    for date in target_prices.index:
        current_signal = buy_signals[date] if date in buy_signals.index else 0
        current_price = target_prices[date]
        
        if current_signal == 1 and not in_position:
            # Enter position
            in_position = True
            entry_equity = current_equity
            entry_price = current_price
            
        elif current_signal == 0 and in_position:
            # Exit position
            trade_return = (current_price - entry_price) / entry_price
            current_equity = entry_equity * (1 + trade_return)
            trades.append(trade_return)
            in_position = False
        
        # Update equity curve
        if in_position:
            current_equity = entry_equity * (current_price / entry_price)
        
        equity_curve[date] = current_equity
    
    # Handle case where we're still in position at the end
    if in_position:
        final_price = target_prices.iloc[-1]
        trade_return = (final_price - entry_price) / entry_price
        current_equity = entry_equity * (1 + trade_return)
        trades.append(trade_return)
        equity_curve.iloc[-1] = current_equity
    
    # Calculate metrics
    total_return = current_equity - 1.0
    annualized_return = (current_equity - 1.0) * (365 / (target_prices.index[-1] - target_prices.index[0]).days)
    
    # Calculate additional metrics
    if len(trades) > 0:
        returns_array = np.array(trades)
        additional_metrics = calculate_additional_metrics(returns_array, equity_curve, annualized_return, use_quantstats)
    else:
        additional_metrics = {
            'win_rate': 0.0,
            'total_trades': 0,
            'avg_hold_days': 0,
            'sortino_ratio': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'calmar_ratio': 0.0,
            'var_95': 0.0,
            'volatility': 0.0,
            'beta': 0.0,
            'alpha': 0.0,
            'information_ratio': 0.0
        }
    
    # Calculate additional metrics for display compatibility
    if len(trades) > 0:
        returns_array = np.array(trades)
        avg_return = np.mean(returns_array)
        median_return = np.median(returns_array)
        return_std = np.std(returns_array)
        best_return = np.max(returns_array)
        worst_return = np.min(returns_array)
        final_equity = current_equity
    else:
        avg_return = 0.0
        median_return = 0.0
        return_std = 0.0
        best_return = 0.0
        worst_return = 0.0
        final_equity = 1.0
    
    return {
        'equity_curve': equity_curve,
        'total_return': total_return,
        'annualized_return': annualized_return,
        'trades': trades,
        'avg_return': avg_return,
        'median_return': median_return,
        'return_std': return_std,
        'best_return': best_return,
        'worst_return': worst_return,
        'final_equity': final_equity,
        **additional_metrics
    }

def analyze_rsi_signals(signal_prices: pd.Series, target_prices: pd.Series, rsi_threshold: float, comparison: str = "less_than", rsi_period: int = 14, rsi_method: str = "wilders", use_quantstats: bool = True, preconditions: List[Dict] = None, precondition_data: Dict[str, pd.Series] = None) -> Dict:
    """Analyze RSI signals for a specific threshold with optional preconditions"""
    # Calculate RSI for the SIGNAL ticker using specified period and method
    signal_rsi = calculate_rsi(signal_prices, window=rsi_period, method=rsi_method)
    
    # Generate buy signals based on SIGNAL RSI threshold and comparison
    if comparison == "less_than":
        # "≤" configuration: Buy TARGET when SIGNAL RSI ≤ threshold, sell when SIGNAL RSI < threshold
        base_signals = (signal_rsi <= rsi_threshold).astype(int)
    else:  # greater_than
        # "≥" configuration: Buy TARGET when SIGNAL RSI ≥ threshold, sell when SIGNAL RSI < threshold
        base_signals = (signal_rsi >= rsi_threshold).astype(int)
    
    # Apply preconditions if provided
    if preconditions and precondition_data:
        # Start with all signals as valid
        precondition_mask = pd.Series(True, index=signal_prices.index)
        
        for precondition in preconditions:
            precondition_ticker = precondition['signal_ticker']
            precondition_comparison = precondition['comparison']
            precondition_threshold = precondition['threshold']
            
            # Get RSI data for this precondition ticker
            if precondition_ticker in precondition_data:
                precondition_rsi = calculate_rsi(precondition_data[precondition_ticker], window=rsi_period, method=rsi_method)
                
                # Apply the precondition condition
                if precondition_comparison == "less_than":
                    precondition_condition = (precondition_rsi <= precondition_threshold)
                else:  # greater_than
                    precondition_condition = (precondition_rsi >= precondition_threshold)
                
                # Update the mask (all preconditions must be True)
                precondition_mask = precondition_mask & precondition_condition
            else:
                st.warning(f"⚠️ No data found for precondition ticker: {precondition_ticker}")
                return {
                    'total_trades': 0,
                    'win_rate': 0,
                    'avg_return': 0,
                    'median_return': 0,
                    'returns': [],
                    'avg_hold_days': 0,
                    'sortino_ratio': 0,
                    'equity_curve': pd.Series(1.0, index=target_prices.index),
                    'trades': [],
                    'annualized_return': 0
                }
        
        # Apply precondition mask to base signals
        signals = base_signals & precondition_mask
    else:
        signals = base_signals
    
    # Calculate equity curve day by day - buy/sell TARGET based on SIGNAL RSI
    equity_curve = pd.Series(1.0, index=target_prices.index)
    current_equity = 1.0
    in_position = False
    entry_equity = 1.0
    entry_date = None
    entry_price = None
    trades = []
    
    for i, date in enumerate(target_prices.index):
        current_signal = signals[date] if date in signals.index else 0
        current_price = target_prices[date]  # TARGET price
        
        if current_signal == 1 and not in_position:
            # Enter position - buy TARGET at close when SIGNAL RSI meets condition
            in_position = True
            entry_equity = current_equity
            entry_date = date
            entry_price = current_price
            
        elif current_signal == 0 and in_position:
            # Exit position - sell TARGET at close when SIGNAL RSI no longer meets condition
            trade_return = (current_price - entry_price) / entry_price
            current_equity = entry_equity * (1 + trade_return)
            
            hold_days = (date - entry_date).days
            trades.append({
                'entry_date': entry_date,
                'exit_date': date,
                'entry_price': entry_price,
                'exit_price': current_price,
                'return': trade_return,
                'hold_days': hold_days
            })
            
            in_position = False
        
        # Update equity curve
        if in_position:
            # Mark-to-market the TARGET position
            current_equity = entry_equity * (current_price / entry_price)
        
        equity_curve[date] = current_equity
    
    # Handle case where we're still in position at the end
    if in_position:
        final_price = target_prices.iloc[-1]
        final_date = target_prices.index[-1]
        trade_return = (final_price - entry_price) / entry_price
        current_equity = entry_equity * (1 + trade_return)
        
        hold_days = (final_date - entry_date).days
        trades.append({
            'entry_date': entry_date,
            'exit_date': final_date,
            'entry_price': entry_price,
            'exit_price': final_price,
            'return': trade_return,
            'hold_days': hold_days
        })
        equity_curve.iloc[-1] = current_equity
    
    if not trades:
        return {
            'total_trades': 0,
            'win_rate': 0,
            'avg_return': 0,
            'median_return': 0,
            'returns': [],
            'avg_hold_days': 0,
            'sortino_ratio': 0,
            'equity_curve': equity_curve,
            'trades': [],
            'annualized_return': 0
        }
    
    returns = np.array([trade['return'] for trade in trades])
    win_rate = (returns > 0).mean()
    avg_return = returns.mean()
    median_return = np.median(returns)
    avg_hold_days = np.mean([trade['hold_days'] for trade in trades])
    sortino_ratio = calculate_sortino_ratio(returns, use_quantstats=use_quantstats)
    
    # Calculate annualized return
    total_days = (target_prices.index[-1] - target_prices.index[0]).days
    total_return = equity_curve.iloc[-1] - 1
    annualized_return = (1 + total_return) ** (365 / total_days) - 1 if total_days > 0 else 0
    
    return {
        'total_trades': len(returns),
        'win_rate': win_rate,
        'avg_return': avg_return,
        'median_return': median_return,
        'returns': returns,
        'avg_hold_days': avg_hold_days,
        'sortino_ratio': sortino_ratio,
        'equity_curve': equity_curve,
        'trades': trades,
        'annualized_return': annualized_return
    }

def calculate_statistical_significance(strategy_equity_curve: pd.Series, benchmark_equity_curve: pd.Series, 
                                    strategy_annualized: float, benchmark_annualized: float) -> Dict:
    """Calculate statistical significance by comparing strategy vs SPY equity curves under same conditions"""
    
    if len(strategy_equity_curve) == 0 or len(benchmark_equity_curve) == 0:
        return {
            't_statistic': 0,
            'p_value': 1.0,
            'confidence_level': 0,
            'significant': False,
            'effect_size': 0,
            'power': 0,
            'insufficient_data': True
        }
    
    # Align the equity curves on the same dates
    common_dates = strategy_equity_curve.index.intersection(benchmark_equity_curve.index)
    if len(common_dates) < 20:  # Increased minimum data points for more reliable tests
        return {
            't_statistic': 0,
            'p_value': 1.0,
            'confidence_level': 0,
            'significant': False,
            'effect_size': 0,
            'power': 0,
            'insufficient_data': True
        }
    
    strategy_aligned = strategy_equity_curve[common_dates]
    benchmark_aligned = benchmark_equity_curve[common_dates]
    
    # Calculate daily returns for both strategies
    strategy_returns = strategy_aligned.pct_change().dropna()
    benchmark_returns = benchmark_aligned.pct_change().dropna()
    
    # Ensure we have enough data points
    if len(strategy_returns) < 20 or len(benchmark_returns) < 20:
        return {
            't_statistic': 0,
            'p_value': 1.0,
            'confidence_level': 0,
            'significant': False,
            'effect_size': 0,
            'power': 0,
            'insufficient_data': True
        }
    
    # Perform two-tailed t-test first
    t_stat, p_value_two_tail = stats.ttest_ind(strategy_returns.values, benchmark_returns.values)
    
    # Calculate the difference in means
    strategy_mean = np.mean(strategy_returns.values)
    benchmark_mean = np.mean(benchmark_returns.values)
    mean_difference = strategy_mean - benchmark_mean
    
    # Calculate effect size (Cohen's d)
    pooled_std = np.sqrt(((len(strategy_returns) - 1) * np.var(strategy_returns.values, ddof=1) + 
                          (len(benchmark_returns) - 1) * np.var(benchmark_returns.values, ddof=1)) / 
                         (len(strategy_returns) + len(benchmark_returns) - 2))
    
    effect_size = mean_difference / pooled_std if pooled_std > 0 else 0
    
    # Calculate one-tailed p-value based on direction
    if mean_difference > 0:
        # Strategy outperforms benchmark
        p_value_one_tail = p_value_two_tail / 2
        confidence_level = (1 - p_value_one_tail) * 100
        significant = p_value_one_tail < 0.05
    else:
        # Strategy underperforms benchmark - calculate confidence for underperformance
        p_value_one_tail = 1 - (p_value_two_tail / 2)
        confidence_level = (1 - p_value_one_tail) * 100
        significant = p_value_one_tail < 0.05  # Significant underperformance
    
    # Calculate statistical power (simplified)
    power = 0.8 if len(strategy_returns) > 30 and abs(effect_size) > 0.5 else 0.5
    
    return {
        't_statistic': t_stat,
        'p_value': p_value_one_tail,
        'confidence_level': confidence_level,
        'significant': significant,
        'effect_size': effect_size,
        'power': power,
        'insufficient_data': False
    }

def calculate_max_drawdown(equity_curve: pd.Series, use_quantstats: bool = True) -> float:
    """Calculate maximum drawdown using QuantStats or fallback"""
    if equity_curve.empty:
        return 0.0
    
    if QUANTSTATS_AVAILABLE and use_quantstats:
        try:
            # Use QuantStats max drawdown calculation
            max_dd = qs.stats.max_drawdown(equity_curve)
            return abs(max_dd) if not np.isnan(max_dd) else 0.0
        except Exception:
            pass  # Fall through to original calculation
    
    # Fallback to original calculation
    running_max = equity_curve.expanding().max()
    drawdown = (equity_curve - running_max) / running_max
    return abs(drawdown.min())

def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.02, use_quantstats: bool = True) -> float:
    """Calculate Sharpe ratio using QuantStats or fallback"""
    if len(returns) == 0:
        return 0.0
    
    # Convert to pandas Series for QuantStats
    returns_series = pd.Series(returns)
    
    if QUANTSTATS_AVAILABLE and use_quantstats:
        try:
            # Use QuantStats sharpe ratio calculation
            sharpe_ratio = qs.stats.sharpe(returns_series, rf=risk_free_rate)
            return sharpe_ratio if not np.isnan(sharpe_ratio) else 0.0
        except Exception:
            pass  # Fall through to original calculation
    
    # Fallback to original calculation
    rf_per_trade = risk_free_rate / 252
    excess_returns = returns - rf_per_trade
    if np.std(excess_returns) == 0:
        return 0.0 if np.mean(excess_returns) == 0 else np.inf
    return np.mean(excess_returns) / np.std(excess_returns)

def calculate_additional_metrics(returns: np.ndarray, equity_curve: pd.Series, annual_return: float, use_quantstats: bool = True) -> Dict:
    """Add more comprehensive risk metrics using QuantStats or fallback"""
    if len(returns) == 0 or equity_curve.empty:
        return {
            'win_rate': 0.0,
            'total_trades': 0,
            'avg_hold_days': 0,
            'sortino_ratio': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'calmar_ratio': 0.0,
            'var_95': 0.0,
            'volatility': 0.0,
            'beta': 0.0,
            'alpha': 0.0,
            'information_ratio': 0.0
        }
    
    # Convert to pandas Series for QuantStats
    returns_series = pd.Series(returns)
    
    # Calculate basic metrics
    win_rate = (returns > 0).mean() if len(returns) > 0 else 0.0
    total_trades = len(returns)
    avg_hold_days = len(equity_curve) / total_trades if total_trades > 0 else 0
    sortino_ratio = calculate_sortino_ratio(returns, use_quantstats=use_quantstats)
    
    # Use QuantStats if available
    if QUANTSTATS_AVAILABLE and use_quantstats:
        try:
            # Use QuantStats for various metrics
            max_dd = calculate_max_drawdown(equity_curve, use_quantstats)
            sharpe = calculate_sharpe_ratio(returns, use_quantstats=use_quantstats)
            
            # Calculate Calmar ratio using QuantStats
            calmar_ratio = qs.stats.calmar(returns_series) if len(returns) > 0 else 0.0
            
            # Calculate Value at Risk using QuantStats
            var_95 = qs.stats.var(returns_series, 0.05) if len(returns) > 0 else 0.0
            
            # Calculate volatility using QuantStats
            volatility = qs.stats.volatility(returns_series) if len(returns) > 0 else 0.0
            
            # Additional QuantStats metrics
            beta = qs.stats.beta(returns_series, returns_series) if len(returns) > 0 else 0.0  # Self-beta as placeholder
            alpha = qs.stats.alpha(returns_series, returns_series) if len(returns) > 0 else 0.0  # Self-alpha as placeholder
            information_ratio = qs.stats.information_ratio(returns_series, returns_series) if len(returns) > 0 else 0.0  # Self-IR as placeholder
            
            return {
                'win_rate': win_rate,
                'total_trades': total_trades,
                'avg_hold_days': avg_hold_days,
                'sortino_ratio': sortino_ratio,
                'max_drawdown': max_dd,
                'calmar_ratio': calmar_ratio if not np.isnan(calmar_ratio) else (annual_return / max_dd if max_dd > 0 else 0.0),
                'var_95': var_95 if not np.isnan(var_95) else (np.percentile(returns, 5) if len(returns) > 0 else 0.0),
                'sharpe_ratio': sharpe,
                'volatility': volatility if not np.isnan(volatility) else (np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0.0),
                'beta': beta if not np.isnan(beta) else 0.0,
                'alpha': alpha if not np.isnan(alpha) else 0.0,
                'information_ratio': information_ratio if not np.isnan(information_ratio) else 0.0
            }
        except Exception:
            pass  # Fall through to fallback calculations
    
    # Fallback to original calculations
    max_dd = calculate_max_drawdown(equity_curve, use_quantstats)
    sharpe = calculate_sharpe_ratio(returns, use_quantstats=use_quantstats)
    
    return {
        'win_rate': win_rate,
        'total_trades': total_trades,
        'avg_hold_days': avg_hold_days,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_dd,
        'calmar_ratio': annual_return / max_dd if max_dd > 0 else 0.0,
        'var_95': np.percentile(returns, 5) if len(returns) > 0 else 0.0,
        'sharpe_ratio': sharpe,
        'volatility': np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0.0,
        'beta': 0.0,
        'alpha': 0.0,
        'information_ratio': 0.0
    }

def validate_data_quality(data: pd.Series, ticker: str) -> Tuple[bool, List[str]]:
    """Add data quality checks and return messages to display later"""
    messages = []
    
    if data is None or data.empty:
        st.error(f"❌ No data available for {ticker}")
        return False, messages
    
    # Check for missing data
    missing_pct = data.isnull().sum() / len(data) * 100
    if missing_pct > 5:  # More than 5% missing
        st.warning(f"⚠️ {missing_pct:.1f}% missing data detected for {ticker}")
    
    # Check for stock splits/dividends (extreme price movements)
    daily_returns = data.pct_change().dropna()
    extreme_moves = abs(daily_returns) > 0.15  # 15% daily moves
    if extreme_moves.sum() > 0:
        messages.append(f"🔍 Detected {extreme_moves.sum()} extreme price movements (>15%) for {ticker}")
    
    # Check for sufficient data
    if len(data) < 252:  # Less than 1 year
        st.warning(f"⚠️ Limited data for {ticker}: {len(data)} days (recommend at least 252 days)")
    
    return True, messages

# QuantStats report generation removed to avoid import issues
# Basic QuantStats metrics are still available in the main analysis functions

def run_rsi_comparison_analysis(signal_ticker: str, comparison_ticker: str, target_ticker: str, fallback_ticker: str = "BIL",
                               signal_rsi_period: int = 10, comparison_rsi_period: int = 10,
                               start_date=None, end_date=None, rsi_method: str = "wilders", benchmark_ticker: str = "SPY", buyhold_benchmark: str = "SPY", use_quantstats: bool = True, preconditions: List[Dict] = None, exclusions: List[Dict] = None) -> Tuple[pd.DataFrame, pd.Series, pd.Series, List[str]]:
    """
    Run RSI comparison analysis: when signal RSI < comparison RSI, buy target, else hold cash
    """
    # Show progress for data fetching
    progress_text = st.empty()
    progress_bar = st.progress(0)
    
    # Get data for all required tickers
    progress_text.text("Fetching signal ticker data...")
    progress_bar.progress(0.1)
    signal_data = get_stock_data(signal_ticker, start_date, end_date, exclusions)
    
    progress_text.text("Fetching comparison ticker data...")
    progress_bar.progress(0.2)
    comparison_data = get_stock_data(comparison_ticker, start_date, end_date, exclusions)
    
    progress_text.text("Fetching target ticker data...")
    progress_bar.progress(0.3)
    target_data = get_stock_data(target_ticker, start_date, end_date, exclusions)
    
    progress_text.text("Fetching benchmark data...")
    progress_bar.progress(0.4)
    benchmark_data = get_stock_data(benchmark_ticker, start_date, end_date, exclusions)
    
    progress_text.text("Fetching buy-and-hold benchmark data...")
    progress_bar.progress(0.5)
    buyhold_data = get_stock_data(buyhold_benchmark, start_date, end_date, exclusions)
    
    # Validate data quality
    progress_text.text("Validating data quality...")
    progress_bar.progress(0.6)
    data_messages = []
    for ticker, data in [(signal_ticker, signal_data), (comparison_ticker, comparison_data), 
                         (target_ticker, target_data), (benchmark_ticker, benchmark_data), (buyhold_benchmark, buyhold_data)]:
        is_valid, messages = validate_data_quality(data, ticker)
        if not is_valid:
            data_messages.extend(messages)
    
    if data_messages:
        progress_bar.progress(1.0)
        progress_text.text("Analysis completed with data quality issues.")
        return pd.DataFrame(), pd.Series(), pd.Series(), data_messages
    
    # Get common date range
    progress_text.text("Aligning data on common dates...")
    progress_bar.progress(0.7)
    common_dates = signal_data.index.intersection(comparison_data.index).intersection(target_data.index).intersection(benchmark_data.index).intersection(buyhold_data.index)
    if len(common_dates) < 30:
        progress_bar.progress(1.0)
        progress_text.text("Analysis completed - insufficient data.")
        return pd.DataFrame(), pd.Series(), pd.Series(), [f"Insufficient overlapping data. Only {len(common_dates)} common dates found."]
    
    signal_data = signal_data[common_dates]
    comparison_data = comparison_data[common_dates]
    target_data = target_data[common_dates]
    benchmark_data = benchmark_data[common_dates]
    buyhold_data = buyhold_data[common_dates]
    
    # Create buy-and-hold benchmark
    benchmark = benchmark_data / benchmark_data.iloc[0]  # Normalize to start at 1.0
    buyhold_benchmark = buyhold_data / buyhold_data.iloc[0]  # Normalize to start at 1.0
    
    # Get precondition data if needed
    progress_text.text("Processing preconditions...")
    progress_bar.progress(0.8)
    precondition_data = {}
    if preconditions:
        unique_tickers = set()
        for i, precondition in enumerate(preconditions):
            unique_tickers.add(precondition['signal_ticker'])
            if precondition.get('type') == 'comparison':
                unique_tickers.add(precondition['comparison_ticker'])
                # Resolve fallback ticker for enhanced options
                fallback_option = precondition.get('fallback_ticker', 'BIL')
                resolved_fallback = resolve_fallback_ticker(fallback_option, signal_ticker, preconditions, i)
                if resolved_fallback not in [signal_ticker, comparison_ticker, target_ticker, benchmark_ticker]:
                    unique_tickers.add(resolved_fallback)
        
        for ticker in unique_tickers:
            if ticker not in [signal_ticker, comparison_ticker, target_ticker, benchmark_ticker]:
                ticker_data = get_stock_data(ticker, start_date, end_date, exclusions)
                if len(ticker_data) > 0:
                    precondition_data[ticker] = ticker_data[common_dates]
    
    # Run single RSI comparison analysis
    progress_text.text("Running RSI comparison analysis...")
    progress_bar.progress(0.9)
    analysis = analyze_rsi_comparison_signals(signal_data, comparison_data, target_data, rsi_period, rsi_method, comparison, use_quantstats, preconditions, precondition_data)
    
    # Create benchmark equity curve that follows the same RSI conditions
    signal_rsi = calculate_rsi(signal_data, window=rsi_period, method=rsi_method)
    comparison_rsi = calculate_rsi(comparison_data, window=rsi_period, method=rsi_method)
    
    # Generate buy signals for benchmark (same as strategy)
    if comparison == "greater_than":
        benchmark_base_signals = (signal_rsi > comparison_rsi).astype(int)
    else:  # less_than (default)
        benchmark_base_signals = (signal_rsi < comparison_rsi).astype(int)
    
    # Apply preconditions to benchmark signals if they exist
    if preconditions and precondition_data:
        benchmark_precondition_mask = pd.Series(True, index=signal_data.index)
        
        for precondition in preconditions:
            precondition_ticker = precondition['signal_ticker']
            if precondition_ticker in precondition_data:
                # Use custom RSI period if specified, otherwise use default
                precondition_rsi_period = precondition.get('rsi_period', rsi_period)
                precondition_rsi = calculate_rsi(precondition_data[precondition_ticker], window=precondition_rsi_period, method=rsi_method)
                
                if precondition.get('type') == 'comparison':
                    # RSI comparison precondition
                    comparison_ticker = precondition['comparison_ticker']
                    if comparison_ticker in precondition_data:
                        # Use custom RSI periods for comparison
                        signal_rsi_period = precondition.get('signal_rsi_period', rsi_period)
                        comparison_rsi_period = precondition.get('comparison_rsi_period', rsi_period)
                        
                        signal_precondition_rsi = calculate_rsi(precondition_data[precondition_ticker], window=signal_rsi_period, method=rsi_method)
                        comparison_precondition_rsi = calculate_rsi(precondition_data[comparison_ticker], window=comparison_rsi_period, method=rsi_method)
                        
                        # Use the same comparison operator as the main strategy
                        if comparison == "greater_than":
                            precondition_condition = (signal_precondition_rsi > comparison_precondition_rsi)
                        else:  # less_than (default)
                            precondition_condition = (signal_precondition_rsi < comparison_precondition_rsi)
                    else:
                        precondition_condition = pd.Series(False, index=signal_data.index)
                else:
                    # RSI threshold precondition
                    precondition_comparison = precondition.get('comparison', 'less_than')
                    precondition_threshold = precondition.get('threshold', 50.0)
                    
                    if precondition_comparison == "less_than":
                        precondition_condition = (precondition_rsi <= precondition_threshold)
                    else:  # greater_than
                        precondition_condition = (precondition_rsi >= precondition_threshold)
                
                benchmark_precondition_mask = benchmark_precondition_mask & precondition_condition
        
        benchmark_signals = benchmark_base_signals & benchmark_precondition_mask
    else:
        benchmark_signals = benchmark_base_signals
    
    # Calculate benchmark equity curve using benchmark prices (same logic as strategy)
    benchmark_equity_curve = pd.Series(1.0, index=benchmark_data.index)
    current_equity = 1.0
    in_position = False
    entry_equity = 1.0
    entry_price = None
    benchmark_trades = []
    
    for date in benchmark_data.index:
        current_signal = benchmark_signals[date] if date in benchmark_signals.index else 0
        current_price = benchmark_data[date]
        
        if current_signal == 1 and not in_position:
            # Enter position
            in_position = True
            entry_equity = current_equity
            entry_price = current_price
            
        elif current_signal == 0 and in_position:
            # Exit position
            trade_return = (current_price - entry_price) / entry_price
            current_equity = entry_equity * (1 + trade_return)
            benchmark_trades.append(trade_return)
            in_position = False
        
        # Update equity curve
        if in_position:
            current_equity = entry_equity * (current_price / entry_price)
        
        benchmark_equity_curve[date] = current_equity
    
    # Handle case where we're still in position at the end
    if in_position:
        final_price = benchmark_data.iloc[-1]
        trade_return = (final_price - entry_price) / entry_price
        current_equity = entry_equity * (1 + trade_return)
        benchmark_trades.append(trade_return)
        benchmark_equity_curve.iloc[-1] = current_equity
    
    # Calculate benchmark metrics
    benchmark_avg_return = np.mean(benchmark_trades) if benchmark_trades else 0
    benchmark_median_return = np.median(benchmark_trades) if benchmark_trades else 0
    benchmark_annualized = (benchmark.iloc[-1] - 1) * (365 / (benchmark.index[-1] - benchmark.index[0]).days)
    
    # Calculate statistical significance
    significance_results = calculate_statistical_significance(
        analysis['equity_curve'], benchmark_equity_curve, 
        analysis['annualized_return'], benchmark_annualized
    )
    
    # Create results DataFrame with all required columns
    results_df = pd.DataFrame({
        'Signal_Ticker': [signal_ticker],
        'Comparison_Ticker': [comparison_ticker],
        'Target_Ticker': [target_ticker],
        'Analysis_Type': ['RSI Comparison'],
        'Total_Return': [analysis['total_return']],
        'annualized_return': [analysis['annualized_return']],
        'Win_Rate': [analysis['win_rate']],
        'Total_Trades': [analysis['total_trades']],
        'Avg_Hold_Days': [analysis['avg_hold_days']],
        'Sortino_Ratio': [analysis['sortino_ratio']],
        'Sharpe_Ratio': [analysis['sharpe_ratio']],
        'Max_Drawdown': [analysis['max_drawdown']],
        'Calmar_Ratio': [analysis['calmar_ratio']],
        'VaR_95': [analysis['var_95']],
        'Volatility': [analysis['volatility']],
        'Beta': [analysis['beta']],
        'Alpha': [analysis['alpha']],
        'Information_Ratio': [analysis['information_ratio']],
        'Benchmark_Return': [benchmark_annualized],
        'Benchmark_Avg_Return': [benchmark_avg_return],
        'Benchmark_Median_Return': [benchmark_median_return],
        'Confidence_Level': [significance_results['confidence_level']],
        'P_Value': [significance_results['p_value']],
        'T_Statistic': [significance_results['t_statistic']],
        'Effect_Size': [significance_results['effect_size']],
        'Power': [significance_results['power']],
        'significant': [significance_results['significant']],
        'equity_curve': [analysis['equity_curve']],
        'benchmark_equity_curve': [benchmark_equity_curve],
        # Add missing columns that the display expects
        'Avg_Return': [analysis.get('avg_return', 0.0)],
        'Median_Return': [analysis.get('median_return', 0.0)],
        'Return_Std': [analysis.get('return_std', 0.0)],
        'Best_Return': [analysis.get('best_return', 0.0)],
        'Worst_Return': [analysis.get('worst_return', 0.0)],
        'Final_Equity': [analysis.get('final_equity', 1.0)],
        'confidence_level': [significance_results['confidence_level']],
        'effect_size': [significance_results['effect_size']],
        'p_value': [significance_results['p_value']],
        'sharpe_ratio': [analysis['sharpe_ratio']],
        'calmar_ratio': [analysis['calmar_ratio']],
        'max_drawdown': [analysis['max_drawdown']],
        'var_95': [analysis['var_95']]
    })
    
    progress_bar.progress(1.0)
    progress_text.text("Analysis completed successfully!")
    
    return results_df, benchmark, buyhold_benchmark, data_messages

def run_rsi_analysis(signal_ticker: str, target_ticker: str, rsi_threshold: float, comparison: str, 
                    start_date=None, end_date=None, rsi_period: int = 14, rsi_method: str = "wilders", benchmark_ticker: str = "SPY", use_quantstats: bool = True, preconditions: List[Dict] = None, exclusions: List[Dict] = None, rsi_min: float = 0.0, rsi_max: float = 100.0) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Run comprehensive RSI analysis across the specified range with optional preconditions and exclusions"""
    
    # Fetch data with quality validation
    all_messages = []
    
    with st.spinner(f"Fetching data for {signal_ticker}..."):
        signal_data = get_stock_data(signal_ticker, start_date, end_date, exclusions)
        is_valid, messages = validate_data_quality(signal_data, signal_ticker)
        all_messages.extend(messages)
        if not is_valid:
            return None, None
    
    with st.spinner(f"Fetching data for {target_ticker}..."):
        target_data = get_stock_data(target_ticker, start_date, end_date, exclusions)
        is_valid, messages = validate_data_quality(target_data, target_ticker)
        all_messages.extend(messages)
        if not is_valid:
            return None, None
    
    # Fetch benchmark data for comparison - use user-selected benchmark
    with st.spinner(f"Fetching benchmark data ({benchmark_ticker})..."):
        benchmark_data = get_stock_data(benchmark_ticker, start_date, end_date, exclusions)
        is_valid, messages = validate_data_quality(benchmark_data, benchmark_ticker)
        all_messages.extend(messages)
        if not is_valid:
            return None, None
    
    # Fetch precondition data if preconditions are set
    precondition_data = {}
    if preconditions:
        unique_precondition_tickers = list(set([p['signal_ticker'] for p in preconditions]))
        for ticker in unique_precondition_tickers:
            if ticker != signal_ticker:  # Don't fetch again if it's the same as main signal
                with st.spinner(f"Fetching precondition data for {ticker}..."):
                    ticker_data = get_stock_data(ticker, start_date, end_date, exclusions)
                    is_valid, messages = validate_data_quality(ticker_data, ticker)
                    all_messages.extend(messages)
                    if not is_valid:
                        return None, None
                    precondition_data[ticker] = ticker_data
        
        # Add main signal data to precondition data for consistency
        precondition_data[signal_ticker] = signal_data
    
    if signal_data is None or target_data is None or benchmark_data is None:
        return None, None
    
    # Align data on common dates
    common_dates = signal_data.index.intersection(target_data.index).intersection(benchmark_data.index)
    signal_data = signal_data[common_dates]
    target_data = target_data[common_dates]
    benchmark_data = benchmark_data[common_dates]
    
    # Create buy-and-hold benchmark
    benchmark = benchmark_data / benchmark_data.iloc[0]  # Normalize to start at 1.0
    
    # Calculate benchmark returns for statistical testing
    benchmark_returns = benchmark_data.pct_change().dropna()
    
    # Generate RSI thresholds based on the specified range
    rsi_thresholds = np.arange(rsi_min, rsi_max + 0.5, 0.5)
    
    results = []
    
    progress_bar = st.progress(0)
    progress_text = st.empty()
    total_thresholds = len(rsi_thresholds)
    
    for i, threshold in enumerate(rsi_thresholds):
        progress_text.text(f"Analyzing RSI threshold {threshold:.1f} ({i+1}/{total_thresholds})")
        analysis = analyze_rsi_signals(signal_data, target_data, threshold, comparison, rsi_period, rsi_method, use_quantstats, preconditions, precondition_data)
        
        # Calculate statistical significance
        strategy_equity_curve = analysis['equity_curve']
        if len(strategy_equity_curve) > 0:
            # Create benchmark equity curve that follows the same RSI conditions
            # This ensures we're comparing strategy vs benchmark under the same conditions
            signal_rsi = calculate_rsi(signal_data, window=rsi_period, method=rsi_method)
            
            # Generate buy signals for benchmark (same as strategy)
            if comparison == "less_than":
                benchmark_base_signals = (signal_rsi <= threshold).astype(int)
            else:  # greater_than
                benchmark_base_signals = (signal_rsi >= threshold).astype(int)
            
            # Apply preconditions to benchmark signals if they exist
            if preconditions and precondition_data:
                benchmark_precondition_mask = pd.Series(True, index=signal_data.index)
                
                for precondition in preconditions:
                    precondition_ticker = precondition['signal_ticker']
                    precondition_comparison = precondition['comparison']
                    precondition_threshold = precondition['threshold']
                    
                    if precondition_ticker in precondition_data:
                        precondition_rsi = calculate_rsi(precondition_data[precondition_ticker], window=rsi_period, method=rsi_method)
                        
                        if precondition_comparison == "less_than":
                            precondition_condition = (precondition_rsi <= precondition_threshold)
                        else:  # greater_than
                            precondition_condition = (precondition_rsi >= precondition_threshold)
                        
                        benchmark_precondition_mask = benchmark_precondition_mask & precondition_condition
                
                benchmark_signals = benchmark_base_signals & benchmark_precondition_mask
            else:
                benchmark_signals = benchmark_base_signals
            
            # Calculate benchmark equity curve using benchmark prices (same logic as strategy)
            benchmark_equity_curve = pd.Series(1.0, index=benchmark_data.index)
            current_equity = 1.0
            in_position = False
            entry_equity = 1.0
            entry_price = None
            benchmark_trades = []
            
            for date in benchmark_data.index:
                current_signal = benchmark_signals[date] if date in benchmark_signals.index else 0
                current_price = benchmark_data[date]
                
                if current_signal == 1 and not in_position:
                    # Enter position
                    in_position = True
                    entry_equity = current_equity
                    entry_price = current_price
                    
                elif current_signal == 0 and in_position:
                    # Exit position
                    trade_return = (current_price - entry_price) / entry_price
                    current_equity = entry_equity * (1 + trade_return)
                    benchmark_trades.append(trade_return)
                    in_position = False
                
                # Update equity curve
                if in_position:
                    current_equity = entry_equity * (current_price / entry_price)
                
                benchmark_equity_curve[date] = current_equity
            
            # Handle case where we're still in position at the end
            if in_position:
                final_price = benchmark_data.iloc[-1]
                trade_return = (final_price - entry_price) / entry_price
                current_equity = entry_equity * (1 + trade_return)
                benchmark_trades.append(trade_return)
                benchmark_equity_curve.iloc[-1] = current_equity
            
            # Calculate benchmark average and median returns
            benchmark_avg_return = np.mean(benchmark_trades) if benchmark_trades else 0
            benchmark_median_return = np.median(benchmark_trades) if benchmark_trades else 0
            benchmark_annualized = (benchmark.iloc[-1] - 1) * (365 / (benchmark.index[-1] - benchmark.index[0]).days)
            stats_result = calculate_statistical_significance(
                strategy_equity_curve, 
                benchmark_equity_curve,
                analysis['annualized_return'],
                benchmark_annualized
            )
            
            # Calculate additional risk metrics
            risk_metrics = calculate_additional_metrics(analysis['returns'], analysis['equity_curve'], analysis['annualized_return'], use_quantstats)
        else:
            # Calculate benchmark average and median returns even when strategy has no trades
            signal_rsi = calculate_rsi(signal_data, window=rsi_period, method=rsi_method)
            
            # Generate buy signals for benchmark (same as strategy)
            if comparison == "less_than":
                benchmark_base_signals = (signal_rsi <= threshold).astype(int)
            else:  # greater_than
                benchmark_base_signals = (signal_rsi >= threshold).astype(int)
            
            # Apply preconditions to benchmark signals if they exist
            if preconditions and precondition_data:
                benchmark_precondition_mask = pd.Series(True, index=signal_data.index)
                
                for precondition in preconditions:
                    precondition_ticker = precondition['signal_ticker']
                    precondition_comparison = precondition['comparison']
                    precondition_threshold = precondition['threshold']
                    
                    if precondition_ticker in precondition_data:
                        precondition_rsi = calculate_rsi(precondition_data[precondition_ticker], window=rsi_period, method=rsi_method)
                        
                        if precondition_comparison == "less_than":
                            precondition_condition = (precondition_rsi <= precondition_threshold)
                        else:  # greater_than
                            precondition_condition = (precondition_rsi >= precondition_threshold)
                        
                        benchmark_precondition_mask = benchmark_precondition_mask & precondition_condition
                
                benchmark_signals = benchmark_base_signals & benchmark_precondition_mask
            else:
                benchmark_signals = benchmark_base_signals
            
            # Calculate benchmark equity curve using benchmark prices (same logic as strategy)
            benchmark_equity_curve = pd.Series(1.0, index=benchmark_data.index)
            current_equity = 1.0
            in_position = False
            entry_equity = 1.0
            entry_price = None
            benchmark_trades = []
            
            for date in benchmark_data.index:
                current_signal = benchmark_signals[date] if date in benchmark_signals.index else 0
                current_price = benchmark_data[date]
                
                if current_signal == 1 and not in_position:
                    # Enter position
                    in_position = True
                    entry_equity = current_equity
                    entry_price = current_price
                    
                elif current_signal == 0 and in_position:
                    # Exit position
                    trade_return = (current_price - entry_price) / entry_price
                    current_equity = entry_equity * (1 + trade_return)
                    benchmark_trades.append(trade_return)
                    in_position = False
                
                # Update equity curve
                if in_position:
                    current_equity = entry_equity * (current_price / entry_price)
                
                benchmark_equity_curve[date] = current_equity
            
            # Handle case where we're still in position at the end
            if in_position:
                final_price = benchmark_data.iloc[-1]
                trade_return = (final_price - entry_price) / entry_price
                current_equity = entry_equity * (1 + trade_return)
                benchmark_trades.append(trade_return)
                benchmark_equity_curve.iloc[-1] = current_equity
            
            # Calculate benchmark average and median returns
            benchmark_avg_return = np.mean(benchmark_trades) if benchmark_trades else 0
            benchmark_median_return = np.median(benchmark_trades) if benchmark_trades else 0
            
            stats_result = {
                't_statistic': 0,
                'p_value': 1.0,
                'confidence_level': 0,
                'significant': False,
                'effect_size': 0,
                'power': 0,
                'insufficient_data': True
            }
            
            # Calculate additional risk metrics (even when no trades)
            risk_metrics = calculate_additional_metrics(analysis['returns'], analysis['equity_curve'], analysis['annualized_return'], use_quantstats)
        
        results.append({
            'RSI_Threshold': threshold,
            'Total_Trades': analysis['total_trades'],
            'Win_Rate': analysis['win_rate'],
            'Avg_Return': analysis['avg_return'],
            'Median_Return': analysis.get('median_return', 0),  # Use get() with default value
            'Benchmark_Avg_Return': benchmark_avg_return,
            'Benchmark_Median_Return': benchmark_median_return,
            'Avg_Hold_Days': analysis['avg_hold_days'],
            'Sortino_Ratio': analysis['sortino_ratio'],
            'Return_Std': np.std(analysis['returns']) if len(analysis['returns']) > 0 else 0,
            'Best_Return': np.max(analysis['returns']) if len(analysis['returns']) > 0 else 0,
            'Worst_Return': np.min(analysis['returns']) if len(analysis['returns']) > 0 else 0,
            'Final_Equity': analysis['equity_curve'].iloc[-1] if analysis['equity_curve'] is not None else 1.0,
            'Total_Return': (analysis['equity_curve'].iloc[-1] - 1) if analysis['equity_curve'] is not None else 0,
            'annualized_return': analysis['annualized_return'],
            'equity_curve': analysis['equity_curve'],
            'trades': analysis['trades'],
            'returns': analysis['returns'],
            't_statistic': stats_result['t_statistic'],
            'p_value': stats_result['p_value'],
            'confidence_level': stats_result['confidence_level'],
            'significant': stats_result['significant'],
            'effect_size': stats_result['effect_size'],
            'power': stats_result['power'],
            'insufficient_data': stats_result.get('insufficient_data', False),
            'max_drawdown': risk_metrics['max_drawdown'],
            'calmar_ratio': risk_metrics['calmar_ratio'],
            'var_95': risk_metrics['var_95'],
            'sharpe_ratio': risk_metrics['sharpe_ratio'],
            'volatility': risk_metrics['volatility'],
            'beta': risk_metrics.get('beta', 0.0),
            'alpha': risk_metrics.get('alpha', 0.0),
            'information_ratio': risk_metrics.get('information_ratio', 0.0)
        })
        
        progress_bar.progress((i + 1) / total_thresholds)
    
    progress_text.text("Analysis completed successfully!")
    
    return pd.DataFrame(results), benchmark, all_messages



# Streamlit Interface
st.sidebar.header("⚙️ Configuration")

# QuantStats Configuration
use_quantstats = st.sidebar.checkbox("Enable QuantStats Integration", value=True, help="Enable use of QuantStats library. When disabled, the app will use fallback calculations.")

# Preconditions System
st.sidebar.subheader("Preconditions", help="Preconditions add additional RSI conditions that must ALL be true before the main signal is considered. This allows for multi-condition signal validation.")

# Initialize preconditions in session state if not exists
if 'preconditions' not in st.session_state:
    st.session_state.preconditions = []

# Display existing preconditions
if st.session_state.preconditions:
    st.sidebar.write("**Current Preconditions:**")
    for i, precondition in enumerate(st.session_state.preconditions):
        with st.sidebar.container():
            col1, col2 = st.columns([3, 1])
            with col1:
                if 'type' in precondition and precondition['type'] == 'comparison':
                    signal_period = precondition.get('signal_rsi_period', 10)
                    comparison_period = precondition.get('comparison_rsi_period', 10)
                    target_ticker = precondition.get('target_ticker', 'TQQQ')
                    fallback_ticker = precondition.get('fallback_ticker', 'BIL')
                    comparison_operator = precondition.get('comparison_operator', 'less_than')
                    
                    # Format comparison operator
                    operator_symbol = ">" if comparison_operator == "greater_than" else "<"
                    
                    # Format target and fallback display
                    if target_ticker == "Main Signal":
                        target_display = "Main Signal Output"
                    elif target_ticker.startswith("Precondition"):
                        target_display = f"{target_ticker} Output"
                    else:
                        target_display = target_ticker
                    
                    if fallback_ticker == "Main Signal":
                        fallback_display = "Main Signal Output"
                    elif fallback_ticker.startswith("Precondition"):
                        fallback_display = f"{fallback_ticker} Output"
                    else:
                        fallback_display = fallback_ticker
                    
                    st.write(f"• {precondition['signal_ticker']} {signal_period}d RSI {operator_symbol} {precondition['comparison_ticker']} {comparison_period}d RSI → {target_display} / {fallback_display}")
                else:
                    # Handle legacy format or threshold type
                    comparison_text = "≤" if precondition.get('comparison') == 'less_than' else "≥"
                    rsi_period = precondition.get('rsi_period', 10)
                    st.write(f"• {precondition['signal_ticker']} {rsi_period}d RSI {comparison_text} {precondition.get('threshold', 'N/A')}")
            with col2:
                if st.button(f"🗑️", key=f"remove_precondition_{i}"):
                    st.session_state.preconditions.pop(i)
                    st.rerun()

# Add new precondition
with st.sidebar.expander("➕ Add Precondition", expanded=False):
    st.write("**Add a new RSI precondition:**")
    
    # Precondition type selection
    precondition_type = st.selectbox("Precondition Type", 
                                    ["RSI Threshold", "RSI Comparison"], 
                                    key="precondition_type",
                                    help="Choose between RSI threshold or RSI comparison precondition.")
    
    if precondition_type == "RSI Threshold":
        # Precondition signal ticker (only for RSI Threshold mode)
        precondition_signal = st.text_input("Precondition Signal Ticker", 
                                           value="QQQ", 
                                           key="precondition_signal",
                                           help="The ticker whose RSI will be checked for this precondition. Can be the same as or different from the main signal ticker.")
        # Precondition RSI comparison
        precondition_comparison = st.selectbox("Precondition RSI Condition", 
                                              ["less_than", "greater_than"], 
                                              format_func=lambda x: "RSI ≤ threshold" if x == "less_than" else "RSI ≥ threshold",
                                              key="precondition_comparison",
                                              help="Choose the RSI condition for this precondition.")
        
        # Precondition RSI threshold
        precondition_threshold = st.number_input("Precondition RSI Threshold", 
                                               min_value=0.0, max_value=100.0, value=80.0, step=0.5,
                                               key="precondition_threshold",
                                               help="The RSI threshold value for this precondition.")
        
        # Precondition RSI period
        precondition_rsi_period = st.number_input("Precondition RSI Period (Days)", 
                                                 min_value=1, max_value=50, value=10,
                                                 key="precondition_rsi_period",
                                                 help="RSI period for this precondition ticker.")
        
        # Add precondition button
        if st.button("➕ Add Precondition", key="add_precondition"):
            new_precondition = {
                'type': 'threshold',
                'signal_ticker': precondition_signal.upper().strip(),
                'comparison': precondition_comparison,
                'threshold': precondition_threshold,
                'rsi_period': precondition_rsi_period
            }
            st.session_state.preconditions.append(new_precondition)
            st.rerun()
    else:
        # RSI Comparison precondition
        st.write("**Precondition RSI Comparison Configuration:**")
        
        # Signal and comparison tickers with RSI periods
        col1, col2 = st.columns(2)
        with col1:
            precondition_signal_ticker = st.text_input("Precondition Signal Ticker", 
                                                      value="QQQ", 
                                                      key="precondition_signal_ticker",
                                                      help="The signal ticker for this RSI comparison precondition.")
            precondition_signal_rsi_period = st.number_input("Precondition Signal RSI Period (Days)", 
                                                            min_value=1, max_value=50, value=10,
                                                            key="precondition_signal_rsi_period",
                                                            help="RSI period for the signal ticker in this comparison.")
        with col2:
            precondition_comparison_ticker = st.text_input("Precondition Comparison Ticker", 
                                                          value="XLK", 
                                                          key="precondition_comparison_ticker",
                                                          help="The comparison ticker for this RSI comparison precondition.")
            precondition_comparison_rsi_period = st.number_input("Precondition Comparison RSI Period (Days)", 
                                                                min_value=1, max_value=50, value=10,
                                                                key="precondition_comparison_rsi_period",
                                                                help="RSI period for the comparison ticker in this comparison.")
        
        # RSI Comparison operator for precondition
        precondition_comparison_operator = st.selectbox("Precondition RSI Comparison Operator", 
                                                       ["less_than", "greater_than"], 
                                                       format_func=lambda x: "Signal RSI < Comparison RSI" if x == "less_than" else "Signal RSI > Comparison RSI",
                                                       key="precondition_comparison_operator",
                                                       help="Choose the RSI comparison condition for this precondition.")
        
        # Target and fallback tickers
        col1, col2 = st.columns(2)
        with col1:
            # Create options for target ticker (same as fallback)
            target_options = ["Main Signal Output", "Custom Ticker"]
            # Add existing preconditions as options
            if st.session_state.get('preconditions'):
                for i, existing_precondition in enumerate(st.session_state.preconditions):
                    if existing_precondition.get('type') == 'comparison':
                        target_options.append(f"Precondition {i+1} Signal Output")
                    elif existing_precondition.get('type') == 'threshold':
                        target_options.append(f"Precondition {i+1} Signal Output")
            
            # Add future precondition options (up to 5 more preconditions)
            current_precondition_count = len(st.session_state.get('preconditions', []))
            for i in range(current_precondition_count + 1, current_precondition_count + 6):
                target_options.append(f"Precondition {i} Signal Output")
            
            precondition_target_type = st.selectbox("Precondition Target Type", 
                                                   target_options,
                                                   key="precondition_target_type",
                                                   help="Choose the target: 'Main Signal Output' = output of main RSI comparison, 'Custom Ticker' = specific ticker, 'Precondition X Signal Output' = output of another precondition.")
            
            if precondition_target_type == "Custom Ticker":
                precondition_target_ticker = st.text_input("Precondition Target Ticker", 
                                                          value="TQQQ", 
                                                          key="precondition_target_ticker",
                                                          help="The custom target ticker for this RSI comparison precondition.")
            else:
                # Convert back to internal format
                if precondition_target_type == "Main Signal Output":
                    precondition_target_ticker = "Main Signal"
                elif precondition_target_type.endswith(" Signal Output"):
                    # Extract precondition number
                    precondition_num = precondition_target_type.split()[1]
                    precondition_target_ticker = f"Precondition {precondition_num} Signal"
                else:
                    precondition_target_ticker = precondition_target_type
        with col2:
            # Create options for fallback ticker
            fallback_options = ["Main Signal Output", "Custom Ticker"]
            # Add existing preconditions as options
            if st.session_state.get('preconditions'):
                for i, existing_precondition in enumerate(st.session_state.preconditions):
                    if existing_precondition.get('type') == 'comparison':
                        fallback_options.append(f"Precondition {i+1} Signal Output")
                    elif existing_precondition.get('type') == 'threshold':
                        fallback_options.append(f"Precondition {i+1} Signal Output")
            
            # Add future precondition options (up to 5 more preconditions)
            current_precondition_count = len(st.session_state.get('preconditions', []))
            for i in range(current_precondition_count + 1, current_precondition_count + 6):
                fallback_options.append(f"Precondition {i} Signal Output")
            
            precondition_fallback_type = st.selectbox("Precondition Fallback Type", 
                                                     fallback_options,
                                                     key="precondition_fallback_type",
                                                     help="Choose the fallback: 'Main Signal' = output of main RSI comparison, 'Custom Ticker' = specific ticker, 'Precondition X Signal' = output of another precondition.")
            
            if precondition_fallback_type == "Custom Ticker":
                precondition_fallback_ticker = st.text_input("Precondition Fallback Ticker", 
                                                            value="BIL", 
                                                            key="precondition_fallback_ticker",
                                                            help="The custom fallback ticker for this RSI comparison precondition.")
            else:
                # Convert back to internal format
                if precondition_fallback_type == "Main Signal Output":
                    precondition_fallback_ticker = "Main Signal"
                elif precondition_fallback_type.endswith(" Signal Output"):
                    # Extract precondition number
                    precondition_num = precondition_fallback_type.split()[1]
                    precondition_fallback_ticker = f"Precondition {precondition_num} Signal"
                else:
                    precondition_fallback_ticker = precondition_fallback_type
        
        # Add precondition button
        if st.button("➕ Add Precondition", key="add_precondition"):
            new_precondition = {
                'type': 'comparison',
                'signal_ticker': precondition_signal_ticker.upper().strip(),
                'comparison_ticker': precondition_comparison_ticker.upper().strip(),
                'target_ticker': precondition_target_ticker.upper().strip(),
                'fallback_ticker': precondition_fallback_ticker.upper().strip(),
                'signal_rsi_period': precondition_signal_rsi_period,
                'comparison_rsi_period': precondition_comparison_rsi_period,
                'comparison_operator': precondition_comparison_operator
            }
            st.session_state.preconditions.append(new_precondition)
            st.rerun()

# Clear all preconditions button
if st.session_state.preconditions:
    if st.sidebar.button("🗑️ Clear All Preconditions", type="secondary"):
        st.session_state.preconditions = []
        st.rerun()

# Analysis Mode Selection
analysis_mode = st.sidebar.selectbox("Analysis Mode", 
                                    ["RSI Threshold", "RSI Comparison"], 
                                    help="Choose between testing RSI thresholds or comparing RSI between two tickers.")

# Input fields with help tooltips (for RSI Threshold mode only)
if analysis_mode == "RSI Threshold":
    signal_ticker = st.sidebar.text_input("Signal Ticker", value="QQQ", help="The ticker that generates RSI signals. This is the stock/ETF whose RSI we'll use to decide when to buy/sell the target ticker.")

# RSI Period selection (for RSI Threshold mode only)
if analysis_mode == "RSI Threshold":
    rsi_period = st.sidebar.number_input("RSI Period (Days)", min_value=1, max_value=50, value=10, 
                                        help="How many days to look back when calculating RSI. 10 is more sensitive to recent changes than the standard 14. Lower numbers make RSI more responsive to recent market movements.")

# RSI Calculation Method - Fixed to Wilder's method
rsi_method = "wilders"

if analysis_mode == "RSI Threshold":
    # RSI Threshold Mode
    comparison = st.sidebar.selectbox("RSI Condition", 
                                   ["greater_than", "less_than"], 
                                   format_func=lambda x: "RSI ≥ threshold" if x == "greater_than" else "RSI ≤ threshold",
                                   help="Choose when to buy: 'RSI ≥ threshold' means buy when RSI is high (overbought), 'RSI ≤ threshold' means buy when RSI is low (oversold).")
else:
    # RSI Comparison Mode
    st.sidebar.subheader("📊 RSI Comparison Configuration")
    
    # Signal and comparison tickers with RSI periods
    col1, col2 = st.sidebar.columns(2)
    with col1:
        signal_ticker = st.sidebar.text_input("Signal Ticker", value="KMLM", 
                                             help="The ticker whose RSI will be compared against the comparison ticker's RSI.")
        signal_rsi_period = st.sidebar.number_input("Signal RSI Period (Days)", 
                                                   min_value=1, max_value=50, value=10,
                                                   help="RSI period for the signal ticker.")
    with col2:
        comparison_ticker = st.sidebar.text_input("Comparison Ticker", value="XLK", 
                                                 help="The ticker whose RSI will be compared against the signal ticker's RSI.")
        comparison_rsi_period = st.sidebar.number_input("Comparison RSI Period (Days)", 
                                                       min_value=1, max_value=50, value=10,
                                                       help="RSI period for the comparison ticker.")
    
    # RSI Comparison operator
    comparison_operator = st.sidebar.selectbox("RSI Comparison Operator", 
                                              ["less_than", "greater_than"], 
                                              format_func=lambda x: "Signal RSI < Comparison RSI" if x == "less_than" else "Signal RSI > Comparison RSI",
                                              help="Choose the RSI comparison condition: 'Signal RSI < Comparison RSI' means buy when signal RSI is lower, 'Signal RSI > Comparison RSI' means buy when signal RSI is higher.")
    
    # Target and fallback tickers
    col1, col2 = st.sidebar.columns(2)
    with col1:
        target_ticker = st.sidebar.text_input("Target Ticker", value="TQQQ", 
                                             help="The ticker to buy when the RSI comparison condition is met.")
    with col2:
        fallback_ticker = st.sidebar.text_input("Fallback Ticker", value="BIL", 
                                               help="The ticker to hold when the RSI comparison condition is not met. Defaults to BIL (cash equivalent).")
    
    comparison = comparison_operator  # Use the selected comparison operator

# Set default target ticker based on RSI condition
if comparison == "less_than":
    default_target = "TQQQ"
else:
    default_target = "VIXY"

if analysis_mode == "RSI Threshold":
    target_ticker = st.sidebar.text_input("Target Ticker", value=default_target, help="The ticker to buy/sell based on the signal ticker's RSI. This is what you'll actually be trading.")

# Benchmark selection
st.sidebar.subheader("📊 Benchmark Configuration")

# Conditional benchmark (under same RSI conditions)
benchmark_options = ["SPY", "BIL", "TQQQ"]
benchmark_ticker = st.sidebar.selectbox("Conditional Benchmark", 
                                       benchmark_options, 
                                       format_func=lambda x: {
                                           "SPY": "SPY (S&P 500)",
                                           "BIL": "BIL (Cash Equivalent)", 
                                           "TQQQ": "TQQQ (3x Nasdaq-100)"
                                       }.get(x, x),
                                       help="Choose your conditional benchmark. This ticker will be traded using the same RSI conditions as your strategy for fair comparison.")

# Allow custom conditional benchmark input
use_custom_benchmark = st.sidebar.checkbox("Use custom conditional benchmark ticker", help="Check this to specify a custom ticker symbol for the conditional benchmark.")

if use_custom_benchmark:
    custom_benchmark = st.sidebar.text_input("Custom Conditional Benchmark Ticker", 
                                            placeholder="e.g., QQQ, VTI, etc.",
                                            help="Enter a custom ticker symbol to use as conditional benchmark.")
else:
    custom_benchmark = ""

# Buy-and-hold benchmark
buyhold_options = ["SPY", "BIL", "TQQQ", "QQQ", "VTI"]
buyhold_benchmark = st.sidebar.selectbox("Buy-and-Hold Benchmark", 
                                         buyhold_options, 
                                         format_func=lambda x: {
                                             "SPY": "SPY (S&P 500)",
                                             "BIL": "BIL (Cash Equivalent)", 
                                             "TQQQ": "TQQQ (3x Nasdaq-100)",
                                             "QQQ": "QQQ (Nasdaq-100)",
                                             "VTI": "VTI (Total Market)"
                                         }.get(x, x),
                                         help="Choose your buy-and-hold benchmark. This ticker will be held throughout the entire period for comparison.")

# Allow custom buy-and-hold benchmark input
use_custom_buyhold = st.sidebar.checkbox("Use custom buy-and-hold benchmark ticker", help="Check this to specify a custom ticker symbol for the buy-and-hold benchmark.")

if use_custom_buyhold:
    custom_buyhold = st.sidebar.text_input("Custom Buy-and-Hold Benchmark Ticker", 
                                           placeholder="e.g., QQQ, VTI, etc.",
                                           help="Enter a custom ticker symbol to use as buy-and-hold benchmark.")
else:
    custom_buyhold = ""

# RSI Threshold and Range options (only for RSI Threshold mode)
if analysis_mode == "RSI Threshold":
    # Set default RSI threshold based on condition
    if comparison == "less_than":
        default_threshold = 40.0
        st.sidebar.write("Buy signals: Signal RSI ≤ threshold")
    else:
        default_threshold = 60.0
        st.sidebar.write("Buy signals: Signal RSI ≥ threshold")

    rsi_threshold = st.sidebar.number_input("RSI Threshold", min_value=0.0, max_value=100.0, value=float(default_threshold), step=0.5, help="The RSI threshold to test. For 'RSI ≤ threshold', try 20-40. For 'RSI ≥ threshold', try 60-80.")

    # RSI Range options
    st.sidebar.subheader("📊 RSI Range Options")
    use_custom_range = st.sidebar.checkbox("Use custom RSI range", help="Check this to specify a custom range of RSI values to test instead of the default range.")

    # Set default ranges based on condition
    if comparison == "less_than":
        default_min, default_max = 0.0, 50.0  # Test from 0 to 50 for less than conditions
    else:
        default_min, default_max = 50.0, 100.0  # Test from 50 to 100 for greater than conditions

    if use_custom_range:
        col1, col2 = st.sidebar.columns(2)
        with col1:
            rsi_min = st.sidebar.number_input("RSI Range Min", min_value=0.0, max_value=100.0, value=float(default_min), step=0.5, help="The lowest RSI threshold to test.")
        with col2:
            rsi_max = st.sidebar.number_input("RSI Range Max", min_value=0.0, max_value=100.0, value=float(default_max), step=0.5, help="The highest RSI threshold to test.")
        
        if rsi_min >= rsi_max:
            st.sidebar.error("RSI Min must be less than RSI Max")
    else:
        rsi_min, rsi_max = default_min, default_max
else:
    # RSI Comparison mode - set default values for compatibility
    rsi_threshold = 50.0  # Not used in comparison mode
    rsi_min, rsi_max = 0.0, 100.0  # Not used in comparison mode
    use_custom_range = False
    rsi_period = signal_rsi_period  # Use signal RSI period for display purposes

# Date range selection
st.sidebar.subheader("📅 Date Range")
use_date_range = st.sidebar.checkbox("Use custom date range", help="Check this to specify your own start and end dates. If unchecked, the app will use all available data.")

if use_date_range:
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=datetime(2020, 1, 1), help="The first date to include in your analysis. Earlier dates give more data but may not reflect current market conditions.")
    with col2:
        end_date = st.date_input("End Date", value=datetime.now(), help="The last date to include in your analysis. More recent dates may be more relevant to current market conditions.")
    
    if start_date >= end_date:
        st.sidebar.error("Start date must be before end date")
        start_date, end_date = None, None
else:
    start_date, end_date = None, None
    st.sidebar.info("Using maximum available data")

# Exclude date windows
st.sidebar.subheader("🚫 Exclude Date Windows")
use_exclusions = st.sidebar.checkbox("Exclude specific date windows", help="Check this to exclude specific periods like the COVID crash from your analysis.")

if use_exclusions:
    # Initialize exclusions in session state if not exists
    if 'date_exclusions' not in st.session_state:
        st.session_state.date_exclusions = []
    
    # Display existing exclusions
    if st.session_state.date_exclusions:
        st.sidebar.write("**Current Exclusions:**")
        for i, exclusion in enumerate(st.session_state.date_exclusions):
            with st.sidebar.container():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"• {exclusion['start']} to {exclusion['end']}")
                with col2:
                    if st.button(f"🗑️", key=f"remove_exclusion_{i}"):
                        st.session_state.date_exclusions.pop(i)
                        st.rerun()
    
    # Add new exclusion
    with st.sidebar.expander("➕ Add Exclusion Window", expanded=False):
        st.write("**Add a new date exclusion:**")
        
        col1, col2 = st.columns(2)
        with col1:
            exclusion_start = st.date_input("Exclusion Start Date", 
                                          value=datetime(2020, 2, 20), 
                                          key="exclusion_start",
                                          help="Start date of the period to exclude.")
        with col2:
            exclusion_end = st.date_input("Exclusion End Date", 
                                        value=datetime(2020, 4, 7), 
                                        key="exclusion_end",
                                        help="End date of the period to exclude.")
        
        # Add exclusion button
        if st.button("➕ Add Exclusion", key="add_exclusion"):
            if exclusion_start < exclusion_end:
                new_exclusion = {
                    'start': exclusion_start,
                    'end': exclusion_end
                }
                st.session_state.date_exclusions.append(new_exclusion)
                st.rerun()
            else:
                st.error("Start date must be before end date")
    
    # Clear all exclusions button
    if st.session_state.date_exclusions:
        if st.sidebar.button("🗑️ Clear All Exclusions", type="secondary"):
            st.session_state.date_exclusions = []
            st.rerun()
else:
    # Clear exclusions if feature is disabled
    if 'date_exclusions' in st.session_state:
        st.session_state.date_exclusions = []

# Add the Run Analysis button to the sidebar
st.sidebar.markdown("---")

# Determine which benchmark to use
final_benchmark_ticker = custom_benchmark.strip() if custom_benchmark.strip() else benchmark_ticker

if st.sidebar.button("🚀 Run RSI Analysis", type="primary", use_container_width=True):
    st.sidebar.info("🔄 Starting analysis...")
    if (not use_date_range or (start_date and end_date and start_date < end_date)):
        try:
            exclusions = st.session_state.get('date_exclusions', []) if use_exclusions else None
            
            if analysis_mode == "RSI Threshold":
                # Ensure RSI range variables are defined
                rsi_min_to_use = rsi_min if 'rsi_min' in locals() else 0.0
                rsi_max_to_use = rsi_max if 'rsi_max' in locals() else 100.0
                
                # RSI Threshold mode validation
                if use_custom_range and (not rsi_min_to_use or not rsi_max_to_use or rsi_min_to_use >= rsi_max_to_use):
                    st.sidebar.error("Please ensure RSI Min is less than RSI Max")
                else:
                    # Ensure rsi_period is defined for RSI Threshold mode
                    rsi_period_to_use = rsi_period if 'rsi_period' in locals() else 10
                    # Get RSI range parameters
                    rsi_min_to_use = rsi_min_to_use if 'rsi_min_to_use' in locals() else 0.0
                    rsi_max_to_use = rsi_max_to_use if 'rsi_max_to_use' in locals() else 100.0
                    results_df, benchmark, data_messages = run_rsi_analysis(signal_ticker, target_ticker, rsi_threshold, comparison, start_date, end_date, rsi_period_to_use, rsi_method, final_benchmark_ticker, use_quantstats, st.session_state.get('preconditions', []), exclusions, rsi_min_to_use, rsi_max_to_use)
            else:
                # RSI Comparison mode
                # Get buy-and-hold benchmark
                final_buyhold_benchmark = custom_buyhold if use_custom_buyhold and custom_buyhold.strip() else buyhold_benchmark
                # Ensure RSI periods are defined for RSI Comparison mode
                signal_rsi_period_to_use = signal_rsi_period if 'signal_rsi_period' in locals() else 10
                comparison_rsi_period_to_use = comparison_rsi_period if 'comparison_rsi_period' in locals() else 10
                results_df, benchmark, buyhold_benchmark, data_messages = run_rsi_comparison_analysis(signal_ticker, comparison_ticker, target_ticker, fallback_ticker, signal_rsi_period_to_use, comparison_rsi_period_to_use, start_date, end_date, rsi_method, final_benchmark_ticker, final_buyhold_benchmark, use_quantstats, st.session_state.get('preconditions', []), exclusions)
            
            if results_df is not None and benchmark is not None and not results_df.empty:
                # Store analysis results in session state
                st.session_state['results_df'] = results_df
                st.session_state['benchmark'] = benchmark
                st.session_state['signal_data'] = get_stock_data(signal_ticker, start_date, end_date, exclusions)
                st.session_state['benchmark_data'] = get_stock_data(final_benchmark_ticker, start_date, end_date, exclusions)
                # Store the correct RSI period based on analysis mode
                if analysis_mode == "RSI Threshold":
                    st.session_state['rsi_period'] = rsi_period_to_use
                else:
                    st.session_state['rsi_period'] = signal_rsi_period_to_use
                
                st.session_state['comparison'] = comparison
                st.session_state['benchmark_ticker'] = final_benchmark_ticker
                st.session_state['analysis_mode'] = analysis_mode
                if analysis_mode == "RSI Comparison":
                    st.session_state['comparison_ticker'] = comparison_ticker
                    st.session_state['buyhold_benchmark'] = buyhold_benchmark
                    st.session_state['buyhold_benchmark_ticker'] = final_buyhold_benchmark
                st.session_state['analysis_completed'] = True
                st.session_state['data_messages'] = data_messages
                
                st.sidebar.success("✅ Analysis completed successfully!")
            else:
                st.sidebar.error("❌ Analysis failed - no results generated")
                
        except Exception as e:
            st.sidebar.error(f"❌ Error during analysis: {str(e)}")
    else:
        if use_date_range and (not start_date or not end_date or start_date >= end_date):
            st.sidebar.error("Please ensure start date is before end date")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("⚙️ Analysis Configuration")
    
    # Display preconditions first
    if st.session_state.get('preconditions'):
        st.write("**Preconditions:**")
        for i, precondition in enumerate(st.session_state.preconditions):
            if 'type' in precondition and precondition['type'] == 'comparison':
                signal_period = precondition.get('signal_rsi_period', 10)
                comparison_period = precondition.get('comparison_rsi_period', 10)
                target_ticker = precondition.get('target_ticker', 'N/A')
                fallback_ticker = precondition.get('fallback_ticker', 'N/A')
                comparison_operator = precondition.get('comparison_operator', 'less_than')
                
                # Format comparison operator
                operator_symbol = ">" if comparison_operator == "greater_than" else "<"
                
                # Format fallback display
                if fallback_ticker == "Main Signal":
                    fallback_display = "Main Signal Output"
                elif fallback_ticker.startswith("Precondition"):
                    fallback_display = f"{fallback_ticker} Output"
                else:
                    fallback_display = fallback_ticker
                
                st.write(f"  • {precondition['signal_ticker']} {signal_period}d RSI {operator_symbol} {precondition['comparison_ticker']} {comparison_period}d RSI → {target_ticker} / {fallback_display}")
            else:
                # Handle legacy format or threshold type
                comparison_symbol = "≤" if precondition.get('comparison') == "less_than" else "≥"
                rsi_period = precondition.get('rsi_period', 10)
                st.write(f"  • {precondition['signal_ticker']} {rsi_period}d RSI {comparison_symbol} {precondition.get('threshold', 'N/A')}")
    
    st.write(f"**Analysis Mode:** {analysis_mode}")
    st.write(f"**Signal Ticker:** {signal_ticker} (generates RSI signals)")
    st.write(f"**Target Ticker:** {target_ticker} (buy/sell based on signals)")
    
    if analysis_mode == "RSI Comparison":
        st.write(f"**Comparison Ticker:** {comparison_ticker} (RSI comparison target)")
    
    # Display benchmark information
    if custom_benchmark.strip():
        benchmark_display = custom_benchmark.strip()
        benchmark_description = "Custom Benchmark"
    else:
        benchmark_display = benchmark_ticker
        benchmark_description = {
            "SPY": "S&P 500",
            "BIL": "Cash Equivalent", 
            "TQQQ": "3x Nasdaq-100"
        }.get(benchmark_ticker, benchmark_ticker)
    
    st.write(f"**Benchmark:** {benchmark_display} ({benchmark_description})")
    st.write(f"**RSI Period:** {rsi_period}-day RSI")
    
    if analysis_mode == "RSI Threshold":
        if use_custom_range:
            st.write(f"**RSI Condition:** {signal_ticker} RSI {'≤' if comparison == 'less_than' else '≥'} {rsi_threshold} (testing {rsi_min} to {rsi_max})")
        else:
            st.write(f"**RSI Condition:** {signal_ticker} RSI {'≤' if comparison == 'less_than' else '≥'} {rsi_threshold} (testing {rsi_min} to {rsi_max})")
    else:
        operator_symbol = ">" if comparison == "greater_than" else "<"
        st.write(f"**RSI Condition:** {signal_ticker} RSI {operator_symbol} {comparison_ticker} RSI")
    
    if use_date_range and start_date and end_date:
        st.write(f"**Date Range:** {start_date} to {end_date}")
    else:
        st.write(f"**Date Range:** Maximum available data")
    
    # Display exclusions
    if use_exclusions and st.session_state.get('date_exclusions'):
        st.write("**Excluded Periods:**")
        for exclusion in st.session_state.date_exclusions:
            st.write(f"  • {exclusion['start']} to {exclusion['end']}")

with col2:
    st.subheader("📋 Signal Logic")
    
    # Build the signal logic description
    if st.session_state.get('preconditions'):
        st.write("**Preconditions (ALL must be true):**")
        for precondition in st.session_state.preconditions:
            if precondition.get('type') == 'comparison':
                # RSI comparison precondition
                signal_period = precondition.get('signal_rsi_period', 10)
                comparison_period = precondition.get('comparison_rsi_period', 10)
                comparison_operator = precondition.get('comparison_operator', 'less_than')
                operator_symbol = ">" if comparison_operator == "greater_than" else "<"
                st.write(f"  • {precondition['signal_ticker']} {signal_period}d RSI {operator_symbol} {precondition['comparison_ticker']} {comparison_period}d RSI")
            else:
                # RSI threshold precondition (legacy format)
                comparison_symbol = "≤" if precondition.get('comparison') == "less_than" else "≥"
                st.write(f"  • {precondition['signal_ticker']} RSI {comparison_symbol} {precondition.get('threshold', 'N/A')}")
        st.write("**Main Signal:**")
    
    if analysis_mode == "RSI Threshold":
        if comparison == "less_than":
            st.info(f"🔵 BUY {target_ticker} when {signal_ticker} {rsi_period}-day RSI ≤ threshold\n\n📈 SELL {target_ticker} when {signal_ticker} {rsi_period}-day RSI > threshold")
        else:
            st.info(f"🔵 BUY {target_ticker} when {signal_ticker} {rsi_period}-day RSI ≥ threshold\n\n📈 SELL {target_ticker} when {signal_ticker} {rsi_period}-day RSI < threshold")
    else:
        operator_symbol = ">" if comparison == "greater_than" else "<"
        opposite_operator = "<=" if comparison == "greater_than" else ">="
        st.info(f"🔵 BUY {target_ticker} when {signal_ticker} {rsi_period}-day RSI {operator_symbol} {comparison_ticker} {rsi_period}-day RSI\n\n📈 SELL {target_ticker} when {signal_ticker} {rsi_period}-day RSI {opposite_operator} {comparison_ticker} {rsi_period}-day RSI")

# Check if we have stored analysis results
if 'analysis_completed' in st.session_state and st.session_state['analysis_completed']:
    # Display stored results
    results_df = st.session_state['results_df']
    benchmark = st.session_state['benchmark']
    
    st.success("✅ Analysis completed successfully!")
    
    # Check for data quality issues
    insufficient_data_count = results_df.get('insufficient_data', pd.Series([False] * len(results_df))).sum()
    low_trade_count = (results_df['Total_Trades'] < 5).sum()
    extreme_rsi_count = 0
    
    # Count extreme RSI values (very high or very low depending on comparison)
    # Only check for RSI thresholds in RSI Threshold mode
    extreme_rsi_count = 0
    if 'RSI_Threshold' in results_df.columns:
        if st.session_state.get('comparison') == 'greater_than':
            extreme_rsi_count = (results_df['RSI_Threshold'] >= 85).sum()
        else:
            extreme_rsi_count = (results_df['RSI_Threshold'] <= 15).sum()
    
    if insufficient_data_count > 0 or low_trade_count > 0 or extreme_rsi_count > 0:
        st.warning("⚠️ **Data Quality Warnings:**")
        if insufficient_data_count > 0:
            st.write(f"• {insufficient_data_count} RSI thresholds had insufficient data for reliable statistical testing")
        if low_trade_count > 0:
            st.write(f"• {low_trade_count} RSI thresholds generated fewer than 5 trades")
        if extreme_rsi_count > 0:
            st.write(f"• {extreme_rsi_count} RSI thresholds are at extreme values (may have limited historical occurrences)")
        st.write("**Recommendation:** Focus on RSI thresholds with more trades and higher confidence levels for more reliable results.")
    
    # Display results based on analysis mode
    if 'RSI_Threshold' in results_df.columns:
        # RSI Threshold mode - show full analysis with filters and charts
        st.subheader("📊 RSI Analysis Results")
        st.info("💡 **What this shows:** This table displays all the RSI thresholds tested and their performance metrics. Each row represents a different RSI level and shows how well that strategy performed.")
        
        # Format the dataframe for display
        display_df = results_df.copy()
    else:
        # RSI Comparison mode - show simplified results
        st.subheader("📊 RSI Comparison Results")
        st.info("💡 **What this shows:** This table displays the performance of your RSI comparison strategy against the benchmark.")
        
        # Format the dataframe for display
        display_df = results_df.copy()
    
    # Check if required columns exist before formatting
    required_columns = ['Win_Rate', 'Avg_Return', 'Median_Return', 'Benchmark_Avg_Return', 'Benchmark_Median_Return', 'Total_Return', 'annualized_return', 
                      'Sortino_Ratio', 'Avg_Hold_Days', 'Return_Std', 'Best_Return', 
                      'Worst_Return', 'Final_Equity', 'confidence_level', 'significant', 'effect_size']
    
    missing_columns = [col for col in required_columns if col not in results_df.columns]
    if missing_columns:
        st.error(f"Missing columns in results: {missing_columns}")
        st.stop()
    
    # Format the columns for display
    display_df['Win_Rate'] = display_df['Win_Rate'].apply(lambda x: f"{x:.1%}" if isinstance(x, (int, float)) else x)
    display_df['Avg_Return'] = display_df['Avg_Return'].apply(lambda x: f"{x:.3%}" if isinstance(x, (int, float)) else x)
    display_df['Median_Return'] = display_df['Median_Return'].apply(lambda x: f"{x:.3%}" if isinstance(x, (int, float)) else x)
    display_df['Benchmark_Avg_Return'] = display_df['Benchmark_Avg_Return'].apply(lambda x: f"{x:.3%}" if isinstance(x, (int, float)) else x)
    display_df['Benchmark_Median_Return'] = display_df['Benchmark_Median_Return'].apply(lambda x: f"{x:.3%}" if isinstance(x, (int, float)) else x)
    display_df['Total_Return'] = display_df['Total_Return'].apply(lambda x: f"{x:.3%}" if isinstance(x, (int, float)) else x)
    display_df['Annualized_Return'] = display_df['annualized_return'].apply(lambda x: f"{x:.3%}" if isinstance(x, (int, float)) else x)
    display_df['Sortino_Ratio'] = display_df['Sortino_Ratio'].apply(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) and not np.isinf(x) else "∞" if isinstance(x, (int, float)) and np.isinf(x) else x)
    display_df['Sharpe_Ratio'] = display_df['sharpe_ratio'].apply(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) and not np.isinf(x) else "∞" if isinstance(x, (int, float)) and np.isinf(x) else x)
    display_df['Calmar_Ratio'] = display_df['calmar_ratio'].apply(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) and not np.isinf(x) else "∞" if isinstance(x, (int, float)) and np.isinf(x) else x)
    display_df['Max_Drawdown'] = display_df['max_drawdown'].apply(lambda x: f"{x:.3%}" if isinstance(x, (int, float)) else x)
    display_df['VaR_95'] = display_df['var_95'].apply(lambda x: f"{x:.3%}" if isinstance(x, (int, float)) else x)
    display_df['Avg_Hold_Days'] = display_df['Avg_Hold_Days'].apply(lambda x: f"{x:.1f}" if isinstance(x, (int, float)) else x)
    display_df['Return_Std'] = display_df['Return_Std'].apply(lambda x: f"{x:.3%}" if isinstance(x, (int, float)) else x)
    display_df['Best_Return'] = display_df['Best_Return'].apply(lambda x: f"{x:.3%}" if isinstance(x, (int, float)) else x)
    display_df['Worst_Return'] = display_df['Worst_Return'].apply(lambda x: f"{x:.3%}" if isinstance(x, (int, float)) else x)
    display_df['Final_Equity'] = display_df['Final_Equity'].apply(lambda x: f"{x:.3f}" if isinstance(x, (int, float)) else x)
    display_df['Confidence_Level'] = display_df['confidence_level'].apply(lambda x: f"{x:.1f}%" if isinstance(x, (int, float)) else x)
    display_df['Significant'] = display_df['significant'].apply(lambda x: "✓" if x else "✗")
    display_df['Effect_Size'] = display_df['effect_size'].apply(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x)
    
    # Add p-value to display columns
    display_df['P_Value'] = display_df['p_value'].apply(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)
    
    # Display comprehensive statistical summary
    st.subheader("📊 Statistical Significance Analysis")
    
    # Get the best performing result for detailed analysis
    if 'RSI_Threshold' in display_df.columns:
        # For RSI Threshold mode, find the best result by total return
        best_idx = display_df['Total_Return'].str.replace('%', '').astype(float).idxmax()
    else:
        # For RSI Comparison mode, use the single result
        best_idx = 0
    
    best_result = display_df.iloc[best_idx]
    
    # Create a comprehensive summary
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Trades", f"{best_result['Total_Trades']}")
        st.metric("Win Rate", best_result['Win_Rate'])
        st.metric("Average Return per Trade", best_result['Avg_Return'])
        st.metric("Median Return per Trade", best_result['Median_Return'])
    
    with col2:
        st.metric("Total Return", best_result['Total_Return'])
        st.metric("Annualized Return", best_result['Annualized_Return'])
        st.metric("Best Single Trade", best_result['Best_Return'])
        st.metric("Worst Single Trade", best_result['Worst_Return'])
    
    with col3:
        st.metric("P-Value", best_result['P_Value'])
        st.metric("Significant", best_result['Significant'])
        st.metric("Effect Size", best_result['Effect_Size'])
        st.metric("Confidence Level", best_result['Confidence_Level'])
    
    # Risk metrics
    st.subheader("📈 Risk Metrics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Sortino Ratio", best_result['Sortino_Ratio'])
        st.metric("Sharpe Ratio", best_result['Sharpe_Ratio'])
        st.metric("Calmar Ratio", best_result['Calmar_Ratio'])
    
    with col2:
        st.metric("Max Drawdown", best_result['Max_Drawdown'])
        st.metric("Value at Risk (95%)", best_result['VaR_95'])
        st.metric("Return Standard Deviation", best_result['Return_Std'])
    
    with col3:
        st.metric("Average Hold Days", best_result['Avg_Hold_Days'])
        st.metric("Final Equity", best_result['Final_Equity'])
        if 'RSI_Threshold' in display_df.columns:
            st.metric("RSI Threshold", f"{best_result['RSI_Threshold']}")
    
    # Display equity curve
    st.subheader("📈 Equity Curve")
    if 'equity_curve' in results_df.columns:
        # Get the equity curve for the best result
        best_equity_curve = results_df.iloc[best_idx]['equity_curve']
        if hasattr(best_equity_curve, 'index') and len(best_equity_curve) > 0:
            fig = go.Figure()
            
            # Add strategy equity curve
            fig.add_trace(go.Scatter(
                x=best_equity_curve.index,
                y=best_equity_curve.values,
                mode='lines',
                name='Strategy',
                line=dict(color='blue', width=2)
            ))
            
            # Add benchmark equity curve if available
            if 'benchmark_equity_curve' in results_df.columns:
                benchmark_equity_curve = results_df.iloc[best_idx]['benchmark_equity_curve']
                if hasattr(benchmark_equity_curve, 'index') and len(benchmark_equity_curve) > 0:
                    fig.add_trace(go.Scatter(
                        x=benchmark_equity_curve.index,
                        y=benchmark_equity_curve.values,
                        mode='lines',
                        name='Benchmark',
                        line=dict(color='red', width=2)
                    ))
            
            fig.update_layout(
                title="Strategy vs Benchmark Performance",
                xaxis_title="Date",
                yaxis_title="Cumulative Return",
                hovermode='x unified',
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ Equity curve data not available for display.")
    else:
        st.warning("⚠️ Equity curve data not available for display.")
    
    # Drop the equity_curve and trades columns for display
    # Use different column sets based on analysis mode
    if 'RSI_Threshold' in display_df.columns:
        # RSI Threshold mode
        display_cols = ['RSI_Threshold', 'Total_Trades', 'Win_Rate', 'Avg_Return', 'Median_Return', 'Benchmark_Avg_Return', 'Benchmark_Median_Return',
                       'Total_Return', 'Annualized_Return', 'Sortino_Ratio', 'Sharpe_Ratio', 'Calmar_Ratio', 'Final_Equity', 'Avg_Hold_Days', 
                       'Return_Std', 'Best_Return', 'Worst_Return', 'Max_Drawdown', 'VaR_95', 'Confidence_Level', 'Significant', 'Effect_Size', 'P_Value']
    else:
        # RSI Comparison mode
        display_cols = ['Signal_Ticker', 'Comparison_Ticker', 'Target_Ticker', 'Analysis_Type', 'Total_Trades', 'Win_Rate', 'Avg_Return', 'Median_Return', 'Benchmark_Avg_Return', 'Benchmark_Median_Return',
                       'Total_Return', 'Annualized_Return', 'Sortino_Ratio', 'Sharpe_Ratio', 'Calmar_Ratio', 'Final_Equity', 'Avg_Hold_Days', 
                       'Return_Std', 'Best_Return', 'Worst_Return', 'Max_Drawdown', 'VaR_95', 'Confidence_Level', 'Significant', 'Effect_Size', 'P_Value']
    
    # Check if all display columns exist
    missing_display_cols = [col for col in display_cols if col not in display_df.columns]
    if missing_display_cols:
        st.error(f"Missing display columns: {missing_display_cols}")
        st.stop()
    
    # Display table based on analysis mode
    if 'RSI_Threshold' in display_df.columns:
        # RSI Threshold mode - show filterable table
        with st.expander("📊 Table of Results", expanded=False):
            st.subheader("🔍 Filter Results")
        
            # Create filter columns
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                rsi_min_filter = st.number_input(
                    "Min RSI Threshold:",
                    min_value=0.0,
                    max_value=100.0,
                    value=0.0,
                    step=0.5,
                    help="Minimum RSI threshold to include in results."
                )
                rsi_max_filter = st.number_input(
                    "Max RSI Threshold:",
                    min_value=0.0,
                    max_value=100.0,
                    value=100.0,
                    step=0.5,
                    help="Maximum RSI threshold to include in results."
                )
            
            with col2:
                confidence_min_filter = st.number_input(
                    "Min Confidence Level (%):",
                    min_value=0.0,
                    max_value=100.0,
                    value=0.0,
                    step=1.0,
                    help="Minimum confidence level to include in results."
                )
                confidence_max_filter = st.number_input(
                    "Max Confidence Level (%):",
                    min_value=0.0,
                    max_value=100.0,
                    value=100.0,
                    step=1.0,
                    help="Maximum confidence level to include in results."
                )
            
            with col3:
                min_trades_filter = st.number_input(
                    "Min Total Trades:",
                    min_value=0,
                    value=0,
                    help="Minimum number of trades to include in results."
                )
                min_win_rate_filter = st.number_input(
                    "Min Win Rate (%):",
                    min_value=0.0,
                    max_value=100.0,
                    value=0.0,
                    step=1.0,
                    help="Minimum win rate percentage to include in results."
                )
            
            with col4:
                min_avg_return_filter = st.number_input(
                    "Min Avg Return (%):",
                    min_value=-100.0,
                    max_value=100.0,
                    value=-100.0,
                    step=0.1,
                    help="Minimum average return percentage to include in results."
                )
                min_total_return_filter = st.number_input(
                    "Min Total Return (%):",
                    min_value=-100.0,
                    max_value=100.0,
                    value=-100.0,
                    step=0.1,
                    help="Minimum total return percentage to include in results."
                )
            
            col5, col6, col7, col8 = st.columns(4)
            
            with col5:
                min_annualized_return_filter = st.number_input(
                    "Min Annualized Return (%):",
                    min_value=-100.0,
                    max_value=100.0,
                    value=-100.0,
                    step=0.1,
                    help="Minimum annualized return percentage to include in results."
                )
            
            with col6:
                min_sortino_filter = st.number_input(
                    "Min Sortino Ratio:",
                    min_value=-10.0,
                    max_value=10.0,
                    value=-10.0,
                    step=0.1,
                    help="Minimum Sortino ratio to include in results."
                )
            
            with col7:
                significance_filter = st.selectbox(
                    "Significance:",
                    ["All", "Significant Only", "Non-Significant Only"],
                    help="Filter by statistical significance."
                )
                max_p_value_filter = st.number_input(
                    "Max P-Value:",
                    min_value=0.0,
                    max_value=1.0,
                    value=1.0,
                    step=0.001,
                    help="Maximum p-value to include in results (lower = more significant)."
                )
            
            with col8:
                if st.button("Clear All Filters", type="secondary"):
                    st.rerun()
            
            # Apply filters to the display dataframe
            filtered_df = display_df.copy()
            
            # Only apply RSI threshold filters in RSI Threshold mode
            if 'RSI_Threshold' in filtered_df.columns:
                filtered_df = filtered_df[
                    (filtered_df['RSI_Threshold'] >= rsi_min_filter) & 
                    (filtered_df['RSI_Threshold'] <= rsi_max_filter)
                ]
            
            filtered_df = filtered_df[
                (filtered_df['Confidence_Level'].str.replace('%', '').astype(float) >= confidence_min_filter) & 
                (filtered_df['Confidence_Level'].str.replace('%', '').astype(float) <= confidence_max_filter)
            ]
            filtered_df = filtered_df[filtered_df['Total_Trades'] >= min_trades_filter]
            filtered_df = filtered_df[
                filtered_df['Win_Rate'].str.replace('%', '').astype(float) >= min_win_rate_filter
            ]
            filtered_df = filtered_df[
                filtered_df['Avg_Return'].str.replace('%', '').astype(float) >= min_avg_return_filter
            ]
            filtered_df = filtered_df[
                filtered_df['Total_Return'].str.replace('%', '').astype(float) >= min_total_return_filter
            ]
            filtered_df = filtered_df[
                filtered_df['Annualized_Return'].str.replace('%', '').astype(float) >= min_annualized_return_filter
            ]
            filtered_df = filtered_df[
                filtered_df['Sortino_Ratio'].apply(lambda x: float(x) if x != "∞" else 999) >= min_sortino_filter
            ]
            if significance_filter == "Significant Only":
                filtered_df = filtered_df[filtered_df['Significant'] == "✓"]
            elif significance_filter == "Non-Significant Only":
                filtered_df = filtered_df[filtered_df['Significant'] == "✗"]
            filtered_df = filtered_df[
                filtered_df['P_Value'].astype(float) <= max_p_value_filter
            ]
            
            # Display the filtered results table
            st.subheader(f"📊 RSI Analysis Results ({len(filtered_df)} signals)")
            st.dataframe(filtered_df[display_cols], use_container_width=True)
    else:
        # RSI Comparison mode - show simple table without filters
        st.subheader("📊 Results Table")
        # Create a simple display without filters
        filtered_df = display_df.copy()
        st.dataframe(filtered_df[display_cols], use_container_width=True)
    
    # Find best strategies (needed for subsequent sections)
    # Handle different display modes
    if 'RSI_Threshold' in display_df.columns:
        # RSI Threshold mode - show full analysis
        best_sortino_idx = filtered_df['Sortino_Ratio'].idxmax()
        best_annualized_idx = filtered_df['annualized_return'].idxmax()
        best_winrate_idx = filtered_df['Win_Rate'].idxmax()
        best_total_return_idx = filtered_df['Total_Return'].idxmax()
        
        # Statistical Significance Analysis - Full analysis for RSI Threshold mode
        with st.expander("📊 Statistical Significance Analysis", expanded=True):
            st.subheader("📊 Statistical Significance Analysis")
            
            # Get benchmark info
            stored_benchmark_ticker = st.session_state.get('benchmark_ticker', 'SPY')
            benchmark_description = {
                "SPY": "S&P 500",
                "BIL": "Cash Equivalent", 
                "TQQQ": "3x Nasdaq-100"
            }.get(stored_benchmark_ticker, "Custom Benchmark")
            benchmark_name = f"{stored_benchmark_ticker} ({benchmark_description})"
            
            # Get significant signals
            significant_signals = filtered_df[filtered_df['Significant'] == "✓"]
            valid_signals = filtered_df[filtered_df['Total_Trades'] > 0]
            
            if len(significant_signals) > 0:
                st.success(f"✅ **{len(significant_signals)} Statistically Significant Signals Found**")
                st.write(f"Your RSI strategy shows statistically significant outperformance against {benchmark_name} for {len(significant_signals)} RSI thresholds.")
                
                # Show confidence level distribution
                st.subheader("📊 Confidence Level Distribution")
                st.info("💡 **What this shows:** This chart displays how confidence levels vary across different RSI thresholds. Higher confidence levels indicate stronger statistical evidence that your strategy outperforms the benchmark.")
                
                # Create confidence level chart
                fig_confidence = go.Figure()
                
                # Add points for significant signals (green)
                significant_data = valid_signals[valid_signals['Significant'] == "✓"]
                if not significant_data.empty:
                    fig_confidence.add_trace(go.Scatter(
                        x=significant_data['RSI_Threshold'],
                        y=significant_data['Confidence_Level'].str.replace('%', '').astype(float),
                        mode='markers',
                        name='Significant Signals',
                        marker=dict(color='green', size=8),
                        hovertemplate='<b>RSI %{x}</b><br>' +
                                    'Confidence: %{y:.1f}%<br>' +
                                    'Significant: ✓<extra></extra>'
                    ))
                
                # Add points for non-significant signals (red)
                non_significant_data = valid_signals[valid_signals['Significant'] == "✗"]
                if not non_significant_data.empty:
                    fig_confidence.add_trace(go.Scatter(
                        x=non_significant_data['RSI_Threshold'],
                        y=non_significant_data['Confidence_Level'].str.replace('%', '').astype(float),
                        mode='markers',
                        name='Non-Significant Signals',
                        marker=dict(color='red', size=8),
                        hovertemplate='<b>RSI %{x}</b><br>' +
                                    'Confidence: %{y:.1f}%<br>' +
                                    'Significant: ✗<extra></extra>'
                    ))
                
                # Add reference lines
                fig_confidence.add_hline(y=95, line_dash="dash", line_color="red", 
                                       annotation_text="95% Confidence")
                fig_confidence.add_hline(y=80, line_dash="dash", line_color="orange", 
                                       annotation_text="80% Confidence")
                
                fig_confidence.update_layout(
                    title="Confidence Level vs RSI Threshold",
                    xaxis_title="RSI Threshold",
                    yaxis_title="Confidence Level (%)",
                    hovermode='closest',
                    xaxis=dict(range=[0, 100]),
                    showlegend=True
                )
                
                st.plotly_chart(fig_confidence, use_container_width=True, key="confidence_chart")
                
                st.write(f"""

                **📊 Improved Statistical Analysis:**
                The confidence levels now show more realistic variation across RSI thresholds. The analysis properly calculates statistical significance for both outperformance and underperformance, avoiding artificial binary outcomes.
                
                **⚠️ Note on Extreme RSI Values:**
                At the extreme ends of RSI thresholds (very low or very high values), there are often not enough historical events to generate statistically confident results. This is why confidence levels may drop off at these extremes - the sample size becomes too small for reliable statistical analysis.
                
                **What This Chart Tells You:**
                
                **📊 X-Axis (RSI Threshold):**
                - Shows different RSI levels tested
                - Helps identify which RSI ranges are most effective
                
                **📈 Y-Axis (Confidence Level):**
                - Higher values = stronger statistical evidence
                - Above 95% = highly significant (strong evidence)
                - 80-95% = borderline significant (moderate evidence)
                - 60-80% = weak evidence
                - Below 60% = very weak or no evidence
                
                **🎯 Interpretation:**
                - **High confidence (95%+)**: Very strong evidence the signal works
                - **Moderate confidence (80-95%)**: Good evidence, worth considering
                - **Low confidence (<80%)**: Weak evidence, results may be due to chance
                - **Extreme RSI values**: Often show low confidence due to insufficient historical data

                """)
            
            else:
                st.warning("No signals reached statistical significance (p < 0.05)")
    else:
        # RSI Comparison mode - show simplified analysis
        st.subheader("📊 Statistical Significance Analysis")
        
        # Get benchmark info
        stored_benchmark_ticker = st.session_state.get('benchmark_ticker', 'SPY')
        benchmark_description = {
            "SPY": "S&P 500",
            "BIL": "Cash Equivalent", 
            "TQQQ": "3x Nasdaq-100"
        }.get(stored_benchmark_ticker, "Custom Benchmark")
        benchmark_name = f"{stored_benchmark_ticker} ({benchmark_description})"
        
        # Get the single result for RSI Comparison mode
        result = filtered_df.iloc[0]
        
        # Display significance explanation
        if result['Significant'] == "✓":
            st.success(f"✅ **Statistically Significant Result**")
            st.write(f"Your RSI comparison strategy shows statistically significant outperformance against {benchmark_name}.")
            st.write(f"**Confidence Level:** {result['Confidence_Level']}")
            st.write(f"**P-Value:** {result['P_Value']}")
            st.write(f"**Effect Size:** {result['Effect_Size']}")
            st.write("**Interpretation:** The results are unlikely to be due to chance, suggesting your strategy has genuine predictive value.")
        else:
            st.warning(f"⚠️ **Not Statistically Significant**")
            st.write(f"Your RSI comparison strategy does not show statistically significant outperformance against {benchmark_name}.")
            st.write(f"**Confidence Level:** {result['Confidence_Level']}")
            st.write(f"**P-Value:** {result['P_Value']}")
            st.write(f"**Effect Size:** {result['Effect_Size']}")
            st.write("**Interpretation:** The results may be due to chance, and the strategy's predictive value is uncertain.")
        
        # Show equity curve comparison
        st.subheader("📈 Equity Curve Comparison")
        st.info("💡 **What this shows:** This chart compares your strategy's performance against the benchmark over time.")
        
        # Create equity curve chart
        if 'equity_curve' in results_df.columns and 'benchmark_equity_curve' in results_df.columns:
            strategy_curve = results_df.iloc[0]['equity_curve']
            benchmark_curve = results_df.iloc[0]['benchmark_equity_curve']
            buyhold_curve = st.session_state.get('buyhold_benchmark', pd.Series())
            
            fig_equity = go.Figure()
            
            # Add strategy equity curve
            fig_equity.add_trace(go.Scatter(
                x=strategy_curve.index,
                y=strategy_curve.values,
                mode='lines',
                name=f'Strategy ({result["Signal_Ticker"]} vs {result["Comparison_Ticker"]})',
                line=dict(color='blue', width=2)
            ))
            
            # Add conditional benchmark equity curve (under same RSI conditions)
            fig_equity.add_trace(go.Scatter(
                x=benchmark_curve.index,
                y=benchmark_curve.values,
                mode='lines',
                name=f'Conditional Benchmark ({stored_benchmark_ticker} under same RSI conditions)',
                line=dict(color='red', width=2)
            ))
            
            # Add buy-and-hold benchmark if available
            if not buyhold_curve.empty:
                buyhold_ticker = st.session_state.get('buyhold_benchmark_ticker', 'SPY')
                fig_equity.add_trace(go.Scatter(
                    x=buyhold_curve.index,
                    y=buyhold_curve.values,
                    mode='lines',
                    name=f'Buy-and-Hold Benchmark ({buyhold_ticker})',
                    line=dict(color='green', width=2, dash='dash')
                ))
            
            fig_equity.update_layout(
                title="Strategy vs Benchmark Performance",
                xaxis_title="Date",
                yaxis_title="Cumulative Return",
                hovermode='x unified',
                showlegend=True
            )
            
            st.plotly_chart(fig_equity, use_container_width=True)
        else:
            st.warning("⚠️ Equity curve data not available for display.")
    
    # Add additional charts for RSI Threshold mode only
    if 'RSI_Threshold' in display_df.columns and len(filtered_df) > 0:
        # Get significant signals for charts
        significant_signals = filtered_df[filtered_df['Significant'] == "✓"]
        valid_signals = filtered_df[filtered_df['Total_Trades'] > 0]
        
        if len(valid_signals) > 0:
            # RSI vs Sortino Ratio Chart
            st.subheader("📊 RSI Threshold vs Sortino Ratio")
            st.info("💡 **What this shows:** This chart displays how the Sortino ratio (risk-adjusted return) varies across different RSI thresholds. Higher Sortino ratios indicate better risk-adjusted performance. Look for peaks in the chart to identify optimal RSI thresholds.")
            
            fig_sortino_rsi = go.Figure()
            
            # Add points for significant signals (green)
            significant_data = valid_signals[valid_signals['Significant'] == "✓"]
            if not significant_data.empty:
                fig_sortino_rsi.add_trace(go.Scatter(
                    x=significant_data['RSI_Threshold'],
                    y=significant_data['Sortino_Ratio'].apply(lambda x: float(x) if x != "∞" else 999),
                    mode='markers',
                    name='Significant Signals',
                    marker=dict(color='green', size=8),
                    line=dict(width=0),  # Explicitly disable lines
                    hovertemplate='<b>RSI %{x}</b><br>' +
                                'Sortino Ratio: %{y:.2f}<br>' +
                                'Significant: ✓<extra></extra>'
                ))
            
            # Add points for non-significant signals (red)
            non_significant_data = valid_signals[valid_signals['Significant'] == "✗"]
            if not non_significant_data.empty:
                fig_sortino_rsi.add_trace(go.Scatter(
                    x=non_significant_data['RSI_Threshold'],
                    y=non_significant_data['Sortino_Ratio'].apply(lambda x: float(x) if x != "∞" else 999),
                    mode='markers',
                    name='Non-Significant Signals',
                    marker=dict(color='red', size=8),
                    line=dict(width=0),  # Explicitly disable lines
                    hovertemplate='<b>RSI %{x}</b><br>' +
                                'Sortino Ratio: %{y:.2f}<br>' +
                                'Significant: ✗<extra></extra>'
                ))
            
            # Add reference line at y=0
            fig_sortino_rsi.add_hline(y=0, line_dash="dash", line_color="gray", 
                                     annotation_text="No Risk-Adjusted Return")
            
            fig_sortino_rsi.update_layout(
                title="Sortino Ratio vs RSI Threshold",
                xaxis_title="RSI Threshold",
                yaxis_title="Sortino Ratio",
                hovermode='closest',
                xaxis=dict(range=[0, 100]),
                showlegend=True
            )
            
            st.plotly_chart(fig_sortino_rsi, use_container_width=True, key="sortino_rsi_chart")
            
            # RSI vs Cumulative Return Chart
            st.subheader("📊 RSI Threshold vs Cumulative Return")
            st.info("💡 **What this shows:** This chart displays how the total cumulative return varies across different RSI thresholds. Higher cumulative returns indicate better overall performance. Look for peaks in the chart to identify optimal RSI thresholds.")
            
            # Use original numerical data for consistency
            original_filtered_data = st.session_state['results_df'][st.session_state['results_df']['RSI_Threshold'].isin(filtered_df['RSI_Threshold'])]
            original_significant_data = original_filtered_data[original_filtered_data['significant'] == True]
            original_non_significant_data = original_filtered_data[original_filtered_data['significant'] == False]
            
            fig_return_rsi = go.Figure()
            
            # Add points for significant signals (green)
            if not original_significant_data.empty:
                fig_return_rsi.add_trace(go.Scatter(
                    x=original_significant_data['RSI_Threshold'],
                    y=original_significant_data['Total_Return'],
                    mode='markers',
                    name='Significant Signals',
                    marker=dict(color='green', size=8),
                    line=dict(width=0),  # Explicitly disable lines
                    hovertemplate='<b>RSI %{x}</b><br>' +
                                'Cumulative Return: %{y:.3%}<br>' +
                                'Significant: ✓<extra></extra>'
                ))
            
            # Add points for non-significant signals (red)
            if not original_non_significant_data.empty:
                fig_return_rsi.add_trace(go.Scatter(
                    x=original_non_significant_data['RSI_Threshold'],
                    y=original_non_significant_data['Total_Return'],
                    mode='markers',
                    name='Non-Significant Signals',
                    marker=dict(color='red', size=8),
                    line=dict(width=0),  # Explicitly disable lines
                    hovertemplate='<b>RSI %{x}</b><br>' +
                                'Cumulative Return: %{y:.3%}<br>' +
                                'Significant: ✗<extra></extra>'
                ))
            
            # Add reference line at y=0
            fig_return_rsi.add_hline(y=0, line_dash="dash", line_color="gray", 
                                    annotation_text="No Return")
            
            fig_return_rsi.update_layout(
                title="Cumulative Return vs RSI Threshold",
                xaxis_title="RSI Threshold",
                yaxis_title="Cumulative Return (%)",
                hovermode='closest',
                xaxis=dict(range=[0, 100]),
                yaxis=dict(tickformat='.1%'),
                showlegend=True
            )
            
            st.plotly_chart(fig_return_rsi, use_container_width=True, key="return_rsi_chart")
            
            # Top significant signals comparison charts
            if len(significant_signals) > 0:
                st.subheader("🏆 Top Statistically Significant Signals")
                
                # Sort by total return (highest cumulative return) instead of confidence level
                # Use the original results_df for sorting since it has numerical values
                original_significant_signals = st.session_state['results_df'][st.session_state['results_df']['significant'] == True].copy()
                top_significant = original_significant_signals.nlargest(5, 'Total_Return')
                
                # Multiple Signal Comparison for Significant Signals
                st.subheader("📊 Highest Cumulative Return Significant Signals Comparison")
                st.info(f"💡 **What this shows:** This chart compares the top 5 signals with the highest cumulative returns among statistically significant signals against {benchmark_name} buy-and-hold. Each line represents a different RSI threshold that showed significant outperformance. The signals are ranked by total return, showing the highest cumulative return signals first.")
                
                # Create comparison chart with all significant signals
                fig_comparison = go.Figure()
                
                # Add benchmark buy-and-hold
                fig_comparison.add_trace(go.Scatter(
                    x=benchmark.index,
                    y=benchmark.values,
                    mode='lines',
                    name=f"{benchmark_name} Buy & Hold",
                    line=dict(color='red', width=2, dash='dash')
                ))
                
                # Add significant signals with their corresponding benchmark curves
                colors = ['blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive']
                for i, (idx, row) in enumerate(top_significant.iterrows()):
                    # Debug: Check if equity curve exists
                    if 'equity_curve' in row and row['equity_curve'] is not None:
                        color = colors[i % len(colors)]
                        
                        # Add strategy equity curve
                        fig_comparison.add_trace(go.Scatter(
                            x=row['equity_curve'].index,
                            y=row['equity_curve'].values,
                            mode='lines',
                            name=f"RSI {row['RSI_Threshold']} Strategy (Cumulative: {row['Total_Return']:.3%}, Annualized: {row['annualized_return']:.3%})",
                            line=dict(color=color, width=2)
                        ))
                        
                        # Add corresponding benchmark equity curve under same conditions
                        # We need to calculate the benchmark equity curve for this specific RSI threshold
                        signal_data = st.session_state.get('signal_data')
                        benchmark_data = st.session_state.get('benchmark_data')
                        rsi_period = st.session_state.get('rsi_period', 14)
                        comparison = st.session_state.get('comparison', 'less_than')
                        
                        if signal_data is not None and benchmark_data is not None:
                            # Calculate RSI for the signal
                            signal_rsi = calculate_rsi(signal_data, window=rsi_period, method="wilders")
                            
                            # Generate buy signals for benchmark (same as strategy)
                            if comparison == "less_than":
                                benchmark_signals = (signal_rsi <= row['RSI_Threshold']).astype(int)
                            else:  # greater_than
                                benchmark_signals = (signal_rsi >= row['RSI_Threshold']).astype(int)
                            
                            # Calculate benchmark equity curve using benchmark prices (same logic as strategy)
                            benchmark_equity_curve = pd.Series(1.0, index=benchmark_data.index)
                            current_equity = 1.0
                            in_position = False
                            entry_equity = 1.0
                            entry_price = None
                            
                            for date in benchmark_data.index:
                                current_signal = benchmark_signals[date] if date in benchmark_signals.index else 0
                                current_price = benchmark_data[date]
                                
                                if current_signal == 1 and not in_position:
                                    # Enter position
                                    in_position = True
                                    entry_equity = current_equity
                                    entry_price = current_price
                                    
                                elif current_signal == 0 and in_position:
                                    # Exit position
                                    trade_return = (current_price - entry_price) / entry_price
                                    current_equity = entry_equity * (1 + trade_return)
                                    in_position = False
                                
                                # Update equity curve
                                if in_position:
                                    current_equity = entry_equity * (current_price / entry_price)
                                
                                benchmark_equity_curve[date] = current_equity
                            
                            # Handle case where we're still in position at the end
                            if in_position:
                                final_price = benchmark_data.iloc[-1]
                                trade_return = (final_price - entry_price) / entry_price
                                current_equity = entry_equity * (1 + trade_return)
                                benchmark_equity_curve.iloc[-1] = current_equity
                            
                            # Add benchmark equity curve under same conditions
                            fig_comparison.add_trace(go.Scatter(
                                x=benchmark_equity_curve.index,
                                y=benchmark_equity_curve.values,
                                mode='lines',
                                name=f"RSI {row['RSI_Threshold']} Benchmark (same conditions)",
                                line=dict(color=color, width=1, dash='dot'),
                                visible='legendonly'  # Hidden by default
                            ))
                    else:
                        st.warning(f"No equity curve found for RSI {row['RSI_Threshold']}")
                
                # Find the shortest time period among visible curves for default scaling
                shortest_period = None
                shortest_duration = float('inf')
                
                # Check strategy curves (these are always visible)
                for i, (idx, row) in enumerate(top_significant.iterrows()):
                    if 'equity_curve' in row and row['equity_curve'] is not None:
                        curve_duration = (row['equity_curve'].index[-1] - row['equity_curve'].index[0]).days
                        if curve_duration < shortest_duration:
                            shortest_duration = curve_duration
                            shortest_period = row['equity_curve']
                
                # If no strategy curves found, use benchmark
                if shortest_period is None:
                    shortest_period = benchmark
                
                fig_comparison.update_layout(
                    title=f"Highest Cumulative Return Significant Signals Comparison vs {benchmark_name}",
                    xaxis_title="Date",
                    yaxis_title="Equity Value",
                    hovermode='x unified',
                    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                    xaxis=dict(range=[shortest_period.index[0], shortest_period.index[-1]])  # Scale to shortest period
                )
                st.plotly_chart(fig_comparison, use_container_width=True, key="most_profitable_comparison")
            else:
                st.warning("No signals reached statistical significance (p < 0.05)")
    
    # Download results
    st.subheader("📥 Download Results")
    st.info("💡 **What this does:** Download your analysis results as a CSV file that you can open in Excel or other spreadsheet programs. This includes all the performance metrics for every RSI threshold tested.")
    # Use the original column names from results_df for CSV download
    download_cols = ['RSI_Threshold', 'Total_Trades', 'Win_Rate', 'Avg_Return', 'Median_Return', 'Benchmark_Avg_Return', 'Benchmark_Median_Return',
                   'Total_Return', 'annualized_return', 'Sortino_Ratio', 'sharpe_ratio', 'calmar_ratio', 'Final_Equity', 'Avg_Hold_Days', 
                   'Return_Std', 'Best_Return', 'Worst_Return', 'max_drawdown', 'var_95', 'beta', 'alpha', 'information_ratio', 'confidence_level', 'significant', 'effect_size']
    csv = st.session_state['results_df'][download_cols].to_csv(index=False)
    filename_suffix = f"_{start_date}_{end_date}" if use_date_range and start_date and end_date else "_max_range"
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv,
        file_name=f"rsi_analysis_{signal_ticker}_{target_ticker}{filename_suffix}_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )
    
    # Statistical interpretation guide
    with st.expander("📚 Statistical Significance Guide"):
        st.write("""
        **Understanding Statistical Significance:**
        
        - **Confidence Level**: Percentage confidence that the signal outperforms the benchmark **under the same RSI conditions**
        - **P-value**: Probability of getting these results by chance (lower is better)
        - **Effect Size**: Magnitude of the difference (Cohen's d)
        - **Significant**: P-value < 0.05 (95% confidence level)
        
        **What This Measures:**
        The confidence level compares your signal (buying/selling the target ticker based on signal RSI) 
        vs. buying/selling the benchmark based on the **same signal RSI conditions**. This ensures a fair comparison 
        of whether your target ticker choice is better than the benchmark when the same RSI signals are applied.
        
        **Interpretation:**
        - ✓ **Significant**: Strong evidence your target ticker beats the benchmark under these RSI conditions
        - ✗ **Not Significant**: Results could be due to chance
        - **Effect Size**: 
          - Small: 0.2-0.5
          - Medium: 0.5-0.8  
          - Large: > 0.8
        
        **Key Metrics Explained:**
        
        **📊 Performance Metrics:**
        - **Total Return**: How much money you would have made (or lost) over the entire period
        - **Annualized Return**: The yearly return rate, useful for comparing signals over different time periods
        - **Win Rate**: Percentage of trades that were profitable
        - **Total Trades**: Number of buy/sell transactions the signal made
        - **Sortino Ratio**: Risk-adjusted return measure (higher is better, focuses on downside risk)
        - **Avg Hold Days**: Average number of days the signal held each position
        
        **📈 Statistical Metrics:**
        - **Confidence Level**: How certain we are that the signal beats the benchmark (higher % = more certain)
        - **P-value**: Probability the results happened by chance (lower = more significant)
        - **Effect Size**: How much better/worse the signal is compared to the benchmark
        - **T-statistic**: Statistical measure of the difference between signal and benchmark
        - **Power**: How likely the test is to detect a real difference if one exists
        
        **🎯 What to Look For:**
        - **High Confidence (>95%)**: Very strong evidence the signal works
        - **Low P-value (<0.05)**: Results are statistically significant
        - **Positive Effect Size**: Signal outperforms the benchmark
        - **High Win Rate**: Signal wins more often than it loses
        - **Good Sortino Ratio**: Signal has good risk-adjusted returns
        """)

else:
    st.info("ℹ️ No analysis results found. Please run the analysis first.")

st.write("---")
st.write("💡 **Tip:** Try different ticker combinations and RSI conditions to find optimal signal thresholds")



# Display data quality messages at the bottom
if 'data_messages' in st.session_state and st.session_state['data_messages']:
    st.write("---")
    st.subheader("📊 Data Quality Information")
    for message in st.session_state['data_messages']:
        st.info(message)

# Footer
st.write("---")
st.markdown("""
<div style='text-align: center; padding: 20px; color: #666;'>
    <strong>Signal Validation Tool</strong><br>
    Questions? Reach out to @Gobi on Discord
</div>
""", unsafe_allow_html=True)
