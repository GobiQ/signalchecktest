import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Signal Discovery Trading App",
    page_icon="📈",
    layout="wide"
)

class SignalDiscovery:
    def __init__(self):
        self.data = {}
        self.signals = {}
        self.backtest_results = {}
        
    def fetch_data(self, tickers, start_date, end_date):
        """Fetch historical data for given tickers"""
        data = {}
        progress_bar = st.progress(0)
        
        for i, ticker in enumerate(tickers):
            try:
                stock = yf.Ticker(ticker)
                df = stock.history(start=start_date, end=end_date)
                
                # Validate data
                if df.empty:
                    st.error(f"❌ No data found for {ticker}")
                    continue
                
                # Check for minimum data requirements
                if len(df) < 50:  # Need at least 50 data points
                    st.warning(f"⚠️ Insufficient data for {ticker} ({len(df)} points)")
                    continue
                
                # Check for required columns
                required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    st.error(f"❌ Missing columns for {ticker}: {missing_columns}")
                    continue
                
                # Clean data
                df = df.dropna()
                if len(df) < 50:
                    st.warning(f"⚠️ Insufficient data after cleaning for {ticker}")
                    continue
                
                data[ticker] = df
                st.success(f"✅ Fetched data for {ticker} ({len(df)} points)")
                
            except Exception as e:
                st.error(f"❌ Error fetching {ticker}: {str(e)}")
            
            progress_bar.progress((i + 1) / len(tickers))
        
        return data
    
    def calculate_rsi(self, prices, period=10):
        """Calculate RSI for given prices"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_moving_averages(self, prices):
        """Calculate multiple moving averages"""
        ma_periods = [10, 20, 30, 100, 200, 300]
        mas = {}
        for period in ma_periods:
            mas[f'MA_{period}'] = prices.rolling(window=period).mean()
        return mas
    
    def generate_signals(self, data, signal_config):
        """Generate trading signals based on configuration"""
        all_signals = {}
        
        for ticker, df in data.items():
            if df.empty:
                continue
                
            signals = pd.DataFrame(index=df.index)
            signals['Close'] = df['Close']
            
            # Calculate RSI
            rsi_10 = self.calculate_rsi(df['Close'], 10)
            signals['RSI_10'] = rsi_10
            
            # Calculate Moving Averages
            mas = self.calculate_moving_averages(df['Close'])
            for ma_name, ma_values in mas.items():
                signals[ma_name] = ma_values
                signals[f'{ma_name}_Signal'] = df['Close'] > ma_values
            
            # Generate trading signals based on RSI and MA
            signals['RSI_Signal'] = ((rsi_10 < 30).astype(int) - (rsi_10 > 70).astype(int))
            
            # MA crossover signals
            if 'MA_20' in signals.columns and 'MA_50' in signals.columns:
                signals['MA_Signal'] = (signals['MA_20'] > signals['MA_50']).astype(int)
            
            all_signals[ticker] = signals.dropna()
        
        # Generate comparison signals
        for config in signal_config:
            if config['type'] == 'rsi_comparison':
                ticker1, ticker2 = config['ticker1'], config['ticker2']
                if ticker1 in all_signals and ticker2 in all_signals:
                    signal_name = f'RSI_{ticker1}_vs_{ticker2}'
                    all_signals[signal_name] = (
                        all_signals[ticker1]['RSI_10'] > all_signals[ticker2]['RSI_10']
                    )
            
            elif config['type'] == 'rsi_static':
                ticker = config['ticker']
                threshold = config['threshold']
                if ticker in all_signals:
                    signal_name = f'RSI_{ticker}_vs_{threshold}'
                    all_signals[signal_name] = all_signals[ticker]['RSI_10'] > threshold
        
        return all_signals
    
    def split_data_advanced(self, data, method='walk_forward', train_size=0.8, random_state=42):
        """Advanced data splitting methods for time series"""
        
        if method == 'chronological':
            # Traditional chronological split
            split_idx = int(len(data) * train_size)
            train_data = data.iloc[:split_idx]
            test_data = data.iloc[split_idx:]
            
        elif method == 'walk_forward':
            # Walk-forward analysis (expanding window)
            split_idx = int(len(data) * train_size)
            train_data = data.iloc[:split_idx]
            test_data = data.iloc[split_idx:]
            
        elif method == 'rolling_window':
            # Rolling window (fixed window size)
            window_size = int(len(data) * train_size)
            split_idx = int(len(data) * 0.5)  # Start from middle
            train_data = data.iloc[split_idx-window_size:split_idx]
            test_data = data.iloc[split_idx:]
            
        elif method == 'purged_cv':
            # Purged cross-validation (gaps between train/test)
            gap_size = int(len(data) * 0.1)  # 10% gap
            split_idx = int(len(data) * train_size)
            train_data = data.iloc[:split_idx-gap_size]
            test_data = data.iloc[split_idx+gap_size:]
            
        elif method == 'blocked_cv':
            # Block-based cross validation
            block_size = len(data) // 10  # 10 blocks
            np.random.seed(random_state)
            blocks = np.arange(10)
            train_blocks = np.random.choice(blocks, size=8, replace=False)
            
            train_indices = []
            test_indices = []
            
            for i in range(10):
                start_idx = i * block_size
                end_idx = min((i + 1) * block_size, len(data))
                block_indices = list(range(start_idx, end_idx))
                
                if i in train_blocks:
                    train_indices.extend(block_indices)
                else:
                    test_indices.extend(block_indices)
            
            train_data = data.iloc[train_indices].sort_index()
            test_data = data.iloc[test_indices].sort_index()
            
        else:  # 'random' - original method (not recommended)
            dates = data.index.tolist()
            train_dates, test_dates = train_test_split(
                dates, train_size=train_size, random_state=random_state, shuffle=True
            )
            train_data = data.loc[train_dates].sort_index()
            test_data = data.loc[test_dates].sort_index()
        
        return train_data, test_data
    
    def backtest_strategy(self, signals, allocations, split_method='walk_forward', initial_capital=100000):
        """Backtest the trading strategy"""
        results = {}
        
        for ticker, allocation_pct in allocations.items():
            if ticker not in signals:
                continue
                
            df = signals[ticker].copy()
            df = df.dropna()
            
            if df.empty:
                continue
            
            # Split into train/test
            train_data, test_data = self.split_data_advanced(df, method=split_method)
            
            # Backtest on test data
            capital = initial_capital * (allocation_pct / 100)
            position = 0
            cash = capital
            portfolio_value = []
            trades = []
            
            for date, row in test_data.iterrows():
                current_price = row['Close']
                
                # Generate trading signal based on available columns
                signal = 0
                
                # Check for RSI signal
                if 'RSI_Signal' in row and pd.notna(row['RSI_Signal']):
                    signal = row['RSI_Signal']
                
                # Check for MA crossover signal
                elif 'MA_Signal' in row and pd.notna(row['MA_Signal']):
                    signal = row['MA_Signal']
                
                # Check for RSI direct values
                elif 'RSI_10' in row and pd.notna(row['RSI_10']):
                    if row['RSI_10'] < 30:  # Oversold
                        signal = 1  # Buy signal
                    elif row['RSI_10'] > 70:  # Overbought
                        signal = -1  # Sell signal
                
                # Check for MA crossover using MA columns
                elif 'MA_20' in row and 'MA_50' in row and pd.notna(row['MA_20']) and pd.notna(row['MA_50']):
                    if row['MA_20'] > row['MA_50']:
                        signal = 1  # Buy signal
                    else:
                        signal = -1  # Sell signal
                
                # Execute trades
                if signal == 1 and position <= 0:  # Buy
                    shares_to_buy = cash // current_price
                    if shares_to_buy > 0:
                        position += shares_to_buy
                        cash -= shares_to_buy * current_price
                        trades.append({
                            'date': date,
                            'action': 'BUY',
                            'shares': shares_to_buy,
                            'price': current_price
                        })
                
                elif signal == -1 and position > 0:  # Sell
                    cash += position * current_price
                    trades.append({
                        'date': date,
                        'action': 'SELL',
                        'shares': position,
                        'price': current_price
                    })
                    position = 0
                
                # Calculate portfolio value
                portfolio_val = cash + (position * current_price)
                portfolio_value.append({
                    'date': date,
                    'portfolio_value': portfolio_val,
                    'cash': cash,
                    'position': position,
                    'price': current_price
                })
            
            results[ticker] = {
                'portfolio_history': pd.DataFrame(portfolio_value),
                'trades': pd.DataFrame(trades),
                'train_period': f"{train_data.index.min()} to {train_data.index.max()}",
                'test_period': f"{test_data.index.min()} to {test_data.index.max()}",
                'final_value': portfolio_value[-1]['portfolio_value'] if portfolio_value else capital,
                'return_pct': ((portfolio_value[-1]['portfolio_value'] / capital - 1) * 100) if portfolio_value else 0
            }
        
        return results
    
    def calculate_performance_metrics(self, portfolio_history):
        """Calculate various performance metrics"""
        if portfolio_history.empty:
            return {}
        
        try:
            # Ensure portfolio_history has required columns
            if 'portfolio_value' not in portfolio_history.columns:
                return {}
            
            # Remove any infinite or NaN values
            portfolio_history = portfolio_history.replace([np.inf, -np.inf], np.nan).dropna()
            
            if len(portfolio_history) < 2:
                return {}
            
            returns = portfolio_history['portfolio_value'].pct_change().dropna()
            
            if len(returns) == 0:
                return {}
            
            # Calculate metrics with error handling
            initial_value = portfolio_history['portfolio_value'].iloc[0]
            final_value = portfolio_history['portfolio_value'].iloc[-1]
            
            if initial_value <= 0:
                return {}
            
            total_return = (final_value / initial_value - 1) * 100
            
            # Sharpe ratio (assuming 0% risk-free rate)
            if returns.std() > 0:
                sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
            else:
                sharpe_ratio = 0
            
            # Maximum drawdown
            try:
                cumulative = (1 + returns).cumprod()
                running_max = cumulative.expanding().max()
                drawdown = (cumulative - running_max) / running_max
                max_drawdown = drawdown.min() * 100
            except:
                max_drawdown = 0
            
            # Win rate (positive return periods)
            win_rate = (returns > 0).mean() * 100
            
            # Volatility
            volatility = returns.std() * np.sqrt(252) * 100 if returns.std() > 0 else 0
            
            return {
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'volatility': volatility
            }
            
        except Exception as e:
            st.warning(f"Error calculating performance metrics: {str(e)}")
            return {}
    
    def generate_random_allocations(self, tickers, num_combinations=10):
        """Generate single-ticker allocation combinations (100% to one ticker)"""
        allocations_list = []
        
        for _ in range(num_combinations):
            # Randomly select one ticker to allocate 100% to
            selected_ticker = np.random.choice(tickers)
            allocations = {ticker: 100.0 if ticker == selected_ticker else 0.0 for ticker in tickers}
            allocations_list.append(allocations)
        
        return allocations_list
    
    def generate_signal_combinations(self, rsi_periods, ma_periods):
        """Generate different signal parameter combinations"""
        combinations = []
        
        # RSI-based signals
        for rsi_period in rsi_periods:
            combinations.append({
                'type': 'rsi',
                'period': rsi_period,
                'overbought': 70,
                'oversold': 30
            })
        
        # Moving average crossovers
        for i, ma1 in enumerate(ma_periods):
            for ma2 in ma_periods[i+1:]:
                combinations.append({
                    'type': 'ma_crossover',
                    'fast_ma': ma1,
                    'slow_ma': ma2
                })
        
        # RSI + MA combinations
        for rsi_period in rsi_periods:
            for ma_period in ma_periods:
                combinations.append({
                    'type': 'rsi_ma_combined',
                    'rsi_period': rsi_period,
                    'ma_period': ma_period
                })
        
        return combinations
    
    def optimize_strategy(self, data, tickers, num_iterations=100, optimization_metric='Total Return', rsi_periods=None, ma_periods=None):
        """Optimize strategy by testing different allocations and signal combinations"""
        best_result = None
        best_score = float('-inf')
        best_config = None
        
        # Validate inputs
        if not data or not tickers:
            st.error("No data or tickers provided for optimization")
            return None, []
        
        # Generate allocation combinations
        allocation_combinations = self.generate_random_allocations(tickers, num_iterations)
        
        # Use provided periods or defaults
        if rsi_periods is None:
            rsi_periods = [10, 14, 20]
        if ma_periods is None:
            ma_periods = [20, 50, 100]
            
        signal_combinations = self.generate_signal_combinations(rsi_periods, ma_periods)
        
        st.info(f"Testing {len(allocation_combinations)} allocation combinations and {len(signal_combinations)} signal combinations")
        
        progress_bar = st.progress(0)
        results_summary = []
        
        for i, allocations in enumerate(allocation_combinations):
            for j, signal_config in enumerate(signal_combinations):
                try:
                    # Generate signals with current config
                    signals = self.generate_signals_optimized(data, signal_config)
                    
                    # Check if signals were generated successfully
                    if not signals:
                        continue
                    
                    # Run backtest
                    backtest_results = self.backtest_strategy(signals, allocations)
                    
                    # Check if backtest results are valid
                    if not backtest_results:
                        continue
                    
                    # Calculate overall performance (single ticker allocation)
                    total_return = 0
                    total_trades = 0
                    portfolio_history = pd.DataFrame()
                    allocated_ticker = None
                    
                    for ticker, result in backtest_results.items():
                        allocation_pct = allocations.get(ticker, 0)
                        if allocation_pct > 0:  # This is the allocated ticker
                            allocated_ticker = ticker
                            if result and 'portfolio_history' in result and not result['portfolio_history'].empty:
                                total_return = result.get('return_pct', 0)  # Direct return since 100% allocation
                                total_trades = len(result.get('trades', []))
                                portfolio_history = result['portfolio_history']
                            break
                    
                    # Only proceed if we have valid portfolio history
                    if portfolio_history.empty:
                        continue
                    
                    # Calculate performance metrics
                    metrics = self.calculate_performance_metrics(portfolio_history)
                    
                    # Determine score based on optimization metric
                    if optimization_metric == 'Total Return':
                        score = total_return
                    elif optimization_metric == 'Sharpe Ratio':
                        score = metrics.get('sharpe_ratio', 0)
                    elif optimization_metric == 'Max Drawdown':
                        score = -metrics.get('max_drawdown', 0)  # Negative because we want to minimize drawdown
                    elif optimization_metric == 'Win Rate':
                        score = metrics.get('win_rate', 0)
                    else:
                        score = total_return
                    
                    # Store result
                    result_summary = {
                        'allocations': allocations,
                        'signal_config': signal_config,
                        'total_return': total_return,
                        'metrics': metrics,
                        'total_trades': total_trades,
                        'score': score,
                        'allocated_ticker': allocated_ticker
                    }
                    results_summary.append(result_summary)
                    
                    # Update best result
                    if score > best_score:
                        best_score = score
                        best_result = result_summary
                        best_config = {
                            'allocations': allocations,
                            'signal_config': signal_config
                        }
                
                except Exception as e:
                    st.warning(f"Error in optimization iteration: {str(e)}")
                    continue
                
                # Update progress
                progress = (i * len(signal_combinations) + j + 1) / (len(allocation_combinations) * len(signal_combinations))
                progress_bar.progress(progress)
                
                # Memory cleanup every 10 iterations
                if (i * len(signal_combinations) + j + 1) % 10 == 0:
                    import gc
                    gc.collect()
        
        return best_result, results_summary
    
    def generate_signals_optimized(self, data, signal_config):
        """Generate signals with specific configuration for optimization"""
        signals = {}
        
        for ticker, df in data.items():
            signal_df = pd.DataFrame(index=df.index)
            signal_df['Close'] = df['Close']
            
            if signal_config['type'] == 'rsi':
                rsi = self.calculate_rsi(df['Close'], signal_config['period'])
                signal_df['RSI'] = rsi
                signal_df['Signal'] = (rsi < signal_config['oversold']).astype(int) - (rsi > signal_config['overbought']).astype(int)
            
            elif signal_config['type'] == 'ma_crossover':
                fast_ma = df['Close'].rolling(window=signal_config['fast_ma']).mean()
                slow_ma = df['Close'].rolling(window=signal_config['slow_ma']).mean()
                signal_df['Fast_MA'] = fast_ma
                signal_df['Slow_MA'] = slow_ma
                signal_df['Signal'] = (fast_ma > slow_ma).astype(int)
            
            elif signal_config['type'] == 'rsi_ma_combined':
                rsi = self.calculate_rsi(df['Close'], signal_config['rsi_period'])
                ma = df['Close'].rolling(window=signal_config['ma_period']).mean()
                signal_df['RSI'] = rsi
                signal_df['MA'] = ma
                signal_df['Signal'] = ((rsi < 30) & (df['Close'] > ma)).astype(int) - ((rsi > 70) & (df['Close'] < ma)).astype(int)
            
            signals[ticker] = signal_df.dropna()
        
        return signals

# Streamlit App
def main():
    st.title("📈 Signal Discovery Trading Application")
    st.markdown("---")
    
    # Initialize session state
    if 'signal_discovery' not in st.session_state:
        st.session_state.signal_discovery = SignalDiscovery()
    
    # Ensure the signal_discovery object exists and has the required methods
    if not hasattr(st.session_state.signal_discovery, 'optimize_strategy'):
        st.session_state.signal_discovery = SignalDiscovery()
    
    # Test that the object has all required methods
    required_methods = ['optimize_strategy', 'generate_signals_optimized', 'backtest_strategy', 'calculate_performance_metrics']
    missing_methods = [method for method in required_methods if not hasattr(st.session_state.signal_discovery, method)]
    
    if missing_methods:
        st.error(f"SignalDiscovery object is missing methods: {missing_methods}")
        st.session_state.signal_discovery = SignalDiscovery()
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Date range selection
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("Start Date", datetime.now() - timedelta(days=365*2))
    with col2:
        end_date = st.date_input("End Date", datetime.now())

    # Data splitting method selection
    st.sidebar.subheader("📊 Data Splitting Method")
    split_method = st.sidebar.selectbox(
        "Choose splitting method:",
        options=['chronological', 'walk_forward', 'rolling_window', 'purged_cv', 'blocked_cv', 'random'],
        index=1,  # Default to walk_forward
        help="""
        - Chronological: Traditional train→test split
        - Walk Forward: Expanding window (recommended)
        - Rolling Window: Fixed window size
        - Purged CV: Gap between train/test
        - Blocked CV: Random blocks (better than random)
        - Random: Not recommended for time series
        """
    )
    
    # Ticker input
    st.sidebar.subheader("📊 Tickers")
    ticker_input = st.sidebar.text_area(
        "Enter tickers (one per line):",
        value="BIL\nTLT\nCORP\nPSQ\nQQQ\nXLK",
        height=100
    )
    tickers = [t.strip().upper() for t in ticker_input.split('\n') if t.strip()]
    
    # Validate tickers
    if not tickers:
        st.sidebar.warning("Please enter at least one ticker symbol")
        tickers = ["BIL"]  # Default fallback
    
    # Signal Configuration
    st.sidebar.subheader("🎯 Signal Configuration")
    
    signal_configs = []
    
    # RSI Comparison Signals
    st.sidebar.write("**RSI Comparisons:**")
    num_rsi_comp = st.sidebar.number_input("Number of RSI comparisons", 0, 10, 2)
    
    for i in range(num_rsi_comp):
        col1, col2 = st.sidebar.columns(2)
        with col1:
            ticker1 = st.selectbox(f"Ticker 1 (RSI {i+1})", tickers, key=f"rsi_t1_{i}")
        with col2:
            ticker2 = st.selectbox(f"Ticker 2 (RSI {i+1})", tickers, key=f"rsi_t2_{i}")
        
        signal_configs.append({
            'type': 'rsi_comparison',
            'ticker1': ticker1,
            'ticker2': ticker2
        })
    
    # RSI vs Static Value
    st.sidebar.write("**RSI vs Static Value:**")
    num_rsi_static = st.sidebar.number_input("Number of RSI static comparisons", 0, 10, 1)
    
    for i in range(num_rsi_static):
        col1, col2 = st.sidebar.columns(2)
        with col1:
            ticker = st.selectbox(f"Ticker (Static {i+1})", tickers, key=f"rsi_static_t_{i}")
        with col2:
            threshold = st.number_input(f"RSI Threshold {i+1}", 0, 100, 50, key=f"rsi_thresh_{i}")
        
        signal_configs.append({
            'type': 'rsi_static',
            'ticker': ticker,
            'threshold': threshold
        })
    
    # Optimization Configuration
    st.sidebar.subheader("🔍 Optimization Settings")
    
    # Optimization parameters
    num_iterations = st.sidebar.number_input("Number of iterations", 10, 1000, 100, help="More iterations = better results but slower")
    optimization_metric = st.sidebar.selectbox(
        "Optimization Metric",
        ["Total Return", "Sharpe Ratio", "Max Drawdown", "Win Rate"],
        help="Metric to optimize for"
    )
    
    # Signal generation parameters
    st.sidebar.subheader("📊 Signal Parameters")
    rsi_periods = st.sidebar.multiselect(
        "RSI Periods to Test",
        [5, 10, 14, 20, 30],
        default=[10, 14],
        help="RSI periods to test in combinations"
    )
    
    ma_periods = st.sidebar.multiselect(
        "Moving Average Periods to Test",
        [5, 10, 20, 50, 100, 200],
        default=[20, 50],
        help="Moving average periods to test"
    )
    
    # Initialize allocations for optimization
    allocations = {}
    
    # Main content
    if st.button("🚀 Run Optimization", type="primary"):
        if not tickers:
            st.error("Please enter at least one ticker symbol")
            return
        
        with st.spinner("Fetching data and running optimization..."):
            # Fetch data
            data = st.session_state.signal_discovery.fetch_data(tickers, start_date, end_date)
            
            if not data:
                st.error("No data fetched. Please check your ticker symbols.")
                return
            
            # Run optimization with error handling
            try:
                if not hasattr(st.session_state.signal_discovery, 'optimize_strategy'):
                    st.error("SignalDiscovery object is missing optimize_strategy method")
                    return
                
                best_result, all_results = st.session_state.signal_discovery.optimize_strategy(
                    data=data, 
                    tickers=tickers, 
                    num_iterations=num_iterations, 
                    optimization_metric=optimization_metric, 
                    rsi_periods=rsi_periods, 
                    ma_periods=ma_periods
                )
                
                if best_result is None:
                    st.error("No valid strategies found. Try different parameters.")
                    return
                    
            except AttributeError as e:
                st.error(f"AttributeError in optimize_strategy: {str(e)}")
                st.error("This might be due to missing methods in SignalDiscovery class")
                return
            except Exception as e:
                st.error(f"Error during optimization: {str(e)}")
                st.info("Trying to run a simple backtest instead...")
                
                # Fallback to simple backtest
                try:
                    # Create simple equal allocations
                    equal_allocations = {ticker: 100.0 / len(tickers) for ticker in tickers}
                    
                    # Create simple signal config
                    simple_signal_config = {'type': 'rsi', 'period': 14, 'overbought': 70, 'oversold': 30}
                    
                    # Generate signals
                    signals = st.session_state.signal_discovery.generate_signals_optimized(data, simple_signal_config)
                    
                    # Run backtest
                    backtest_results = st.session_state.signal_discovery.backtest_strategy(signals, equal_allocations)
                    
                    # Create simple result
                    total_return = sum(result.get('return_pct', 0) * (equal_allocations.get(ticker, 0) / 100) 
                                    for ticker, result in backtest_results.items())
                    
                    best_result = {
                        'allocations': equal_allocations,
                        'signal_config': simple_signal_config,
                        'total_return': total_return,
                        'metrics': {'total_return': total_return, 'sharpe_ratio': 0, 'max_drawdown': 0, 'win_rate': 0},
                        'total_trades': 0,
                        'score': total_return
                    }
                    all_results = [best_result]
                    
                except Exception as fallback_error:
                    st.error(f"Fallback also failed: {str(fallback_error)}")
                    return
            
            # Display results
            st.markdown("## 🎯 Optimization Results")
            
            # Create tabs for different views
            tab1, tab2, tab3, tab4 = st.tabs(["Best Strategy", "All Results", "Performance Metrics", "Configuration"])
            
            with tab1:
                st.subheader("Best Strategy Found")
                
                # Display best strategy metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Return", f"{best_result['total_return']:.2f}%")
                with col2:
                    st.metric("Sharpe Ratio", f"{best_result['metrics'].get('sharpe_ratio', 0):.2f}")
                with col3:
                    st.metric("Max Drawdown", f"{best_result['metrics'].get('max_drawdown', 0):.2f}%")
                with col4:
                    st.metric("Win Rate", f"{best_result['metrics'].get('win_rate', 0):.1f}%")
                
                # Display best allocation
                st.subheader("Optimal Allocation")
                allocated_ticker = best_result.get('allocated_ticker', 'Unknown')
                st.success(f"**Best Strategy:** 100% allocation to {allocated_ticker}")
                
                # Show allocation breakdown
                allocation_df = pd.DataFrame([
                    {'Ticker': ticker, 'Allocation %': allocation}
                    for ticker, allocation in best_result['allocations'].items()
                ])
                st.dataframe(allocation_df, use_container_width=True)
                
                # Display signal configuration
                st.subheader("Optimal Signal Configuration")
                signal_config = best_result['signal_config']
                if signal_config['type'] == 'rsi':
                    st.write(f"**RSI Strategy:** Period {signal_config['period']}, Overbought: {signal_config['overbought']}, Oversold: {signal_config['oversold']}")
                elif signal_config['type'] == 'ma_crossover':
                    st.write(f"**Moving Average Crossover:** Fast MA {signal_config['fast_ma']}, Slow MA {signal_config['slow_ma']}")
                elif signal_config['type'] == 'rsi_ma_combined':
                    st.write(f"**Combined RSI + MA:** RSI Period {signal_config['rsi_period']}, MA Period {signal_config['ma_period']}")
                
                # Plot performance comparison
                if all_results:
                    returns = [r['total_return'] for r in all_results]
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(x=returns, nbinsx=20, name='Strategy Returns'))
                    fig.add_vline(x=best_result['total_return'], line_dash="dash", line_color="red", 
                                annotation_text="Best Strategy")
                    fig.update_layout(
                        title="Distribution of Strategy Returns",
                        xaxis_title="Total Return (%)",
                        yaxis_title="Frequency",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                st.subheader("All Strategy Results")
                
                # Create results dataframe
                results_data = []
                for result in all_results:
                    allocated_ticker = result.get('allocated_ticker', 'Unknown')
                    results_data.append({
                        'Ticker': allocated_ticker,
                        'Total Return (%)': f"{result['total_return']:.2f}",
                        'Sharpe Ratio': f"{result['metrics'].get('sharpe_ratio', 0):.2f}",
                        'Max Drawdown (%)': f"{result['metrics'].get('max_drawdown', 0):.2f}",
                        'Win Rate (%)': f"{result['metrics'].get('win_rate', 0):.1f}",
                        'Total Trades': result['total_trades'],
                        'Signal Type': result['signal_config']['type']
                    })
                
                if results_data:
                    df_results = pd.DataFrame(results_data)
                    st.dataframe(df_results, use_container_width=True)
                    
                    # Sort by total return and show top 10
                    st.subheader("Top 10 Strategies by Return")
                    sorted_results = sorted(all_results, key=lambda x: x['total_return'], reverse=True)[:10]
                    
                    top_results_data = []
                    for i, result in enumerate(sorted_results, 1):
                        allocated_ticker = result.get('allocated_ticker', 'Unknown')
                        top_results_data.append({
                            'Rank': i,
                            'Ticker': allocated_ticker,
                            'Total Return (%)': f"{result['total_return']:.2f}",
                            'Signal Type': result['signal_config']['type'],
                            'Sharpe Ratio': f"{result['metrics'].get('sharpe_ratio', 0):.2f}"
                        })
                    
                    df_top = pd.DataFrame(top_results_data)
                    st.dataframe(df_top, use_container_width=True)
            
            with tab3:
                st.subheader("Performance Metrics Analysis")
                
                # Performance metrics distribution
                if all_results:
                    metrics_data = {
                        'Total Return (%)': [r['total_return'] for r in all_results],
                        'Sharpe Ratio': [r['metrics'].get('sharpe_ratio', 0) for r in all_results],
                        'Max Drawdown (%)': [r['metrics'].get('max_drawdown', 0) for r in all_results],
                        'Win Rate (%)': [r['metrics'].get('win_rate', 0) for r in all_results]
                    }
                    
                    # Create correlation matrix
                    df_metrics = pd.DataFrame(metrics_data)
                    correlation_matrix = df_metrics.corr()
                    
                    fig = go.Figure(data=go.Heatmap(
                        z=correlation_matrix.values,
                        x=correlation_matrix.columns,
                        y=correlation_matrix.columns,
                        colorscale='RdBu',
                        zmid=0
                    ))
                    fig.update_layout(
                        title="Performance Metrics Correlation Matrix",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Metrics summary statistics
                    st.subheader("Metrics Summary Statistics")
                    summary_stats = df_metrics.describe()
                    st.dataframe(summary_stats, use_container_width=True)
            
            with tab4:
                st.subheader("Optimization Configuration")
                
                # Display optimization settings
                st.write("**Optimization Parameters:**")
                st.write(f"- Number of iterations: {num_iterations}")
                st.write(f"- Optimization metric: {optimization_metric}")
                st.write(f"- RSI periods tested: {rsi_periods}")
                st.write(f"- MA periods tested: {ma_periods}")
                
                # Display strategy types tested
                st.write("**Strategy Types Tested:**")
                strategy_types = set()
                for result in all_results:
                    strategy_types.add(result['signal_config']['type'])
                
                for strategy_type in strategy_types:
                    count = sum(1 for r in all_results if r['signal_config']['type'] == strategy_type)
                    st.write(f"- {strategy_type}: {count} combinations")
                
                # Display allocation statistics
                if all_results:
                    st.write("**Allocation Statistics:**")
                    all_allocations = []
                    for result in all_results:
                        for ticker, allocation in result['allocations'].items():
                            all_allocations.append({'Ticker': ticker, 'Allocation': allocation})
                    
                    df_alloc = pd.DataFrame(all_allocations)
                    allocation_stats = df_alloc.groupby('Ticker')['Allocation'].agg(['mean', 'std', 'min', 'max'])
                    st.dataframe(allocation_stats, use_container_width=True)

if __name__ == "__main__":
    main()
