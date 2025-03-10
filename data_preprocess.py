# all the imports
import os
import yfinance as yf
import yahoo_fin.stock_info as si
import pandas as pd
from datetime import datetime
import numpy as np
from concurrent.futures import process
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import warnings

# get the price data from yfinance
def get_price_data(ticker, start_date, end_date):
    """Get raw OHLCV data and format it into the desired DataFrame structure."""
    df = yf.download(ticker, start=start_date, end=end_date)
    # Reset index to turn Date into a column
    df = df.reset_index()
    # Flatten the MultiIndex columns (assuming columns have three levels)
    # Extract the metric names (second level of MultiIndex)
    new_columns = []
    for col in df.columns:
        if col == 'Date':
            new_columns.append('Date')
        else:
            # Get the second level of the MultiIndex (e.g., 'Close', 'High')
            new_columns.append(col[1])
    df.columns = new_columns
    # Add ticker column
    df['ticker'] = ticker
    df.columns.values[0] = "Date"
    df.columns.values[1] = "Close"
    df.columns.values[2] = "High"
    df.columns.values[3] = "Low"
    df.columns.values[4] = "Open"
    df.columns.values[5] = "Volume"

    # Reorder columns to the desired order
    return df

# add the trend indicators
def add_trend_indicators(df):
    """Add SMA, EMA, MACD (grouped by ticker)"""
    # Ensure we group by ticker if multiple stocks exist
    grouped = df.groupby('ticker', group_keys=False)

    # Calculate indicators per ticker
    def calculate_indicators(group):
        # Moving Averages
        group['SMA_20'] = group['Close'].rolling(window=20).mean()
        group['EMA_20'] = group['Close'].ewm(span=20, adjust=False).mean()

        # MACD
        exp12 = group['Close'].ewm(span=12, adjust=False).mean()
        exp26 = group['Close'].ewm(span=26, adjust=False).mean()
        group['MACD'] = exp12 - exp26
        group['Signal_Line'] = group['MACD'].ewm(span=9, adjust=False).mean()
        return group

    return grouped.apply(calculate_indicators)

# add the momentum indicators
def add_momentum_indicators(df):
    """Add RSI and Stochastic Oscillator"""
    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Stochastic
    low14 = df['Low'].rolling(14).min()
    high14 = df['High'].rolling(14).max()
    df['%K'] = 100 * ((df['Close'] - low14) / (high14 - low14))
    df['%D'] = df['%K'].rolling(3).mean()
    return df

# add the volatility indicators
def add_volatility_indicators(df):
    """Add Bollinger Bands and ATR"""
    # Bollinger Bands
    df['Middle_Band'] = df['Close'].rolling(20).mean()
    df['Upper_Band'] = df['Middle_Band'] + 2*df['Close'].rolling(20).std()
    df['Lower_Band'] = df['Middle_Band'] - 2*df['Close'].rolling(20).std()

    # ATR
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()
    return df

# add the volume indicators
def add_volume_indicators(df):
    """Add CMF and Volume ROC"""
    # Chaikin Money Flow
    mfm = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
    mfm = mfm.replace([np.inf, -np.inf], 0)
    mfv = mfm * df['Volume']
    df['CMF'] = mfv.rolling(20).sum() / df['Volume'].rolling(20).sum()

    # Volume ROC
    df['Volume_ROC'] = df['Volume'].pct_change(1) * 100
    return df

# add the risk metrics
def add_risk_metrics(df, risk_free_rate=0.0135):
    """Add returns and risk metrics"""
    # Returns
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Rolling_Vol_21D'] = df['Log_Returns'].rolling(21).std() * np.sqrt(252)

    # Drawdown
    cumulative_returns = (1 + df['Log_Returns']).cumprod()
    peak = cumulative_returns.expanding().max()
    df['Drawdown'] = (peak - cumulative_returns) / peak

    # Sharpe Ratio
    df['Sharpe'] = (df['Log_Returns'].mean() - risk_free_rate/252) / df['Log_Returns'].std()
    return df

# get the corporate actions
def get_corporate_actions(ticker, start_date, end_date):
    """Get dividends and splits"""
    yf_ticker = yf.Ticker(ticker)
    return {
        'Dividends': yf_ticker.dividends.loc[start_date:end_date].sum(),
        'Splits': yf_ticker.splits.loc[start_date:end_date].count()
    }

# build the full dataset
warnings.simplefilter(action='ignore', category=FutureWarning)

start_date = "2020-01-01"
end_date = datetime.today().strftime('%Y-%m-%d')

dow_list = si.tickers_dow()
def build_full_dataset(tickers, start_date, end_date):
    final_data = []

    for ticker in tickers:
        try:
            # Core data
            df = get_price_data(ticker, start_date, end_date)

            # Add features
            df = add_trend_indicators(df)
            df = add_momentum_indicators(df)
            df = add_volatility_indicators(df)
            df = add_volume_indicators(df)
            df = add_risk_metrics(df)

            # Add corporate actions
            actions = get_corporate_actions(ticker, start_date, end_date)
            df['Dividends'] = actions['Dividends']
            df['Splits'] = actions['Splits']

            final_data.append(df)

        except Exception as e:
            print(f"Skipped {ticker}: {str(e)}")

    return pd.concat(final_data, ignore_index=True).dropna()

full_dataset = build_full_dataset(dow_list[:3], "2020-01-01", "2023-01-01")  # Test with first 3 tickers
print("\nFinal Dataset Columns:", full_dataset.columns.tolist())
print("\nSample Data:\n", full_dataset.head())

processed_df = build_full_dataset(dow_list, start_date, end_date)

# scale the features via min-max scaling optionally (I used this to understand how it works and learning)
def scale_features_min_max(df):
    """Normalize feature columns using Min-Max scaling"""
    # Columns to exclude from scaling
    non_feature_cols = ['Date', 'ticker']
    feature_cols = [col for col in df.columns if col not in non_feature_cols]

    # Initialize scaler and fit to training data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_values = scaler.fit_transform(df[feature_cols])

    # Rebuild DataFrame
    scaled_df = pd.DataFrame(scaled_values, columns=feature_cols)
    return pd.concat([df[non_feature_cols], scaled_df], axis=1), scaler

# scale the features via z-score scaling (I used this for the final dataset)
def scale_features_z_score(df, scaler=None):
    """Z-score normalization while preserving market data"""
    # Columns to preserve unchanged
    preserved = [
        'Date', 'ticker', 'Close', 'High', 'Low', 'Open', 'Volume',
        'Log_Returns', 'Dividends', 'Splits', 
        'SMA_20', 'EMA_20', 'Middle_Band', 'Upper_Band', 'Lower_Band',  # Price-based indicators
        'ATR', 'Rolling_Vol_21D', 'Drawdown', 'Sharpe'  # Risk metrics
    ]
    
    # Columns to scale (technical indicators)
    scale_cols = [c for c in df.columns if c not in preserved]
    
    # Separate data
    metadata = df[preserved].copy()
    features = df[scale_cols]
    
    # Initialize/reuse scaler
    if scaler is None:
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
    else:
        scaled_features = scaler.transform(features)
    
    # Rebuild DataFrame
    scaled_df = pd.DataFrame(scaled_features, 
                            columns=scale_cols, 
                            index=df.index)
    
    return pd.concat([metadata, scaled_df], axis=1), scaler

# Modified time series split to maintain temporal structure
def time_series_split(df, train_ratio=0.7, test_ratio=0.2):
    """Split data sequentially while maintaining time order"""
    df = df.sort_values('Date').reset_index(drop=True)
    n = len(df)
    
    train_end = int(n * train_ratio)
    test_end = train_end + int(n * test_ratio)
    
    return (
        df.iloc[:train_end],  # Train
        df.iloc[train_end:test_end],  # Test
        df.iloc[test_end:]  # Black swan (remaining data)
    )

# 1. Split first to prevent leakage
train_raw, test_raw, black_swan_raw = time_series_split(processed_df)

# 2. Scale using training data statistics
train_scaled, scaler = scale_features_z_score(train_raw)
test_scaled, _ = scale_features_z_score(test_raw, scaler)
black_swan_scaled, _ = scale_features_z_score(black_swan_raw, scaler)

# 3. Verify critical columns remain unscaled
print(train_scaled[['Date', 'ticker', 'Close', 'Log_Returns']].head())

# Final formatted datasets
print(f"Training set shape: {train_scaled.shape}")
print(f"Test set shape: {test_scaled.shape}")
print(f"Black swan reserve shape: {black_swan_scaled.shape}")

# Verify columns
print("\nSample scaled training data:")
print(train_scaled[['Date', 'ticker', 'Close', 'SMA_20']].head())


# Create directory if needed
os.makedirs('data', exist_ok=True)

# Save scaled datasets
train_scaled.to_csv('data/train_scaled.csv', index=False)
test_scaled.to_csv('data/test_scaled.csv', index=False)
black_swan_scaled.to_csv('data/black_swan_scaled.csv', index=False)
print("Saved files:")
print(os.listdir('data'))
