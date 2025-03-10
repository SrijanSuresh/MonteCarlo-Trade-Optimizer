#!/usr/bin/env python3
"""
Data Preprocessing Module for Stock Market Analysis
This module handles the collection, preprocessing, and feature engineering of stock market data
using various technical indicators and risk metrics.
"""

# Standard library imports
import os
from datetime import datetime
from concurrent.futures import process
import warnings

# Third-party imports
import numpy as np
import pandas as pd
import yfinance as yf
import yahoo_fin.stock_info as si
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class StockDataCollector:
    """Class to handle data collection from financial APIs."""
    
    @staticmethod
    def get_price_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Get raw OHLCV data and format it into the desired DataFrame structure.
        
        Args:
            ticker (str): Stock ticker symbol
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            
        Returns:
            pd.DataFrame: Formatted OHLCV data
        """
        df = yf.download(ticker, start=start_date, end=end_date)
        df = df.reset_index()
        
        # Format column names
        new_columns = []
        for col in df.columns:
            if col == 'Date':
                new_columns.append('Date')
            else:
                new_columns.append(col[1])
        df.columns = new_columns
        
        # Standardize column names
        column_mapping = {
            'Date': 'Date',
            'Adj Close': 'Close',
            'High': 'High',
            'Low': 'Low',
            'Open': 'Open',
            'Volume': 'Volume'
        }
        df = df.rename(columns=column_mapping)
        df['ticker'] = ticker
        
        return df
    
    @staticmethod
    def get_corporate_actions(ticker: str, start_date: str, end_date: str) -> dict:
        """
        Get dividends and splits information for a stock.
        
        Args:
            ticker (str): Stock ticker symbol
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            
        Returns:
            dict: Dictionary containing dividends and splits information
        """
        yf_ticker = yf.Ticker(ticker)
        return {
            'Dividends': yf_ticker.dividends.loc[start_date:end_date].sum(),
            'Splits': yf_ticker.splits.loc[start_date:end_date].count()
        }

class TechnicalIndicators:
    """Class containing methods for calculating technical indicators."""
    
    @staticmethod
    def add_trend_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add trend indicators (SMA, EMA, MACD) grouped by ticker.
        
        Args:
            df (pd.DataFrame): Input DataFrame with OHLCV data
            
        Returns:
            pd.DataFrame: DataFrame with added trend indicators
        """
        grouped = df.groupby('ticker', group_keys=False)
        
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
    
    @staticmethod
    def add_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add momentum indicators (RSI and Stochastic Oscillator).
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with added momentum indicators
        """
        # RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Stochastic Oscillator
        low14 = df['Low'].rolling(14).min()
        high14 = df['High'].rolling(14).max()
        df['%K'] = 100 * ((df['Close'] - low14) / (high14 - low14))
        df['%D'] = df['%K'].rolling(3).mean()
        
        return df
    
    @staticmethod
    def add_volatility_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volatility indicators (Bollinger Bands and ATR).
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with added volatility indicators
        """
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
    
    @staticmethod
    def add_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volume indicators (CMF and Volume ROC).
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with added volume indicators
        """
        # Chaikin Money Flow
        mfm = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
        mfm = mfm.replace([np.inf, -np.inf], 0)
        mfv = mfm * df['Volume']
        df['CMF'] = mfv.rolling(20).sum() / df['Volume'].rolling(20).sum()
        
        # Volume Rate of Change
        df['Volume_ROC'] = df['Volume'].pct_change(1) * 100
        
        return df
    
    @staticmethod
    def add_risk_metrics(df: pd.DataFrame, risk_free_rate: float = 0.0135) -> pd.DataFrame:
        """
        Add returns and risk metrics.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            risk_free_rate (float): Annual risk-free rate
            
        Returns:
            pd.DataFrame: DataFrame with added risk metrics
        """
        # Returns and Volatility
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Rolling_Vol_21D'] = df['Log_Returns'].rolling(21).std() * np.sqrt(252)
        
        # Drawdown
        cumulative_returns = (1 + df['Log_Returns']).cumprod()
        peak = cumulative_returns.expanding().max()
        df['Drawdown'] = (peak - cumulative_returns) / peak
        
        # Sharpe Ratio
        df['Sharpe'] = (df['Log_Returns'].mean() - risk_free_rate/252) / df['Log_Returns'].std()
        
        return df

class DataProcessor:
    """Class to handle data processing and scaling operations."""
    
    @staticmethod
    def scale_features_min_max(df: pd.DataFrame) -> tuple:
        """
        Normalize feature columns using Min-Max scaling.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            tuple: (scaled DataFrame, fitted scaler)
        """
        non_feature_cols = ['Date', 'ticker']
        feature_cols = [col for col in df.columns if col not in non_feature_cols]
        
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_values = scaler.fit_transform(df[feature_cols])
        
        scaled_df = pd.DataFrame(scaled_values, columns=feature_cols)
        return pd.concat([df[non_feature_cols], scaled_df], axis=1), scaler
    
    @staticmethod
    def scale_features_z_score(df: pd.DataFrame, scaler: StandardScaler = None) -> tuple:
        """
        Z-score normalization while preserving market data.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            scaler (StandardScaler, optional): Pre-fitted scaler
            
        Returns:
            tuple: (scaled DataFrame, fitted scaler)
        """
        preserved = [
            'Date', 'ticker', 'Close', 'High', 'Low', 'Open', 'Volume',
            'Log_Returns', 'Dividends', 'Splits', 
            'SMA_20', 'EMA_20', 'Middle_Band', 'Upper_Band', 'Lower_Band',
            'ATR', 'Rolling_Vol_21D', 'Drawdown', 'Sharpe'
        ]
        
        scale_cols = [c for c in df.columns if c not in preserved]
        metadata = df[preserved].copy()
        features = df[scale_cols]
        
        if scaler is None:
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features)
        else:
            scaled_features = scaler.transform(features)
        
        scaled_df = pd.DataFrame(scaled_features, columns=scale_cols, index=df.index)
        return pd.concat([metadata, scaled_df], axis=1), scaler
    
    @staticmethod
    def time_series_split(df: pd.DataFrame, train_ratio: float = 0.7, test_ratio: float = 0.2) -> tuple:
        """
        Split data sequentially while maintaining time order.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            train_ratio (float): Ratio of data for training
            test_ratio (float): Ratio of data for testing
            
        Returns:
            tuple: (train, test, black_swan) DataFrames
        """
        df = df.sort_values('Date').reset_index(drop=True)
        n = len(df)
        
        train_end = int(n * train_ratio)
        test_end = train_end + int(n * test_ratio)
        
        return (
            df.iloc[:train_end],
            df.iloc[train_end:test_end],
            df.iloc[test_end:]
        )

def main():
    """Main execution function."""
    warnings.simplefilter(action='ignore', category=FutureWarning)
    
    # Configuration
    start_date = "2020-01-01"
    end_date = datetime.today().strftime('%Y-%m-%d')
    dow_list = si.tickers_dow()
    
    def build_full_dataset(tickers, start_date, end_date):
        collector = StockDataCollector()
        indicators = TechnicalIndicators()
        final_data = []
        
        for ticker in tickers:
            try:
                # Collect data
                df = collector.get_price_data(ticker, start_date, end_date)
                
                # Add technical indicators
                df = indicators.add_trend_indicators(df)
                df = indicators.add_momentum_indicators(df)
                df = indicators.add_volatility_indicators(df)
                df = indicators.add_volume_indicators(df)
                df = indicators.add_risk_metrics(df)
                
                # Add corporate actions
                actions = collector.get_corporate_actions(ticker, start_date, end_date)
                df['Dividends'] = actions['Dividends']
                df['Splits'] = actions['Splits']
                
                final_data.append(df)
            
            except Exception as e:
                print(f"Skipped {ticker}: {str(e)}")
        
        return pd.concat(final_data, ignore_index=True).dropna()
    
    # Process data
    processor = DataProcessor()
    processed_df = build_full_dataset(dow_list, start_date, end_date)
    
    # Split and scale data
    train_raw, test_raw, black_swan_raw = processor.time_series_split(processed_df)
    train_scaled, scaler = processor.scale_features_z_score(train_raw)
    test_scaled, _ = processor.scale_features_z_score(test_raw, scaler)
    black_swan_scaled, _ = processor.scale_features_z_score(black_swan_raw, scaler)
    
    # Save processed data
    os.makedirs('data', exist_ok=True)
    train_scaled.to_csv('data/train_scaled.csv', index=False)
    test_scaled.to_csv('data/test_scaled.csv', index=False)
    black_swan_scaled.to_csv('data/black_swan_scaled.csv', index=False)
    
    print("Data processing completed successfully.")
    print(f"Training set shape: {train_scaled.shape}")
    print(f"Test set shape: {test_scaled.shape}")
    print(f"Black swan reserve shape: {black_swan_scaled.shape}")

if __name__ == "__main__":
    main()
