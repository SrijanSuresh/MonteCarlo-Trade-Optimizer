#!/usr/bin/env python3
"""
Monte Carlo Trade Optimizer
This module implements a Monte Carlo simulation for financial trading optimization
using Geometric Brownian Motion (GBM) with jump processes.
"""

# Standard library imports
from datetime import datetime

# Third-party imports
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import seaborn as sns

# Constants
RISK_FREE_RATE = 0.03
NUM_SIMULATIONS = 10000
JUMP_PROBABILITY = 0.05
EPSILON = 1e-9  # Small value for numerical stability

def load_data():
    """Load and return the training, testing, and black swan datasets."""
    train_scaled = pd.read_csv('data/train_scaled.csv')
    test_scaled = pd.read_csv('data/test_scaled.csv')
    black_swan_scaled = pd.read_csv('data/black_swan_scaled.csv')
    return train_scaled, test_scaled, black_swan_scaled

def simulateGBM(mu, sigma, S0, num_days, dt=1/252, apply_jump=False, jump_size=-0.22314):
    """
    Simulate Geometric Brownian Motion with optional jump process.
    
    Args:
        mu (float): Drift parameter
        sigma (float): Volatility parameter
        S0 (float): Initial price
        num_days (int): Number of days to simulate
        dt (float): Time step size (default: 1/252 for daily)
        apply_jump (bool): Whether to apply a jump
        jump_size (float): Size of the jump (default: ln(0.8) ≈ -0.22314 for 20% drop)
    
    Returns:
        numpy.ndarray: Array of simulated prices
    """
    daily_drift = (mu - 0.5 * sigma**2) * dt
    daily_vol = sigma * np.sqrt(dt)

    log_returns = np.random.normal(daily_drift, daily_vol, num_days)

    if apply_jump:
        jump_day = np.random.randint(0, num_days)
        log_returns[jump_day] = np.log(0.8)

    price_path = S0 * np.exp(np.cumsum(log_returns))
    return np.concatenate([[S0], price_path])

def calculate_metrics(price_path):
    """
    Calculate risk metrics for a single price path.
    
    Args:
        price_path (numpy.ndarray): Array of prices
    
    Returns:
        dict: Dictionary containing Sharpe ratio, Sortino ratio, and max drawdown
    """
    prices = np.maximum(price_path, EPSILON)
    returns = prices[1:] / prices[:-1] - 1

    with np.errstate(divide='ignore', invalid='ignore'):
        sharpe = np.nanmean(returns) / np.nanstd(returns) * np.sqrt(252)
        
        downside_returns = returns[returns < 0]
        sortino = (np.nanmean(returns) / np.nanstd(downside_returns)) * np.sqrt(252) \
                   if len(downside_returns) > 0 else np.nan
        
        peaks = np.maximum.accumulate(prices)
        drawdowns = (peaks - prices) / (peaks + EPSILON)
        max_drawdown = np.nanmax(drawdowns)

    return {
        'Sharpe': sharpe if not np.isnan(sharpe) else 0,
        'Sortino': sortino if not np.isnan(sortino) else 0,
        'Max_Drawdown': max_drawdown
    }

def vectorized_metrics(paths, risk_free_rate=RISK_FREE_RATE):
    """
    Calculate metrics for multiple paths in a vectorized manner.
    
    Args:
        paths (list): List of price paths
        risk_free_rate (float): Annualized risk-free rate
    
    Returns:
        pandas.DataFrame: DataFrame containing risk metrics for all paths
    """
    prices = np.vstack(paths)
    returns = prices[:,1:] / prices[:,:-1] - 1
    daily_rf = risk_free_rate / 252

    excess_returns = returns - daily_rf
    sharpe = np.nanmean(excess_returns, axis=1) / np.nanstd(returns, axis=1) * np.sqrt(252)

    downside_returns = np.where(returns < daily_rf, returns - daily_rf, 0)
    downside_std = np.sqrt(np.nanmean(downside_returns**2, axis=1))
    sortino = np.nanmean(excess_returns, axis=1) / downside_std * np.sqrt(252)

    peaks = np.maximum.accumulate(prices, axis=1)
    drawdowns = (peaks - prices) / (peaks + EPSILON)
    max_dd = np.nanmax(drawdowns, axis=1)

    return pd.DataFrame({
        'Sharpe': np.clip(sharpe, -5, 5),
        'Sortino': np.clip(sortino, -5, 5),
        'Max_Drawdown': max_dd
    })

def main():
    """Main execution function."""
    # Load data
    training_data, test_data, _ = load_data()
    
    # Prepare training data
    training_data['Rolling_Vol_21D'] = np.abs(training_data['Rolling_Vol_21D'])
    training_data['Arithmetic_Returns'] = training_data['Close'].pct_change()
    
    # Calculate parameters
    mu = np.mean(training_data['Log_Returns'].dropna()) + 0.5*np.var(training_data['Log_Returns'].dropna())
    mu *= 252  # Annualize
    
    rolling_vols = np.clip(training_data['Rolling_Vol_21D'].dropna().values, 0.05, 1.5)
    S0 = training_data['Close'].iloc[-1]
    num_days = len(test_data)
    
    # Generate paths
    paths = []
    jump_flags = []
    
    for i in range(NUM_SIMULATIONS):
        sigma = np.clip(np.random.choice(rolling_vols), 0.05, 2.0)
        apply_jump = np.random.rand() < JUMP_PROBABILITY

        try:
            path = simulateGBM(mu, sigma, S0, num_days, apply_jump=apply_jump)
            if np.any(path <= 0):
                raise ValueError("Negative prices in path")
            paths.append(path)
            jump_flags.append(apply_jump)
        except Exception as e:
            print(f"Error in path {i}: {str(e)}")
            continue

    # Calculate metrics
    metrics_df = vectorized_metrics(paths)
    metrics_df['Jump'] = jump_flags

    # Filter and visualize results
    valid_mask = (metrics_df['Max_Drawdown'] < 1.0) & (np.abs(metrics_df['Sharpe']) < 5)
    print(f"Valid paths ratio: {valid_mask.mean():.2%}")

    plt.figure(figsize=(12, 6))
    sns.violinplot(x='Jump', y='Sharpe', data=metrics_df[valid_mask])
    plt.title("Sharpe Ratio Distribution (Valid Paths Only)")
    plt.show()

if __name__ == "__main__":
    main()
