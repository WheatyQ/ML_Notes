import numpy as np
import pandas as pd
from scipy import optimize

def estimate_dynamic_beta(stock_returns, index_returns, window=60):
    """
    Estimate dynamic betas using a rolling window approach.
    
    :param stock_returns: DataFrame with stock returns
    :param index_returns: DataFrame with index returns
    :param window: Size of the rolling window
    :return: DataFrame with dynamic beta values
    """
    dynamic_betas = pd.DataFrame(index=index_returns.index, columns=index_returns.columns)
    for index in index_returns.columns:
        dynamic_betas[index] = stock_returns.rolling(window=window).apply(lambda x: np.linalg.lstsq(index_returns[index].iloc[x.index].values.reshape(-1, 1), x.values, rcond=None)[0][0])
    return dynamic_betas

def optimize_robust_hedge_ratios(stock_values, futures_values, stock_returns, futures_returns, transaction_costs):
    """
    Optimize hedge ratios using a robust approach, accounting for transaction costs.
    
    :param stock_values: Current values of the stocks in the basket
    :param futures_values: Current values of the futures contracts
    :param stock_returns: Historical returns of the stocks
    :param futures_returns: Historical returns of the futures
    :param transaction_costs: Transaction costs for buying/selling futures contracts
    :return: Dictionary with optimal number of contracts for each future
    """
    num_stocks = len(stock_values)
    num_futures = len(futures_values)
    
    # Covariance matrix of stocks and futures
    cov_matrix = np.cov(np.concatenate([stock_returns, futures_returns], axis=1).T)
    
    # Robust quadratic programming setup
    P = cov_matrix[num_stocks:, num_stocks:].astype(float)  # Futures covariance
    q = -np.dot(cov_matrix[num_stocks:, :num_stocks], stock_values).astype(float)
    
    # Constraints setup including transaction costs
    G = np.zeros((num_futures + 1, num_futures))
    G[-1, :] = -1  # No short selling
    h = np.concatenate([transaction_costs * np.ones(num_futures), [0]])
    
    A = np.ones((1, num_futures))
    b = np.array([1.0])  # Sum of weights equal to 1
    
    # Solve quadratic program
    result = optimize.quadprog(P, q, G, h, A, b)
    
    optimal_contracts = {}
    for i, future in enumerate(futures_values.index):
        optimal_contracts[future] = result['x'][i] * stock_values.sum() / futures_values[future]
    
    return optimal_contracts

def rebalance_hedge(hedge_ratios, stock_values, futures_values, threshold):
    """
    Determine if rebalancing is needed based on changes in stock and futures values.
    
    :param hedge_ratios: Current hedge ratios
    :param stock_values: Current values of the stocks in the basket
    :param futures_values: Current values of the futures contracts
    :param threshold: Threshold for triggering rebalancing
    :return: Boolean indicating whether rebalancing is necessary
    """
    current_exposure = sum(hedge_ratios[future] * futures_values[future] for future in hedge_ratios)
    required_exposure = stock_values.sum()
    return abs(current_exposure - required_exposure) > threshold * required_exposure

# Example usage
if __name__ == "__main__":
    # Dummy data for stock and futures
    stock_returns = pd.DataFrame(np.random.randn(100, 5), columns=['Stock1', 'Stock2', 'Stock3', 'Stock4', 'Stock5'])
    index_returns = pd.DataFrame(np.random.randn(100, 3), columns=['SP500', 'NASDAQ', 'DOW'])
    
    stock_values = pd.Series([100000, 150000, 200000, 120000, 80000], index=stock_returns.columns)
    futures_values = pd.Series([10000, 12000, 11000], index=index_returns.columns)
    futures_returns = pd.DataFrame(np.random.randn(100, 3), columns=futures_values.index)
    transaction_costs = pd.Series([100, 120, 110], index=futures_values.index)
    
    # Calculate dynamic betas
    dynamic_betas = estimate_dynamic_beta(stock_returns, index_returns)
    print("Dynamic Betas:\n", dynamic_betas.tail())
    
    # Optimize hedge considering transaction costs
    hedge_ratios = optimize_robust_hedge_ratios(stock_values, futures_values, stock_returns, futures_returns, transaction_costs)
    print("Optimal Hedge Ratios:", hedge_ratios)
    
    # Check if rebalancing is needed
    should_rebalance = rebalance_hedge(hedge_ratios, stock_values, futures_values, threshold=0.05)
    print("Rebalancing Needed:", should_rebalance)
