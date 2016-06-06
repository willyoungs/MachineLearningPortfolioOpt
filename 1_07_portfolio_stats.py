"""Compute daily returns."""

import os
import pandas as pd
import matplotlib.pyplot as plt
import math

def symbol_to_path(symbol, base_dir="ml4t/data"):
    """Return CSV file path given ticker symbol."""
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))


def get_data(symbols, dates):
    """Read stock data (adjusted close) for given symbols from CSV files."""
    df = pd.DataFrame(index=dates)
    if 'SPY' not in symbols:  # add SPY for reference, if absent
        symbols.insert(0, 'SPY')

    for symbol in symbols:
        df_temp = pd.read_csv(symbol_to_path(symbol), index_col='Date',
                parse_dates=True, usecols=['Date', 'Adj Close'], na_values=['nan'])
        df_temp = df_temp.rename(columns={'Adj Close': symbol})
        df = df.join(df_temp)
        if symbol == 'SPY':  # drop dates SPY did not trade
            df = df.dropna(subset=["SPY"])

    return df


def plot_data(df, title="Stock prices", xlabel="Date", ylabel="Price"):
    """Plot stock prices with a custom title and meaningful axis labels."""
    ax = df.plot(title=title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()


def compute_daily_returns(df):
    """Compute and return the daily return values."""
    # TODO: Your code here
    # Note: Returned DataFrame must have the same number of rows
    # print df
    df = df/(df.shift(1)) - 1
    #Advertised solution^^^^
    # df = df.pct_change()
    #My solution^^^^
    #
    df.ix[0, :] = 0
    #0th column set to NaNs in both cases^^^
    return df
def cum_portfolio_return(df,symbols,weights,budget):
    current_value = 0
    original_value = 0
    index = 0
    # print df.ix[0]
    while index < len(df.ix[0]):
        original_value += weights[index]*df.ix[0][index]
        index += 1
    index = 0
    while index < len(df.ix[0]):
        current_value += weights[index]*df.ix[-1][index]
        index += 1
    # print "original_value: " + str(original_value)+"\n"
    # print "current_value: " + str(current_value)+"\n"
    return (current_value - original_value)/original_value
"""Applies weights to returns of """
def apply_weights(df, weights):
    df = df.multiply(weights)
    return df
"""High level method retrieves the stats requested"""
def get_stats(df,symbols,weights,budget):
    cumReturn = cum_portfolio_return(df,symbols,weights,budget)
    ####
    returns_df = compute_daily_returns(df)
    weighted_daily_returns_df = apply_weights(returns_df,weights)
    port_daily_return_df = port_daily_return(weighted_daily_returns_df)
    ####
    avgDailyReturn = avg_port_daily_return(port_daily_return_df)
    ####
    stdDev = port_std_dev_daily_returns(port_daily_return_df)
    ####
    sharpeRatio =  math.sqrt(252.0)*(avg_port_daily_return(port_daily_return_df)/port_std_dev_daily_returns(port_daily_return_df))
    return cumReturn,avgDailyReturn,stdDev,sharpeRatio
"""Calculates daily return of the portfolio"""
def port_daily_return(df):
    df = df.sum(1)
    return df
def avg_port_daily_return(df):
    df = df.mean()
    return df
def port_std_dev_daily_returns(df):
    return df.std()
def test_run():
    # Read data
    #Assert that the weights array is the same size as columns in dataFrame
    dates = pd.date_range('2012-07-01', '2012-07-31')  # one month only
    symbols = ['SPY','XOM']
    weights = [.5,.5]
    budget  = 10000
    df = get_data(symbols, dates)
    cumumlative_return,avg_daily_return,std_dev_daily_returns,sharpe_ratio = get_stats(df,symbols,weights,budget)
    print "Cumulative Return:"
    print cumumlative_return
    print "Avg Daily Return:"
    print avg_daily_return
    print "Standard Deviation of Daily Return:"
    print std_dev_daily_returns
    print "Sharpe Ratio or Risk adjusted return (Risk-free Rate of zero):"
    print sharpe_ratio
if __name__ == "__main__":
    test_run()