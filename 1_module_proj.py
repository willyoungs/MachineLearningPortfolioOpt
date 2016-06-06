import os
import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
import scipy.optimize as spo
def symbol_to_path(symbol, base_dir="ml4t/data"):
    """Return CSV file path given ticker symbol."""
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))

class Portfolio(object):
    """Class Representation of a portfolio"""
    
    def __init__(self, symbols,weights,dates):
        self.weights = weights
        self.symbols = symbols
        self.dates = dates
        self.data = self.get_data(dates)
    
    def __str__(self):
        return "Symbols: " + str(self.symbols)+"\nWeights: " + str(self.weights)
    
    def __repr__(self):
        return "Symbols: " + str(self.symbols)+"\nWeights: " + str(self.weights)
    
    def get_data(self, dates):
        """Read stock data (adjusted close) for given symbols from CSV files."""
        removeSPY = False
        df = pd.DataFrame(index=dates)
        if 'SPY' not in self.symbols:  # add SPY for reference, if absent
            self.symbols.insert(0, 'SPY')
            removeSPY = True
        for symbol in self.symbols:
            df_temp = pd.read_csv(symbol_to_path(symbol), index_col='Date',
                parse_dates=True, usecols=['Date', 'Adj Close'], na_values=['nan'])
            df_temp = df_temp.rename(columns={'Adj Close': symbol})
            df = df.join(df_temp)
            if symbol == 'SPY':  # drop dates SPY did not trade
                df = df.dropna(subset=["SPY"])
        if removeSPY:
            del df['SPY']
        return df
    
    def plot_data(self,df, title="Stock prices", xlabel="Date", ylabel="Price"):
        """Plot stock prices with a custom title and meaningful axis labels."""
        # print df
        ax = df.plot(title=title, fontsize=12)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        # plt.show()


    """Compute and return the daily return values."""
    def compute_daily_returns(self,df):
        df = df/(df.shift(1)) - 1
        #Advertised solution^^^^
        # df = df.pct_change()
        #My solution^^^^
        #
        df.ix[0, :] = 0
        #0th column set to NaNs in both cases^^^
        return df
    
    """Applies weights to df collumns"""
    def apply_weights(self,df, weights):
        df = df.multiply(weights)
        return df
    
    ######
    ###### PORTFOLIO STATS
    def cum_portfolio_return(self):
        df = self.data
        weights = self.weights
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
    def avg_daily_return_port(self):
        df = self.data
        weights = self.weights
        df = self.compute_daily_returns(df)
        df = self.apply_weights(df,self.weights)
        df = df.sum(1).mean()
        return df
    def port_std_dev_daily_returns(self):
        df = self.data
        df = self.compute_daily_returns(df)
        df = self.apply_weights(df,self.weights)
        df = df.sum(1)
        return df.std()
    def sharpe_ratio(self):
        result = math.sqrt(252.0) * ((self.avg_daily_return_port() - 0)/self.port_std_dev_daily_returns())
        return result
    def negative_sharpe_ratio(self):
        return self.sharpe_ratio() * -1
    """Computes portfolio asset's percentage deviation from the Start Date"""
    def compute_port_returns_from_start(self):
        df = self.compute_returns_from_start()
        df = self.apply_weights(df,self.weights)
        df = df.sum(1)
        return df
    ######
    ###### PER-ASSET STATS 
    """Generates cum return for each asset in the portfolio"""
    def cum_return_by_asset(self):
        df = self.data
        return (df.ix[-1] - df.ix[0])/df.ix[0]
    """Generates series of daily returns for each asset in the portfolio"""
    def avg_return_by_asset(self):
        df = self.data
        df = self.compute_daily_returns(df)
        return df.mean()
    def std_dev_by_asset(self):
        df = self.compute_daily_returns(self.data)
        return df.std()

    """Computes individual asset's percentage deviation from the Start Date"""
    def compute_returns_from_start(self):
        df = self.data
        df = df/df.ix[1,:]
        df.ix[0,:] = 1
        return df

    ##### Optimization  functions
    def change_weight_return_sharpe_ratio(self, weights):
        self.weights = weights
        return self.negative_sharpe_ratio()


"Function to be Minimzed: Here the Sharpe Ratio"
def error_poly(C,port):
    return port.change_weight_return_sharpe_ratio(C)



def fit_poly(data, err_func, degree = 3):
    Cguess = np.poly1d(np.ones(degree + 1, dtype = np.float32))
    x = np.linspace(-5,5,21)
    plt.plot(x,np.polyval(Cguess,x),'m--', linewidth = 2.0, label = "Initial guess")
    cons = {'type':'eq', 'fun': con}
    bnds = []
    for elem in Cguess:
        bnds.append((0.0,1.0))
    result = spo.minimize(err_func,Cguess, args=(data,), constraints=cons, method='SLSQP', bounds = bnds, options = {'disp': False})
    return np.poly1d(result.x)
def con(t):
    return sum(t) - 1 
def test_run():
    # Read data
    #Assert that the weights array is the same size as columns in dataFrame
    
    dates = pd.date_range('2010-01-01', '2011-01-01')  
    symbols = ['GOOG','AAPL','GLD','XOM'] 
    weights = [.25,.25,.25,.25]
    p1 = Portfolio(symbols,weights,dates)
    # print p1.cum_portfolio_return()
    # print p1.compute_port_returns_from_start()

    ############### PRINT TESTS
    # print "Cumulative Return for Each asset:"
    # print p1.cum_return_by_asset()
    # print "Cumulative Return:"
    # print p1.cum_portfolio_return()
    # print "Avg Daily Return of each asset:"
    # print p1.avg_return_by_asset()
    # print "Average Daily Return of the Portfolio"
    # print p1.avg_daily_return_port()
    # print "Standard Deviation of Daily Return By Asset:"
    # print p1.std_dev_by_asset()
    # print "Standard Deviation of Portfolio's Daily Return"
    # print p1.port_std_dev_daily_returns()
    # print "Sharpe Ratio or Risk adjusted return (Risk-free Rate of zero):"
    # print p1.sharpe_ratio()
    ############### PRINT TESTS
    p1_unopt = p1.compute_port_returns_from_start()
    p1.plot_data(p1.compute_port_returns_from_start())
    result = fit_poly(p1,error_poly)
    # print p1.compute_port_returns_from_start()
    # print("Nope not yet!\n")
    plt.cla()
    p2 = Portfolio(['GOOG','AAPL','GLD','XOM'],weights,dates)
    p2.plot_data(p2.compute_port_returns_from_start())
    p1.plot_data(p1.compute_port_returns_from_start())
    plt.show()


def pad_leading_zeros(c,port):
    for elem in range(abs(len(c) - len(port.weights))):
        c.insert(0,0.0)
    return c
if __name__ == "__main__":
    test_run()