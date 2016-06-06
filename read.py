import pandas as pd
import matplotlib.pyplot as plt

# def test_run():
#     """Function called by Test Run."""
#     df = pd.read_csv("ml4t/data/AAPL.csv").head()
#     print df
def test_run():
    """Function called by Test Run."""
    plotCloses("IBM")
    #plotClose("IBM")
    # for symbol in ['AAPL', 'IBM']:
    #     print "Mean Volume"
    #     print symbol, get_mean_volume(symbol)

def plotCloses(symbol):
    df = pd.read_csv("ml4t/data/{}.csv".format(symbol))
    df["High"].plot()
    plt.show()

def get_mean_volume(symbol):
    """Return the mean volume for stock indicated by symbol.
    
    Note: Data for a stock is stored in file: data/<symbol>.csv
    """
    df = pd.read_csv("ml4t/data/{}.csv".format(symbol))  # read in data
    # TODO: Compute and return the mean volume for this stock
    return df["Close"].mean()


def symbol_to_path(symbol, base_dir="ml4t/data"):
    """Return CSV file path given ticker symbol."""
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))


def get_data(symbols, dates):
    """Read stock data (adjusted close) for given symbols from CSV files."""
    df = pd.DataFrame(index=dates)
    if 'SPY' not in symbols:  # add SPY for reference, if absent
        symbols.insert(0, 'SPY')

    for symbol in symbols:
        symbol_to_path()

    return df
if __name__ == "__main__":
    test_run()
