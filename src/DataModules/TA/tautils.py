import pandas as pd
import numpy as np

import prices
import config
import utils


'''
from enum import Enum
class Signal(Enum):
    default_signal = ""
    buy_signal = "Buy"
    sell_signal = "Sell"
    soft_buy_signal = "SoftBuy"
    soft_sell_signal = "SoftSell"

    def __str__(self):
        return str(self.value)
'''

signal_name = "Signal"
default_signal = ""
buy_signal = "Buy"
sell_signal = "Sell"
soft_buy_signal = "SoftBuy"
soft_sell_signal = "SoftSell"
signals = [buy_signal, sell_signal, soft_buy_signal, soft_sell_signal]

signal_colors = {
    buy_signal: "green",
    sell_signal: "red",
    soft_buy_signal: "yellow",
    soft_sell_signal: "yellow"
}
# could also use vline but they're too thin
signal_markers = {
    buy_signal: "^",
    sell_signal: "v",
    soft_buy_signal: "^",
    soft_sell_signal: "v"
}


class InsufficientDataException(Exception):
    pass


def get_performance(symbol, start_date=config.start_date, end_date=config.end_date):
    """Returns the overall performance of the given symbol

    Parameters:
        symbol : str
        start_date : date, optional
        end_date : date, optional

    Returns:
        float
            The overall performance of the given symbol
    """

    if utils.refresh(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=symbol), refresh=False):
        prices.download_data_from_yahoo(symbol, start_date=start_date, end_date=end_date)
    df = pd.read_csv(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=symbol), index_col="Date", parse_dates=["Date"])[start_date:end_date]
    return df["Close"].add(df["Dividends"].cumsum())[-1] / df["Close"][0]


def get_annualized_performance(symbol, start_date=config.start_date, end_date=config.end_date):
    """Returns the annualized performance of the given symbol

    Parameters:
        symbol : str
        start_date : date, optional
        end_date : date, optional

    Returns:
        float
            The annualized performance of the given symbol
    """

    if utils.refresh(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=symbol), refresh=False):
        prices.download_data_from_yahoo(symbol, start_date=start_date, end_date=end_date)
    df = pd.read_csv(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=symbol), index_col="Date", parse_dates=["Date"])[start_date:end_date]
    # Not sure about this formula, seems weird. Formula normally has a -1 at the end
    # return (1 + (df["Close"].add(df["Dividends"].cumsum())[-1] / df["Close"][0])) ** (365 / (df.index[-1] - df.index[0]).days)  # exponent equivalent to (252 / len(df.index))
    return (df["Close"].add(df["Dividends"].cumsum())[-1] / df["Close"][0]) / (len(df.index) / 252) + 1


def get_cgar(symbol, start_date=config.start_date, end_date=config.end_date):
    """Returns the compound annual growth rate of the given symbol

    Parameters:
        symbol : str
        start_date : date, optional
        end_date : date, optional

    Returns:
        float
            The compound annual growth rate  of the given symbol
    """

    if utils.refresh(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=symbol), refresh=False):
        prices.download_data_from_yahoo(symbol, start_date=start_date, end_date=end_date)
    df = pd.read_csv(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=symbol), index_col="Date", parse_dates=["Date"])[start_date:end_date]
    # Formula normally has a -1 at the end
    return (df["Close"].add(df["Dividends"].cumsum())[-1] / df["Close"][0]) ** (1 / ((df.index[-1] - df.index[0]).days) / 252)


# 5 yr looks good, 10yr looks off of Yahoo's values for SPY
def get_sharpe_ratio(symbol, start_date=config.start_date, end_date=config.end_date):
    """Returns the sharpe ratio of the given symbol

    Parameters:
        symbol : str
        start_date : date, optional
        end_date : date, optional

    Returns:
        float
            The sharpe ratio of the given symbol
    """

    if utils.refresh(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=symbol), refresh=False):
        prices.download_data_from_yahoo(symbol, start_date=start_date, end_date=end_date)
    df = pd.read_csv(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=symbol), index_col="Date", parse_dates=["Date"])[start_date:end_date]
    # return (df["Close"].add(df["Dividends"].cumsum()) / df["Close"].add(df["Dividends"].cumsum()).shift(1)).mean() / ((df["Close"].add(df["Dividends"].cumsum()) / df["Close"].add(df["Dividends"].cumsum()).shift(1)).std() * np.sqrt(252))
    return df["Close"].add(df["Dividends"].cumsum()).pct_change().mean() / df["Close"].add(df["Dividends"].cumsum()).pct_change().std() * np.sqrt(252)


# Number is off of Yahoo's due to dividends. Numbers match after removing dividends.
def get_beta(symbol_a, symbol_b, start_date=config.start_date, end_date=config.end_date):
    """Returns the beta of symbol_a to symbol_b

    Parameters:
        symbol_a : str
        symbol_b : str
        start_date : date, optional
        end_date : date, optional

    Returns:
        float
            The beta of symbol_a to symbol_b
    """

    if utils.refresh(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=symbol_a), refresh=False):
        prices.download_data_from_yahoo(symbol_a, start_date=start_date, end_date=end_date)
    df_a = pd.read_csv(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=symbol_a), index_col="Date", parse_dates=["Date"])[start_date:end_date]
    a = df_a["Close"].add(df_a["Dividends"].cumsum()).pct_change()[1:]
    if utils.refresh(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=symbol_b), refresh=False):
        prices.download_data_from_yahoo(symbol_b, start_date=start_date, end_date=end_date)
    df_b = pd.read_csv(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=symbol_b), index_col="Date", parse_dates=["Date"])[start_date:end_date]
    b = df_b["Close"].add(df_b["Dividends"].cumsum()).pct_change()[1:]

    # rolling beta
    # df["Beta"] = pd.rolling_cov(df_b["Close"].add(df_a["Dividends"].cumsum()), df_b["Close"].add(df_b["Dividends"].cumsum()), window=window) / pd.rolling_var(df_b["Close"].add(df_b["Dividends"].cumsum()), window=window)

    beta = np.cov(a, b)[0][1] / np.var(b)  # Alternately, np.var(b) -> np.cov(a, b)[1][1]
    return beta
