import pandas as pd
import numpy as np

import prices
import config
import utils

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

    df = pd.read_csv(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=symbol), index_col="Date", parse_dates=["Date"])[start_date:end_date]
    return df["Close"].add(df["Dividends"].cumsum())[-1] / df["Close"][0]


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

    df = pd.read_csv(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=symbol), index_col="Date", parse_dates=["Date"])[start_date:end_date]
    return (df["Close"].add(df["Dividends"].cumsum()) / df["Close"].add(df["Dividends"].cumsum()).shift(1)).mean() / ((df["Close"].add(df["Dividends"].cumsum()) / df["Close"].add(df["Dividends"].cumsum()).shift(1)).std() * np.sqrt(252))
    # return df["Close"].add(df["Dividends"].cumsum()).diff().mean() / df["Close"].add(df["Dividends"].cumsum()).diff().std() * np.sqrt(252)


# Pretty sure this one is wrong. Even when no purchases are made, a non-zero sharpe ratio is returned
# In the above sharpe ratio, for no purchases, a sharpe ratio of inf due to div by 0 is generated, which seems more correct
# ok the above comment seems to not be true anymore but I still have no idea what this one is doing
# this one says that the sharpe ratio of spy is 0.01?? Other one says its 5.71
def get_sharpe_ratio2(symbol, start_date=config.start_date, end_date=config.end_date):
    """Returns the sharpe ratio of the given symbol

    Parameters:
        symbol : str
        start_date : date, optional
        end_date : date, optional

    Returns:
        float
            The sharpe ratio of the given symbol
    """

    df = pd.read_csv(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=symbol), index_col="Date", parse_dates=["Date"])[start_date:end_date]
    big_r = df["Close"].add(df["Dividends"].cumsum()).cumsum()
    small_r = (big_r - big_r.shift(1)) / big_r.shift(1)
    sharpe_ratio = small_r.mean() / (small_r.std() * np.sqrt(252))
    return sharpe_ratio

