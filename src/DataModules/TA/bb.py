import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import prices
import config
import utils
import stats as ta

from pandas_datareader._utils import RemoteDataError

table_filename = "BB.csv"
graph_filename = ".png"

default_period = 20
default_std = 2


def bb(symbol, period=default_period, std=default_std, refresh=False, start_date=config.start_date, end_date=config.end_date):
    """Calculates the bollinger bands for the given symbol, saves this data in a .csv file, and returns this data
    The BB is a lagging volatility indicator.

    Parameters:
        symbol : str
        period : int, optional
        std : int, optional
        refresh : bool, optional
        start_date : date, optional
        end_date : date, optional

    Returns:
        dataframe
            A dataframe containing the bollinger bands for the given symbol
    """

    if not utils.refresh(utils.get_file_path(config.ta_data_path, table_filename, symbol=symbol), refresh=refresh):
        df = pd.read_csv(utils.get_file_path(config.ta_data_path, table_filename, symbol=symbol), index_col="Date", parse_dates=["Date"])[start_date:end_date]
    else:
        if utils.refresh(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=symbol), refresh=refresh):
            prices.download_data_from_yahoo(symbol, start_date=start_date, end_date=end_date)
        df = pd.read_csv(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=symbol), usecols=["Date", "Close"], index_col="Date", parse_dates=["Date"])[start_date:end_date]

    if "Lower" not in df.columns or "Upper" not in df.columns:
        df["Mid"] = df["Close"].rolling(window=period, min_periods=period).mean()
        df["Std"] = df["Close"].rolling(window=period, min_periods=period).std()
        df["Lower"] = df["Mid"] - std * df["Std"]
        df["Upper"] = df["Mid"] + std * df["Std"]

        utils.debug(df["Lower"])
        utils.debug(df["Upper"])
        df.to_csv(utils.get_file_path(config.ta_data_path, table_filename, symbol=symbol))

    return [df["Lower"], df["Upper"]]


def plot_bb(symbol, period=default_period, std=default_std, refresh=False, start_date=config.start_date, end_date=config.end_date):
    """Calculates the bollinger bands for each period for the given symbol, saves this data in a .csv file, and plots this data
    The BB is a lagging volatility indicator.

    Parameters:
        symbol : str
        period : int, optional
        std : int, optional
        refresh : bool, optional
        start_date : date, optional
        end_date : date, optional

    Returns:
        figure, axes
            A figure and axes containing the bollinger bands for the given symbol
    """

    if not utils.refresh(utils.get_file_path(config.ta_data_path, table_filename, symbol=symbol), refresh=refresh):
        df = pd.read_csv(utils.get_file_path(config.ta_data_path, table_filename, symbol=symbol), index_col="Date", parse_dates=["Date"])[start_date:end_date]
    else:
        if utils.refresh(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=symbol), refresh=refresh):
            prices.download_data_from_yahoo(symbol, start_date=start_date, end_date=end_date)
        df = pd.read_csv(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=symbol), usecols=["Date", "Close"], index_col="Date", parse_dates=["Date"])[start_date:end_date]

    if len(df) < period:
        raise ta.InsufficientDataException("Not enough data to compute a period length of " + str(period))

    fig, ax = plt.subplots(figsize=config.figsize)
    ax.plot(df.index, df["Close"], label="Price")

    if "Lower" not in df.columns or "Upper" not in df.columns:
        df = df.join(bb(symbol, period, std, refresh=False, start_date=start_date, end_date=end_date))
    # if len(df) > p:  # to prevent AttributeError when the column is all None
    ax.plot(df.index, df["Lower"], label="Lower", color="skyblue")
    ax.plot(df.index, df["Upper"], label="Upper", color="skyblue")
    ax.fill_between(df.index, df["Lower"], df["Upper"], color='lightskyblue')

    utils.prettify_ax(ax, title=symbol + "BB", start_date=start_date, end_date=end_date)

    utils.prettify_fig(fig)
    fig.savefig(utils.get_file_path(config.ta_graphs_path, get_signal_name(period, std) + graph_filename, symbol=symbol))
    utils.debug(fig)
    return fig, ax


def generate_signals(symbol, period=default_period, std=default_std, refresh=False, start_date=config.start_date, end_date=config.end_date):
    """Calculates the bollinger bands buy/sell signals for each period for the given symbol, saves this data in a .csv file, and plots this data. Only uses the first and last periods
    The BB is a lagging volatility indicator.

    Parameters:
        symbol : str
        period : int, optional
        std : int, optional
        refresh : bool, optional
        start_date : date, optional
        end_date : date, optional

    Returns:
        dataframe
            A dataframe containing the bollinger bands signals for the given symbol
    """

    bb(symbol, period, std, refresh=False, start_date=start_date, end_date=end_date)
    df = pd.read_csv(utils.get_file_path(config.ta_data_path, table_filename, symbol=symbol), index_col="Date", parse_dates=["Date"])[start_date:end_date]

    signal_column_name = get_signal_name(period=period, std=std)
    if signal_column_name not in df.columns:
        lower_column_name = "Lower"
        upper_column_name = "Upper"

        conditions = [
            ((df["Close"].shift(1) > df[lower_column_name].shift(1)) & (df["Close"] < df[lower_column_name])),  # price crosses lower band; buy signal
            ((df["Close"].shift(1) < df[upper_column_name].shift(1)) & (df["Close"] > df[upper_column_name])),  # price crosses upper band; sell signal
            False,  # ((df["Close"].shift(1) < df["Mid"].shift(1)) & (df["Close"] > df["Mid"]))  # bb breaches the mid line after a buy signal, soft sell
            False  # ((df["Close"].shift(1) > df["Mid"].shift(1)) & (df["Close"] < df["Mid"]))  # bb breaches the mid line after a sell signal, soft buy
        ]

        df[signal_column_name] = np.select(conditions, ta.signals, default=ta.default_signal)
        utils.debug(df[signal_column_name])
        df.to_csv(utils.get_file_path(config.ta_data_path, table_filename, symbol=symbol))

    return df[signal_column_name]


def plot_signals(symbol, period=default_period, std=default_std, refresh=False, start_date=config.start_date, end_date=config.end_date):
    """Plots the bollinger bands buy/sell signals for each period for the given symbol, saves this data in a .csv file, and plots this data. Only uses the first and last periods
    The BB is a lagging volatility indicator.

    Parameters:
        symbol : str
        period : int, optional
        std : int, optional
        refresh : bool, optional
        start_date : date, optional
        end_date : date, optional

    Returns:
        figure, axes
            A figure and axes containing the bollinger bands signals for the given symbol
    """

    generate_signals(symbol, period=period, std=std, refresh=refresh, start_date=start_date, end_date=end_date)
    fig, ax = plot_bb(symbol, period=period, std=std, refresh=refresh, start_date=start_date, end_date=end_date)
    df = pd.read_csv(utils.get_file_path(config.ta_data_path, table_filename, symbol=symbol), index_col="Date", parse_dates=["Date"])[start_date:end_date]

    lower_column_name = "Lower"
    upper_column_name = "Upper"
    signal_column_name = get_signal_name(period=period, std=std)

    buy_signals = df.loc[df[signal_column_name] == ta.buy_signal]
    ax.scatter(buy_signals.index, df.loc[df.index.isin(buy_signals.index)][lower_column_name], label=ta.buy_signal, color=ta.signal_colors[ta.buy_signal], marker=ta.signal_markers[ta.buy_signal], s=config.scatter_size, alpha=config.scatter_alpha)
    ax.plot((buy_signals.index, buy_signals.index), (df.loc[df.index.isin(buy_signals.index)][lower_column_name], buy_signals["Close"]), color=ta.signal_colors[ta.buy_signal])

    sell_signals = df.loc[df[signal_column_name] == ta.sell_signal]
    ax.scatter(sell_signals.index, df.loc[df.index.isin(sell_signals.index)][upper_column_name], label=ta.sell_signal, color=ta.signal_colors[ta.sell_signal], marker=ta.signal_markers[ta.sell_signal], s=config.scatter_size, alpha=config.scatter_alpha)
    ax.plot((sell_signals.index, sell_signals.index), (df.loc[df.index.isin(sell_signals.index)][upper_column_name], sell_signals["Close"]), color=ta.signal_colors[ta.sell_signal])

    soft_buy_signals = df.loc[df[signal_column_name] == ta.soft_buy_signal]
    ax.scatter(soft_buy_signals.index, df.loc[df.index.isin(soft_buy_signals.index)][lower_column_name], label=ta.soft_buy_signal, color=ta.signal_colors[ta.soft_buy_signal], marker=ta.signal_markers[ta.soft_buy_signal], s=config.scatter_size, alpha=config.scatter_alpha)
    ax.plot((soft_buy_signals.index, soft_buy_signals.index), (df.loc[df.index.isin(soft_buy_signals.index)][lower_column_name], soft_buy_signals["Close"]), color=ta.signal_colors[ta.soft_buy_signal])

    soft_sell_signals = df.loc[df[signal_column_name] == ta.soft_sell_signal]
    ax.scatter(soft_sell_signals.index, df.loc[df.index.isin(soft_sell_signals.index)][upper_column_name], label=ta.soft_sell_signal, color=ta.signal_colors[ta.soft_sell_signal], marker=ta.signal_markers[ta.soft_sell_signal], s=config.scatter_size, alpha=config.scatter_alpha)
    ax.plot((soft_sell_signals.index, soft_sell_signals.index), (df.loc[df.index.isin(soft_sell_signals.index)][upper_column_name], soft_sell_signals["Close"]), color=ta.signal_colors[ta.soft_sell_signal])

    utils.prettify_ax(ax, title=symbol + signal_column_name, start_date=start_date, end_date=end_date)

    utils.prettify_fig(fig)
    fig.savefig(utils.get_file_path(config.ta_graphs_path, get_signal_name(period, std) + graph_filename, symbol=symbol))
    utils.debug(fig)

    return fig, ax


def get_signal_name(period=default_period, std=default_std):
    return "BB" + "Signal" + str(period) + "-" + str(std)
