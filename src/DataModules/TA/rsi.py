import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import prices
import config
import utils
import tautils as ta

table_filename = "RSI.csv"

rsi_graph_filename = "RSI.png"
rsi_table_filename = "RSI.csv"

default_period = 14
rsi_thresholds = {"Low": 30, "High": 70}  # Oversold and overbought


# TODO: This is the ema rsi. Commercial software generally use Wilder's smoothed moving average
def rsi(symbol, period=default_period, refresh=False, start_date=config.start_date, end_date=config.end_date):
    """Calculates the relative strength indexe for the given symbol, saves this data in a .csv file, and returns this data
    The RSI is a leading momentum indicator.

    Parameters:
        symbol : str
        period : int, optional
        refresh : bool, optional
        start_date : date, optional
        end_date : date, optional

    Returns:
        dataframe
            A dataframe containing the relative strength index for the given symbol
    """

    if not utils.refresh(utils.get_file_path(config.ta_data_path, table_filename, symbol=symbol), refresh=refresh):
        df = pd.read_csv(utils.get_file_path(config.ta_data_path, table_filename, symbol=symbol), index_col="Date", parse_dates=["Date"])[start_date:end_date]
    else:
        if utils.refresh(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=symbol), refresh=refresh):
            prices.download_data_from_yahoo(symbol, start_date=start_date, end_date=end_date)
        df = pd.read_csv(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=symbol), usecols=["Date", "Close"], index_col="Date", parse_dates=["Date"])[start_date:end_date]

    if len(df) < period:
        raise ta.InsufficientDataException("Not enough data to compute a period length of " + str(period))

    if ("RSI" + str(period)) not in df.columns:
        delta = df["Close"].diff()
        up, down = delta.copy(), delta.copy()
        up[up < 0], down[down > 0] = 0, 0
        # df["RSI" + str(period)] = 100 - (100 / (1 + up.rolling(period).mean() / down.abs().rolling(period).mean()))  # sma rsi
        df["RSI" + str(period)] = 100 - (100 / (1 + up.ewm(span=period, min_periods=period).mean() / down.abs().ewm(span=period, min_periods=period).mean()))  # ema rsi
        utils.debug(df["RSI" + str(period)])
        df.to_csv(utils.get_file_path(config.ta_data_path, table_filename, symbol=symbol))
    return df["RSI" + str(period)]


def plot_rsi(symbol, period=default_period, refresh=False, start_date=config.start_date, end_date=config.end_date):
    """Calculates the relative strength index for each period for the given symbol, saves this data in a .csv file, and plots this data
    The RSI is a leading momentum indicator.

    Parameters:
        symbol : str
        period : int, optional
        refresh : bool, optional
        start_date : date, optional
        end_date : date, optional

    Returns:
        figure, axes
            A figure and axes containing the relative strength index for the given symbol
    """

    if not utils.refresh(utils.get_file_path(config.ta_data_path, table_filename, symbol=symbol), refresh=refresh):
        df = pd.read_csv(utils.get_file_path(config.ta_data_path, table_filename, symbol=symbol), index_col="Date", parse_dates=["Date"])[start_date:end_date]
    else:
        if utils.refresh(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=symbol), refresh=refresh):
            prices.download_data_from_yahoo(symbol, start_date=start_date, end_date=end_date)
        df = pd.read_csv(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=symbol), usecols=["Date", "Close"], index_col="Date", parse_dates=["Date"])[start_date:end_date]

    if len(df) < period:
        raise ta.InsufficientDataException("Not enough data to compute a period length of " + str(period))

    fig, ax = plt.subplots(2, figsize=config.figsize)
    ax[0].plot(df.index, df["Close"], label="Price")
    utils.prettify_ax(ax[0], title=symbol + "Price", start_date=start_date, end_date=end_date)
    if "RSI" + str(period) not in df.columns:
        df = df.join(rsi(symbol, period, refresh=False, start_date=start_date, end_date=end_date))
    # if len(df) > period:  # to prevent AttributeError when the column is all None
    ax[1].plot(df.index, df["RSI" + str(period)], label="RSI" + str(period))
    ax[1].plot(df.index, [rsi_thresholds["Low"]] * len(df.index), label="Oversold", color="red")
    ax[1].plot(df.index, [50] * len(df.index), color="black")
    ax[1].plot(df.index, [rsi_thresholds["High"]] * len(df.index), label="Overbought", color="red")

    utils.prettify_ax(ax[1], title=symbol + "RSI" + str(period), start_date=start_date, end_date=end_date)

    utils.prettify_fig(fig)
    fig.savefig(utils.get_file_path(config.ta_graphs_path, str(period) + rsi_graph_filename, symbol=symbol))
    utils.debug(fig)
    return fig, ax


def generate_signals(symbol, period=default_period, refresh=False, start_date=config.start_date, end_date=config.end_date):
    """Calculates the rsi buy/sell signals for the given symbol, saves this data in a .csv file, and plots this data. Only uses the first and last periods
    The RSI is a leading momentum indicator.

    Parameters:
        symbol : str
        period : int or list of int, optional
            Must contain 3 values. First value is signal line, second is fast line, third is slow line.
        refresh : bool, optional
        start_date : date, optional
        end_date : date, optional

    Returns:
        dataframe
            A dataframe containing the rsi signals for the given symbol
    """

    rsi(symbol, period=period, refresh=refresh, start_date=start_date, end_date=end_date)
    df = pd.read_csv(utils.get_file_path(config.ta_data_path, table_filename, symbol=symbol), index_col="Date", parse_dates=["Date"])[start_date:end_date]

    signal_column_name = get_signal_name(period=period)
    if signal_column_name not in df.columns:
        rsi_column_name = "RSI" + str(period)

        conditions = [
            (df[rsi_column_name].shift(1) > rsi_thresholds["Low"]) & (df[rsi_column_name] < rsi_thresholds["Low"]),  # rsi breaches lower threshold, buy signal
            (df[rsi_column_name].shift(1) < rsi_thresholds["High"]) & (df[rsi_column_name] > rsi_thresholds["High"]),  # rsi breaches upper threshold, buy signal
            False,  # (df[rsi_column_name].shift(1) > rsi_thresholds["Low"]) & (df[rsi_column_name] < rsi_thresholds["Low"]),  # rsi breaches 50 after a buy signal, soft sell
            False   # (df[rsi_column_name].shift(1) < rsi_thresholds["High"]) & (df[rsi_column_name] > rsi_thresholds["High"])  # rsi breaches 50 after a sell signal, soft buy
        ]

        df[signal_column_name] = np.select(conditions, ta.signals, default=ta.default_signal)
        utils.debug(df[signal_column_name])
        df.to_csv(utils.get_file_path(config.ta_data_path, table_filename, symbol=symbol))

    return df[signal_column_name]

def plot_signals(symbol, period=default_period, refresh=False, start_date=config.start_date, end_date=config.end_date):
    """Plots the rsi buy/sell signals for the given symbol, saves this data in a .csv file, and plots this data. Only uses the first and last periods
    The RSI is a leading momentum indicator.

    Parameters:
        symbol : str
        period : int or list of int, optional
            Must contain 3 values. First value is signal line, second is fast line, third is slow line.
        refresh : bool, optional
        start_date : date, optional
        end_date : date, optional

    Returns:
        figure, axes
            A figure and axes containing the rsi signals for the given symbol
    """

    generate_signals(symbol, period=period, refresh=refresh, start_date=start_date, end_date=end_date)
    fig, ax = plot_rsi(symbol, period=period, refresh=refresh, start_date=start_date, end_date=end_date)
    df = pd.read_csv(utils.get_file_path(config.ta_data_path, table_filename, symbol=symbol), index_col="Date", parse_dates=["Date"])[start_date:end_date]

    rsi_column_name = "RSI" + str(period)
    signal_column_name = get_signal_name(period=period)

    buy_signals = df.loc[df[signal_column_name] == ta.buy_signal]
    ax[0].scatter(buy_signals.index, df.loc[df.index.isin(buy_signals.index)]["Close"], label=ta.buy_signal, color=ta.signal_colors[ta.buy_signal], marker=ta.signal_markers[ta.buy_signal], s=config.scatter_size, alpha=config.scatter_alpha)
    ax[1].scatter(buy_signals.index, df.loc[df.index.isin(buy_signals.index)][rsi_column_name], label=ta.buy_signal, color=ta.signal_colors[ta.buy_signal], marker=ta.signal_markers[ta.buy_signal], s=config.scatter_size, alpha=config.scatter_alpha)

    sell_signals = df.loc[df[signal_column_name] == ta.sell_signal]
    ax[0].scatter(sell_signals.index, df.loc[df.index.isin(sell_signals.index)]["Close"], label=ta.sell_signal, color=ta.signal_colors[ta.sell_signal], marker=ta.signal_markers[ta.sell_signal], s=config.scatter_size, alpha=config.scatter_alpha)
    ax[1].scatter(sell_signals.index, df.loc[df.index.isin(sell_signals.index)][rsi_column_name], label=ta.sell_signal, color=ta.signal_colors[ta.sell_signal], marker=ta.signal_markers[ta.sell_signal], s=config.scatter_size, alpha=config.scatter_alpha)

    utils.prettify_ax(ax[0], title=symbol + "Price", start_date=start_date, end_date=end_date)
    utils.prettify_ax(ax[1], title=symbol + signal_column_name, percentage=True, start_date=start_date, end_date=end_date)

    utils.prettify_fig(fig)
    fig.savefig(utils.get_file_path(config.ta_graphs_path, str(period) + rsi_graph_filename, symbol=symbol))
    utils.debug(fig)

    return fig, ax


def get_signal_name(period=default_period):
    return "RSI" + ta.signal_name + str(period)
