import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import prices
import config
import utils

ma_table_filename = "MA.csv"

sma_name = "SMA"
ema_name = "EMA"
sma_graph_filename = sma_name + ".png"
ema_graph_filename = ema_name + ".jpg"

sma_column_name = sma_name
ema_column_name = ema_name
signal_name = "Signal"
# sma_signal_column_name = sma_column_name + signal_name
# ema_signal_column_name = sma_column_name + signal_name

default_periods = [20, 50, 200]
default_signal = ""
buy_signal = "Buy"
sell_signal = "Sell"
soft_sell_signal = "SoftSell"
signals = [buy_signal, sell_signal, soft_sell_signal]
signal_colors = {
  buy_signal: "green",
  sell_signal: "red",
  soft_sell_signal: "green"
}
signal_markers = {
  buy_signal: "^",
  sell_signal: "v",
  soft_sell_signal: "v"
}


def sma(symbol, period, backfill=False, refresh=False, start_date=config.start_date, end_date=config.end_date):
    """Calculates the simple moving agerage for the given symbol, saves this data in a .csv file, and returns this data

    Parameters:
        symbol : str
        period : int
        backfill : bool, optional
        refresh : bool, optional
        start_date : date, optional
        end_date : date, optional

    Returns:
        dataframe
            A dataframe containing the simple moving agerage for the given symbol
    """

    if not utils.refresh(utils.get_file_path(config.prices_data_path, ma_table_filename, symbol=symbol), refresh=refresh):
        df = pd.read_csv(utils.get_file_path(config.prices_data_path, ma_table_filename, symbol=symbol), index_col="Date", parse_dates=["Date"])[start_date:end_date]
    else:
        if utils.refresh(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=symbol), refresh=refresh):
            prices.download_data_from_yahoo(symbol, backfill=backfill, start_date=start_date, end_date=end_date)
        df = pd.read_csv(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=symbol), index_col="Date", parse_dates=["Date"])[start_date:end_date]

    if (sma_column_name + str(period)) not in df.columns:
        df[sma_column_name + str(period)] = df["Close"].rolling(period).mean()
        utils.debug(df[sma_column_name + str(period)])
        df.to_csv(utils.get_file_path(config.prices_data_path, ma_table_filename, symbol=symbol))
    return df[sma_column_name + str(period)]


# TODO: 200 day ema is slightly off. 50 and 20 day emas have no issue. Diverence seems to start around 70
# Data matches 20 and 50 ema from Yahoo Finance, TradingView, and MarketWatch. 200 ema matches TradingView but not the others
def ema(symbol, period, backfill=False, refresh=False, start_date=config.start_date, end_date=config.end_date):
    """Calculates the exponential moving agerage for the given symbol, saves this data in a .csv file, and returns this data

    Parameters:
        symbol : str
        period : int
        backfill : bool, optional
        refresh : bool, optional
        start_date : date, optional
        end_date : date, optional

    Returns:
        dataframe
            A dataframe containing the exponential moving agerage for the given symbol
    """

    if not utils.refresh(utils.get_file_path(config.prices_data_path, ma_table_filename, symbol=symbol), refresh=refresh):
        df = pd.read_csv(utils.get_file_path(config.prices_data_path, ma_table_filename, symbol=symbol), index_col="Date", parse_dates=["Date"])[start_date:end_date]
    else:
        if utils.refresh(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=symbol), refresh=refresh):
            prices.download_data_from_yahoo(symbol, backfill=backfill, start_date=start_date, end_date=end_date)
        df = pd.read_csv(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=symbol), index_col="Date", parse_dates=["Date"])[start_date:end_date]

    if (ema_column_name + str(period)) not in df.columns:
        df[ema_column_name + str(period)] = df["Close"].ewm(span=period, min_periods=period, adjust=False).mean()
        utils.debug(df[ema_column_name + str(period)])
        df.to_csv(utils.get_file_path(config.prices_data_path, ma_table_filename, symbol=symbol))
    return df[ema_column_name + str(period)]


def plot_sma(symbol, period=default_periods, backfill=False, refresh=False, start_date=config.start_date, end_date=config.end_date):
    """Calculates the simple moving agerage for each period for the given symbol, saves this data in a .csv file, and plots this data

    Parameters:
        symbol : str
        period : int or list of int, optional
        backfill : bool, optional
        refresh : bool, optional
        start_date : date, optional
        end_date : date, optional

    Returns:
        figure, axes
            A subplot containing the simple moving agerage for the given symbol
    """

    if not utils.refresh(utils.get_file_path(config.prices_data_path, ma_table_filename, symbol=symbol), refresh=refresh):
        df = pd.read_csv(utils.get_file_path(config.prices_data_path, ma_table_filename, symbol=symbol), index_col="Date", parse_dates=["Date"])[start_date:end_date]
    else:
        if utils.refresh(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=symbol), refresh=refresh):
            prices.download_data_from_yahoo(symbol, backfill=backfill, start_date=start_date, end_date=end_date)
        df = pd.read_csv(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=symbol), index_col="Date", parse_dates=["Date"])[start_date:end_date]

    if isinstance(period, int):
        period = [period]

    fig, ax = plt.subplots(figsize=config.figsize)
    # TODO: replace this with a call to prices.plot_prices?
    ax.plot(df.index, df["Close"], label="Price")
    for p in period:
        column_name = sma_column_name + str(p)
        if column_name not in df.columns:
            df = df.join(sma(symbol, p, backfill=backfill, refresh=False, start_date=start_date, end_date=end_date))  # we already refreshed, dont refresh 3 times
        ax.plot(df.index, df[column_name], label=column_name)
        #utils.debug(ax)  # doesn't work
    utils.prettify_ax(ax, title=symbol + "-".join(str(p) for p in period) + sma_name)

    utils.prettify_fig(fig)
    fig.savefig(utils.get_file_path(config.prices_graphs_path, "-".join(str(p) for p in period) + sma_graph_filename, symbol=symbol))
    utils.debug(fig)
    return fig, ax


def plot_ema(symbol, period=default_periods, backfill=False, refresh=False, start_date=config.start_date, end_date=config.end_date):
    """Calculates the exponential moving agerage for each period for the given symbol, saves this data in a .csv file, and plots this data

    Parameters:
        symbol : str
        period : int or list of int, optional
        backfill : bool, optional
        refresh : bool, optional
        start_date : date, optional
        end_date : date, optional

    Returns:
        figure, axes
            A figure and axes containing the exponential moving agerage for the given symbol
    """

    if not utils.refresh(utils.get_file_path(config.prices_data_path, ma_table_filename, symbol=symbol), refresh=refresh):
        df = pd.read_csv(utils.get_file_path(config.prices_data_path, ma_table_filename, symbol=symbol), index_col="Date", parse_dates=["Date"])[start_date:end_date]
    else:
        if utils.refresh(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=symbol), refresh=refresh):
            prices.download_data_from_yahoo(symbol, backfill=backfill, start_date=start_date, end_date=end_date)
        df = pd.read_csv(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=symbol), index_col="Date", parse_dates=["Date"])[start_date:end_date]

    if isinstance(period, int):
        period = [period]

    fig, ax = plt.subplots(figsize=config.figsize)
    # TODO: replace this with a call to prices.plot_prices?
    ax.plot(df.index, df["Close"], label="Price")
    for p in period:
        column_name = ema_column_name + str(p)
        if column_name not in df.columns:
            df = df.join(ema(symbol, p, backfill=backfill, refresh=False, start_date=start_date, end_date=end_date))  # we already refreshed, dont refresh 3 times
        ax.plot(df.index, df[column_name], label=column_name)
        #utils.debug(ax)  # doesn't work
    utils.prettify_ax(ax, title=symbol + "-".join(str(p) for p in period) + ema_name)

    utils.prettify_fig(fig)
    fig.savefig(utils.get_file_path(config.prices_graphs_path, "-".join(str(p) for p in period) + ema_graph_filename, symbol=symbol))
    utils.debug(fig)
    return fig, ax


def generate_signals(symbol, func=ema_name, period=default_periods, backfill=False, refresh=False, start_date=config.start_date, end_date=config.end_date):
    """Calculates the moving agerage and buy/sell signals for each period for the given symbol, saves this data in a .csv file, and plots this data

    Parameters:
        symbol : str
        func : str
            What function to calculate the moving average with. Valid valies are "sma" and "ema"
        period : int or list of int, optional
        backfill : bool, optional
        refresh : bool, optional
        start_date : date, optional
        end_date : date, optional

    Returns:
        figure, axes
            A figure and axes containing the exponential moving agerage for the given symbol
    """

    if len(period) < 2:
        raise ValueError("Requires at least two periods")
    if func != sma_name and func != ema_name:
        raise ValueError("Valid functions are 'sma' and 'ema'")

    if func == sma_name:
        fig, ax = plot_sma(symbol, period=default_periods, backfill=backfill, refresh=refresh, start_date=start_date, end_date=end_date)
        column_name = sma_column_name
        graph_filename = sma_graph_filename
    if func == ema_name:
        fig, ax = plot_ema(symbol, period=default_periods, backfill=backfill, refresh=refresh, start_date=start_date, end_date=end_date)
        column_name = ema_column_name
        graph_filename = ema_graph_filename

    df = pd.read_csv(utils.get_file_path(config.prices_data_path, ma_table_filename, symbol=symbol), index_col="Date", parse_dates=["Date"])[start_date:end_date]

    fast_column_name = column_name + str(period[0])
    slow_column_name = column_name + str(period[-1])

    conditions = [
        ((df[fast_column_name].shift(1) < df[slow_column_name].shift(1)) & (df[fast_column_name] > df[slow_column_name])),  # gold cross; buy signal
        ((df[fast_column_name].shift(1) > df[slow_column_name].shift(1)) & (df[fast_column_name] < df[slow_column_name])),  # death cross; sell signal
        ((df[fast_column_name] > df[slow_column_name]) & (df["Close"].shift(1) > df[fast_column_name].shift(1)) & (df["Close"] < df[fast_column_name]))  # custom condition, soft sell
    ]

    column_name = column_name + signal_name + "-".join(str(p) for p in period)
    if column_name not in df.columns:
        df[column_name] = np.select(conditions, signals, default=default_signal)
        utils.debug(df[column_name])
        df.to_csv(utils.get_file_path(config.prices_data_path, ma_table_filename, symbol=symbol))

    buy_signals = df.loc[df[column_name] == buy_signal]
    ax.scatter(buy_signals.index, df.loc[df.index.isin(buy_signals.index)][fast_column_name], label=buy_signal, color=signal_colors[buy_signal], marker=signal_markers[buy_signal], s=config.scatter_size, alpha=config.scatter_alpha)
    ax.plot((buy_signals.index, buy_signals.index), (df.loc[df.index.isin(buy_signals.index)][fast_column_name], buy_signals["Close"]), color=signal_colors[buy_signal])

    sell_signals = df.loc[df[column_name] == sell_signal]
    ax.scatter(sell_signals.index, df.loc[df.index.isin(sell_signals.index)][fast_column_name], label=sell_signal, color=signal_colors[sell_signal], marker=signal_markers[sell_signal], s=config.scatter_size, alpha=config.scatter_alpha)
    ax.plot((sell_signals.index, sell_signals.index), (df.loc[df.index.isin(sell_signals.index)][fast_column_name], sell_signals["Close"]), color=signal_colors[sell_signal])

    soft_sell_signals = df.loc[df[column_name] == soft_sell_signal]
    ax.scatter(soft_sell_signals.index, df.loc[df.index.isin(soft_sell_signals.index)][fast_column_name], label=soft_sell_signal, color=signal_colors[soft_sell_signal], marker=signal_markers[soft_sell_signal], s=config.scatter_size, alpha=config.scatter_alpha)
    ax.plot((soft_sell_signals.index, soft_sell_signals.index), (df.loc[df.index.isin(soft_sell_signals.index)][fast_column_name], soft_sell_signals["Close"]), color=signal_colors[soft_sell_signal])

    utils.prettify_ax(ax, title=symbol + "-".join(str(p) for p in period) + column_name)

    utils.prettify_fig(fig)
    fig.savefig(utils.get_file_path(config.prices_graphs_path, "-".join(str(p) for p in period) + graph_filename, symbol=symbol))
    utils.debug(fig)
    return fig, ax


# TODO: save files in ta_data_path/ta_graphs_path instead
# TODO: only save relevent data in ta_data_path
# TODO: different color + marker for soft sell
