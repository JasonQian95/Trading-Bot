import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import prices
import config
import utils
import tautils as ta

ma_table_filename = "MA.csv"

sma_name = "SMA"
ema_name = "EMA"
sma_graph_filename = sma_name + ".png"
ema_graph_filename = ema_name + ".png"

sma_column_name = sma_name
ema_column_name = ema_name

default_periods = [20, 50, 200]


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

    if not utils.refresh(utils.get_file_path(config.ta_data_path, ma_table_filename, symbol=symbol), refresh=refresh):
        df = pd.read_csv(utils.get_file_path(config.ta_data_path, ma_table_filename, symbol=symbol), index_col="Date", parse_dates=["Date"])[start_date:end_date]
    else:
        if utils.refresh(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=symbol), refresh=refresh):
            prices.download_data_from_yahoo(symbol, backfill=backfill, start_date=start_date, end_date=end_date)
        df = pd.read_csv(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=symbol), index_col="Date", parse_dates=["Date"])[start_date:end_date]

    if len(df) < period[-1]:
        raise ta.InsufficientDataException("Not enough data to compute a period length of " + str(period))

    if (sma_column_name + str(period)) not in df.columns:
        df[sma_column_name + str(period)] = df["Close"].rolling(period).mean()
        utils.debug(df[sma_column_name + str(period)])
        df.to_csv(utils.get_file_path(config.ta_data_path, ma_table_filename, symbol=symbol))
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

    if not utils.refresh(utils.get_file_path(config.ta_data_path, ma_table_filename, symbol=symbol), refresh=refresh):
        df = pd.read_csv(utils.get_file_path(config.ta_data_path, ma_table_filename, symbol=symbol), index_col="Date", parse_dates=["Date"])[start_date:end_date]
    else:
        if utils.refresh(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=symbol), refresh=refresh):
            prices.download_data_from_yahoo(symbol, backfill=backfill, start_date=start_date, end_date=end_date)
        df = pd.read_csv(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=symbol), index_col="Date", parse_dates=["Date"])[start_date:end_date]

    if len(df) < period:
        raise ta.InsufficientDataException("Not enough data to compute a period length of " + str(period))

    if (ema_column_name + str(period)) not in df.columns:
        df[ema_column_name + str(period)] = df["Close"].ewm(span=period, min_periods=period, adjust=False).mean()
        utils.debug(df[ema_column_name + str(period)])
        df.to_csv(utils.get_file_path(config.ta_data_path, ma_table_filename, symbol=symbol))
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

    if not utils.refresh(utils.get_file_path(config.ta_data_path, ma_table_filename, symbol=symbol), refresh=refresh):
        df = pd.read_csv(utils.get_file_path(config.ta_data_path, ma_table_filename, symbol=symbol), index_col="Date", parse_dates=["Date"])[start_date:end_date]
    else:
        if utils.refresh(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=symbol), refresh=refresh):
            prices.download_data_from_yahoo(symbol, backfill=backfill, start_date=start_date, end_date=end_date)
        df = pd.read_csv(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=symbol), index_col="Date", parse_dates=["Date"])[start_date:end_date]

    if isinstance(period, int):
        period = [period]
    period.sort()

    if len(df) < period[-1]:
        raise ta.InsufficientDataException("Not enough data to compute a period length of " + str(period))

    fig, ax = plt.subplots(figsize=config.figsize)
    # TODO: replace this with a call to prices.plot_prices?
    ax.plot(df.index, df["Close"], label="Price")
    for p in period:
        column_name = sma_column_name + str(p)
        if column_name not in df.columns:
            # refresh=False causes this to read from ma file, which means ma numbers are not refreshed
            # refresh=True causes this to refresh 3 times  # harcoded refresh=False in sma to avoid this
            df = df.join(sma(symbol, p, backfill=backfill, refresh=False, start_date=start_date, end_date=end_date))
        if len(df) > p:  # to prevent AttributeError when the column is all None
            ax.plot(df.index, df[column_name], label=column_name)
    utils.prettify_ax(ax, title=symbol + "-".join(str(p) for p in period) + sma_name, start_date=start_date, end_date=end_date)

    utils.prettify_fig(fig)
    fig.savefig(utils.get_file_path(config.ta_graphs_path, "-".join(str(p) for p in period) + sma_graph_filename, symbol=symbol))
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

    if not utils.refresh(utils.get_file_path(config.ta_data_path, ma_table_filename, symbol=symbol), refresh=refresh):
        df = pd.read_csv(utils.get_file_path(config.ta_data_path, ma_table_filename, symbol=symbol), index_col="Date", parse_dates=["Date"])[start_date:end_date]
    else:
        if utils.refresh(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=symbol), refresh=refresh):
            prices.download_data_from_yahoo(symbol, backfill=backfill, start_date=start_date, end_date=end_date)
        df = pd.read_csv(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=symbol), index_col="Date", parse_dates=["Date"])[start_date:end_date]

    if isinstance(period, int):
        period = [period]
    period.sort()

    if len(df) < period[-1]:
        raise ta.InsufficientDataException("Not enough data to compute a period length of " + str(period))

    fig, ax = plt.subplots(figsize=config.figsize)
    # TODO: replace this with a call to prices.plot_prices?
    ax.plot(df.index, df["Close"], label="Price")
    for p in period:
        column_name = ema_column_name + str(p)
        if column_name not in df.columns:
            # refresh=False causes this to read from ma file, which means ma numbers are not refreshed
            # refresh=True causes this to refresh 3 times  # harcoded refresh=False in ema to avoid this
            df = df.join(ema(symbol, p, backfill=backfill, refresh=False, start_date=start_date, end_date=end_date))
        if len(df) > p:  # to prevent AttributeError when the column is all None
            ax.plot(df.index, df[column_name], label=column_name)
    utils.prettify_ax(ax, title=symbol + "-".join(str(p) for p in period) + ema_name, start_date=start_date, end_date=end_date)

    utils.prettify_fig(fig)
    fig.savefig(utils.get_file_path(config.ta_graphs_path, "-".join(str(p) for p in period) + ema_graph_filename, symbol=symbol))
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
        fig, ax = plot_sma(symbol, period=period, backfill=backfill, refresh=refresh, start_date=start_date, end_date=end_date)
        column_name = sma_column_name
        graph_filename = sma_graph_filename
    if func == ema_name:
        fig, ax = plot_ema(symbol, period=period, backfill=backfill, refresh=refresh, start_date=start_date, end_date=end_date)
        column_name = ema_column_name
        graph_filename = ema_graph_filename

    df = pd.read_csv(utils.get_file_path(config.ta_data_path, ma_table_filename, symbol=symbol), index_col="Date", parse_dates=["Date"])[start_date:end_date]

    period.sort()

    fast_column_name = column_name + str(period[0])
    slow_column_name = column_name + str(period[-1])

    conditions = [
        ((df[fast_column_name].shift(1) < df[slow_column_name].shift(1)) & (df[fast_column_name] > df[slow_column_name])),  # fast line crosses slow line from below; buy signal
        ((df[fast_column_name].shift(1) > df[slow_column_name].shift(1)) & (df[fast_column_name] < df[slow_column_name])),  # fast line crosses slow line from above; sell signal
        ((df[fast_column_name] < df[slow_column_name]) & (df["Close"].shift(1) < df[fast_column_name].shift(1)) & (df["Close"] > df[fast_column_name])),  # price crosses fast line from below, soft buy
        ((df[fast_column_name] > df[slow_column_name]) & (df["Close"].shift(1) > df[fast_column_name].shift(1)) & (df["Close"] < df[fast_column_name]))  # price crosses fast line from above, soft sell
    ]

    column_name = column_name + ta.signal_name + str(period[0]) + "-" + str(period[-1])
    if column_name not in df.columns:
        df[column_name] = np.select(conditions, ta.signals, default=ta.default_signal)
        utils.debug(df[column_name])

    df[ta.signal_name] = df[column_name]
    df.to_csv(utils.get_file_path(config.ta_data_path, ma_table_filename, symbol=symbol))

    buy_signals = df.loc[df[column_name] == ta.buy_signal]
    ax.scatter(buy_signals.index, df.loc[df.index.isin(buy_signals.index)][fast_column_name], label=ta.buy_signal, color=ta.signal_colors[ta.buy_signal], marker=ta.signal_markers[ta.buy_signal], s=config.scatter_size, alpha=config.scatter_alpha)
    ax.plot((buy_signals.index, buy_signals.index), (df.loc[df.index.isin(buy_signals.index)][fast_column_name], buy_signals["Close"]), color=ta.signal_colors[ta.buy_signal])

    sell_signals = df.loc[df[column_name] == ta.sell_signal]
    ax.scatter(sell_signals.index, df.loc[df.index.isin(sell_signals.index)][fast_column_name], label=ta.sell_signal, color=ta.signal_colors[ta.sell_signal], marker=ta.signal_markers[ta.sell_signal], s=config.scatter_size, alpha=config.scatter_alpha)
    ax.plot((sell_signals.index, sell_signals.index), (df.loc[df.index.isin(sell_signals.index)][fast_column_name], sell_signals["Close"]), color=ta.signal_colors[ta.sell_signal])

    soft_buy_signals = df.loc[df[column_name] == ta.soft_buy_signal]
    ax.scatter(soft_buy_signals.index, df.loc[df.index.isin(soft_buy_signals.index)][fast_column_name], label=ta.soft_buy_signal, color=ta.signal_colors[ta.soft_buy_signal], marker=ta.signal_markers[ta.soft_buy_signal], s=config.scatter_size, alpha=config.scatter_alpha)
    ax.plot((soft_buy_signals.index, soft_buy_signals.index), (df.loc[df.index.isin(soft_buy_signals.index)][fast_column_name], soft_buy_signals["Close"]), color=ta.signal_colors[ta.soft_buy_signal])

    soft_sell_signals = df.loc[df[column_name] == ta.soft_sell_signal]
    ax.scatter(soft_sell_signals.index, df.loc[df.index.isin(soft_sell_signals.index)][fast_column_name], label=ta.soft_sell_signal, color=ta.signal_colors[ta.soft_sell_signal], marker=ta.signal_markers[ta.soft_sell_signal], s=config.scatter_size, alpha=config.scatter_alpha)
    ax.plot((soft_sell_signals.index, soft_sell_signals.index), (df.loc[df.index.isin(soft_sell_signals.index)][fast_column_name], soft_sell_signals["Close"]), color=ta.signal_colors[ta.soft_sell_signal])

    utils.prettify_ax(ax, title=symbol + "-".join(str(p) for p in period) + column_name, start_date=start_date, end_date=end_date)

    utils.prettify_fig(fig)
    fig.savefig(utils.get_file_path(config.ta_graphs_path, "-".join(str(p) for p in period) + graph_filename, symbol=symbol))
    utils.debug(fig)
    return fig, ax

# TODO: MACD
# symbols should be all caps