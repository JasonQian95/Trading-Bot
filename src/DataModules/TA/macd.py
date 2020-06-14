import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import prices
import config
import utils
import tautils as ta

table_filename = "MACD.csv"
graph_filename = "MACD.png"

default_periods = [9, 12, 26]


def macd(symbol, period=default_periods, refresh=False, start_date=config.start_date, end_date=config.end_date):
    """Calculates the exponential moving agerage for the given symbol, saves this data in a .csv file, and returns this data
    The EMA is a lagging trend indicator.

    Parameters:
        symbol : str
        period : int\
        refresh : bool, optional
        start_date : date, optional
        end_date : date, optional

    Returns:
        dataframe
            A dataframe containing the exponential moving agerage for the given symbol
    """

    if not utils.refresh(utils.get_file_path(config.ta_data_path, table_filename, symbol=symbol), refresh=refresh):
        df = pd.read_csv(utils.get_file_path(config.ta_data_path, table_filename, symbol=symbol), index_col="Date", parse_dates=["Date"])[start_date:end_date]
    else:
        if utils.refresh(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=symbol), refresh=refresh):
            prices.download_data_from_yahoo(symbol, start_date=start_date, end_date=end_date)
        df = pd.read_csv(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=symbol), usecols=["Date", "Close"], index_col="Date", parse_dates=["Date"])[start_date:end_date]

    if len(period) != 3:
        raise ValueError("MACD requires 3 periods")
    if len(df) < period[-1]:
        raise ta.InsufficientDataException("Not enough data to compute a period length of " + str(period))

    macd_column_name = "MACD" + str(period[1]) + "-" + str(period[2])
    signal_column_name = "MACD" + str(period[0])
    if macd_column_name not in df.columns or signal_column_name not in df.columns:
        if macd_column_name not in df.columns:
            '''
            # Intermediate steps, can uncomment this part if I want to keep the steps
            slow_column_name = "EMA" + str(period[1])
            if slow_column_name not in df.columns:
                df[slow_column_name] = df["Close"].ewm(span=period[1], min_periods=period[1], adjust=False).mean()
            fast_column_name = "EMA" + str(period[2])
            if fast_column_name not in df.columns:
                df[fast_column_name] = df["Close"].ewm(span=period[2], min_periods=period[2], adjust=False).mean()
            df[macd_column_name] = df[slow_column_name] - df[fast_column_name]
            '''
            df[macd_column_name] = df["Close"].ewm(span=period[1], min_periods=period[1], adjust=False).mean() - df["Close"].ewm(span=period[2], min_periods=period[2], adjust=False).mean()
            utils.debug(df[macd_column_name])

        if signal_column_name not in df.columns:
            df[signal_column_name] = df[macd_column_name].ewm(span=period[0], min_periods=period[0], adjust=False).mean()
            utils.debug(df[signal_column_name])

        df.to_csv(utils.get_file_path(config.ta_data_path, table_filename, symbol=symbol))

    return df[[macd_column_name, signal_column_name]]


def plot_macd(symbol, period=default_periods, refresh=False, start_date=config.start_date, end_date=config.end_date):
    """Calculates the macd for the given symbol, saves this data in a .csv file, and plots this data
    The MACD is a lagging trend indicator.

    Parameters:
        symbol : str
        period : int or list of int, optional
            Must contain 3 values. First value is signal line, second is fast line, third is slow line.\
        refresh : bool, optional
        start_date : date, optional
        end_date : date, optional

    Returns:
        figure, axes
            A figure and axes containing the macd for the given symbol
    """

    if not utils.refresh(utils.get_file_path(config.ta_data_path, table_filename, symbol=symbol), refresh=refresh):
        df = pd.read_csv(utils.get_file_path(config.ta_data_path, table_filename, symbol=symbol), index_col="Date", parse_dates=["Date"])[start_date:end_date]
    else:
        if utils.refresh(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=symbol), refresh=refresh):
            prices.download_data_from_yahoo(symbol, start_date=start_date, end_date=end_date)
        df = pd.read_csv(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=symbol), usecols=["Date", "Close"], index_col="Date", parse_dates=["Date"])[start_date:end_date]

    if len(period) != 3:
        raise ValueError("MACD requires 3 periods")
    if len(df) < period[-1]:
        raise ta.InsufficientDataException("Not enough data to compute a period length of " + str(period))

    fig, ax = plt.subplots(2, figsize=config.figsize)
    ax[0].plot(df.index, df["Close"], label="Price")
    utils.prettify_ax(ax[0], title=symbol + "Price", start_date=start_date, end_date=end_date)

    macd_column_name = "MACD" + str(period[1]) + "-" + str(period[2])
    signal_column_name = "MACD" + str(period[0])
    if macd_column_name not in df.columns or signal_column_name not in df.columns:
        df = df.join(macd(symbol, period, refresh=False, start_date=start_date, end_date=end_date))
    # if len(df) > period[0] and len(df) > period[1] and len(df) > period[2]:  # to prevent AttributeError when the column is all None
    ax[1].plot(df.index, df[macd_column_name], label="MACD")
    ax[1].plot(df.index, df[signal_column_name], label="Signal")
    ax[1].plot(df.index, (df[macd_column_name] - df[signal_column_name]), label="Histogram")
    # Can't overlay a histogram with line plots so the histogram has to also be a line plot
    # ax[1].bar(df.index, np.histogram(np.isfinite(df[signal_column_name] - df[macd_column_name])), normed=True, alpha=config.alpha)  # ValueError: incompatible sizes: argument 'height' must be length 3876 or scalar

    utils.prettify_ax(ax[1], title=symbol + "MACD", center=True, start_date=start_date, end_date=end_date)

    utils.prettify_fig(fig)
    fig.savefig(utils.get_file_path(config.ta_graphs_path, "-".join(str(p) for p in period) + graph_filename, symbol=symbol))
    utils.debug(fig)
    return fig, ax


def generate_signals(symbol, period=default_periods, refresh=False, start_date=config.start_date, end_date=config.end_date):
    """Calculates the macd buy/sell signals for the given symbol, saves this data in a .csv file, and plots this data. Only uses the first and last periods
    The MACD is a lagging trend indicator.

    Parameters:
        symbol : str
        period : int or list of int, optional
            Must contain 3 values. First value is signal line, second is fast line, third is slow line.\
        refresh : bool, optional
        start_date : date, optional
        end_date : date, optional

    Returns:
        dataframe
            A dataframe containing the macd signals for the given symbol
    """

    if len(period) != 3:
        raise ValueError("MACD requires 3 periods")

    macd(symbol, period, refresh=False, start_date=start_date, end_date=end_date)
    df = pd.read_csv(utils.get_file_path(config.ta_data_path, table_filename, symbol=symbol), index_col="Date", parse_dates=["Date"])[start_date:end_date]

    signal_column_name = get_signal_name(period=period)
    if signal_column_name not in df.columns:
        macd_column_name = "MACD" + str(period[1]) + "-" + str(period[2])
        macd_signal_column_name = "MACD" + str(period[0])

        conditions = [
            ((df[macd_column_name].shift(1) < df[macd_signal_column_name].shift(1)) & (df[macd_column_name] > df[macd_signal_column_name]) & (df[macd_signal_column_name] < 0)),  # macd line crosses signal line from below and the crossover occurs below 0; hard buy signal
            ((df[macd_column_name].shift(1) > df[macd_signal_column_name].shift(1)) & (df[macd_column_name] < df[macd_signal_column_name]) & (df[macd_signal_column_name] > 0)),  # macd line crosses signal line from above and the crossover occurs above 0; hard sell signal
            ((df[macd_column_name].shift(1) < df[macd_signal_column_name].shift(1)) & (df[macd_column_name] > df[macd_signal_column_name])),  # macd line crosses signal line from below; buy signal
            ((df[macd_column_name].shift(1) > df[macd_signal_column_name].shift(1)) & (df[macd_column_name] < df[macd_signal_column_name]))  # macd line crosses signal line from above; sell signal
        ]

        df[signal_column_name] = np.select(conditions, ta.signals, default=ta.default_signal)
        utils.debug(df[signal_column_name])
        df.to_csv(utils.get_file_path(config.ta_data_path, table_filename, symbol=symbol))

    return df[signal_column_name]


def plot_signals(symbol, period=default_periods, refresh=False, start_date=config.start_date, end_date=config.end_date):
    """Plots the macd buy/sell signals for the given symbol, saves this data in a .csv file, and plots this data. Only uses the first and last periods
    The MACD is a lagging trend indicator.

    Parameters:
        symbol : str
        period : int or list of int, optional
            Must contain 3 values. First value is signal line, second is fast line, third is slow line.
        refresh : bool, optional
        start_date : date, optional
        end_date : date, optional

    Returns:
        figure, axes
            A figure and axes containing the macd signals for the given symbol
    """

    if len(period) != 3:
        raise ValueError("MACD requires 3 periods")

    generate_signals(symbol, period=period, refresh=refresh, start_date=start_date, end_date=end_date)
    fig, ax = plot_macd(symbol, period=period, refresh=refresh, start_date=start_date, end_date=end_date)
    df = pd.read_csv(utils.get_file_path(config.ta_data_path, table_filename, symbol=symbol), index_col="Date", parse_dates=["Date"])[start_date:end_date]

    macd_column_name = "MACD" + str(period[1]) + "-" + str(period[2])
    signal_column_name = "MACD" + str(period[0])
    signal_column_name = get_signal_name(period=period)

    buy_signals = df.loc[df[signal_column_name] == ta.buy_signal]
    ax[0].scatter(buy_signals.index, df.loc[df.index.isin(buy_signals.index)]["Close"], label=ta.buy_signal, color=ta.signal_colors[ta.buy_signal], marker=ta.signal_markers[ta.buy_signal], s=config.scatter_size, alpha=config.scatter_alpha)
    ax[1].scatter(buy_signals.index, df.loc[df.index.isin(buy_signals.index)][macd_column_name], label=ta.buy_signal, color=ta.signal_colors[ta.buy_signal], marker=ta.signal_markers[ta.buy_signal], s=config.scatter_size, alpha=config.scatter_alpha)

    sell_signals = df.loc[df[signal_column_name] == ta.sell_signal]
    ax[0].scatter(sell_signals.index, df.loc[df.index.isin(sell_signals.index)]["Close"], label=ta.sell_signal, color=ta.signal_colors[ta.sell_signal], marker=ta.signal_markers[ta.sell_signal], s=config.scatter_size, alpha=config.scatter_alpha)
    ax[1].scatter(sell_signals.index, df.loc[df.index.isin(sell_signals.index)][macd_column_name], label=ta.sell_signal, color=ta.signal_colors[ta.sell_signal], marker=ta.signal_markers[ta.sell_signal], s=config.scatter_size, alpha=config.scatter_alpha)

    soft_buy_signals = df.loc[df[signal_column_name] == ta.soft_buy_signal]
    ax[0].scatter(soft_buy_signals.index, df.loc[df.index.isin(soft_buy_signals.index)]["Close"], label=ta.soft_buy_signal, color=ta.signal_colors[ta.soft_buy_signal], marker=ta.signal_markers[ta.soft_buy_signal], s=config.scatter_size, alpha=config.scatter_alpha)
    ax[1].scatter(soft_buy_signals.index, df.loc[df.index.isin(soft_buy_signals.index)][macd_column_name], label=ta.soft_buy_signal, color=ta.signal_colors[ta.soft_buy_signal], marker=ta.signal_markers[ta.soft_buy_signal], s=config.scatter_size, alpha=config.scatter_alpha)

    soft_sell_signals = df.loc[df[signal_column_name] == ta.soft_sell_signal]
    ax[0].scatter(soft_sell_signals.index, df.loc[df.index.isin(soft_sell_signals.index)]["Close"], label=ta.soft_sell_signal, color=ta.signal_colors[ta.soft_sell_signal], marker=ta.signal_markers[ta.soft_sell_signal], s=config.scatter_size, alpha=config.scatter_alpha)
    ax[1].scatter(soft_sell_signals.index, df.loc[df.index.isin(soft_sell_signals.index)][macd_column_name], label=ta.soft_sell_signal, color=ta.signal_colors[ta.soft_sell_signal], marker=ta.signal_markers[ta.soft_sell_signal], s=config.scatter_size, alpha=config.scatter_alpha)

    utils.prettify_ax(ax[0], title=symbol + "Price", start_date=start_date, end_date=end_date)
    utils.prettify_ax(ax[1], title=symbol + signal_column_name, center=True, start_date=start_date, end_date=end_date)

    utils.prettify_fig(fig)
    fig.savefig(utils.get_file_path(config.ta_graphs_path, "-".join(str(p) for p in period) + graph_filename, symbol=symbol))
    utils.debug(fig)

    return fig, ax

def get_signal_name(period=default_periods):
    return "MACD" + ta.signal_name + "-".join(str(p) for p in period)
