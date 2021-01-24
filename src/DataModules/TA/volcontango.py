import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import prices
import config
import utils
import tautils as ta

import datetime

table_filename = "VolContango.csv"
graph_filename = ".png"

short_vol_symbol = "UPRO"  # SVXY is 0.5x # UPRO to long SPY when vol decreases
long_vol_symbol = "SPXU"  # UVXY is 1.5x, VXX is 1x (0.5x in practice) # SPXU to short SPY when vol increases
default_symbols = [short_vol_symbol, long_vol_symbol]

short_term_vol_symbol = "^VIX3M"
long_term_vol_symbol = "^VIX6M"

default_period = 60  # 60, 125 and 150 are reasonable, based on backtests, and 0 should be used to trade contango with no smoothing

# algo from: https://quantstrattrader.wordpress.com/2014/12/04/a-new-volatility-strategy-and-a-heuristic-for-analyzing-robustness/
# Consider VXV and VXMT, the three month and six month implied volatility on the usual SP500.
# Define contango as VXV/VXMT < 1, and backwardation vice versa. Additionally, take an SMA of said ratio.
# Go long VXX when the ratio is greater than 1 and above its SMA, and go long XIV when the converse holds.

# Summary from http://www.naaim.org/wp-content/uploads/2013/10/00R_Easy-Volatility-Investing-+-Abstract-Tony-Cooper.pdf section 9.3
# The main aim of this strategy is to seek to maximize the roll yield by investing in XIV when the VIX
# term structure is in contango and in VXX when the term structure is in backwardation.
# One simply invests in XIV if VIX3M >= VIX and in VXX if VIX > VIX3M.
# To implement the above description exactly, set default_period = 0


def volcontango(period=default_period, refresh=False, start_date=config.start_date, end_date=config.end_date):
    # if start_date < datetime.date(2018, 3, 1):
    #     raise ta.InsufficientDataException("UVXY and SVXY had their leveraged changes on Feb 27 2018, data before than will not apply now")
    # if start_date < datetime.date(2011, 11, 1):
    #     raise ta.InsufficientDataException("UVXY and SVXY inception on Oct 7 2011")
    # if start_date < datetime.date(2009, 10, 1):
    #     raise ta.InsufficientDataException("VIX3M inception on Sept 18 2009, VIX6M inception on Jan 3 2008")

    if not utils.refresh(utils.get_file_path(config.ta_data_path, table_filename, symbol=""), refresh=refresh):
        df = pd.read_csv(utils.get_file_path(config.ta_data_path, table_filename, symbol=""), index_col="Date", parse_dates=["Date"])[start_date:end_date]
    else:
        if utils.refresh(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=""), refresh=refresh):
            df = pd.DataFrame()

    # don't refresh any volatility indices, yahoo doesn't work for them
    if short_term_vol_symbol in config.broken_symbols or not utils.refresh(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=short_term_vol_symbol), refresh=refresh):
        short_term_vol = pd.read_csv(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=short_term_vol_symbol), index_col="Date", parse_dates=["Date"])[start_date:end_date]
    else:
        prices.download_data_from_yahoo(short_term_vol_symbol, start_date=start_date, end_date=end_date)
        short_term_vol = pd.read_csv(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=short_term_vol_symbol), usecols=["Date", "Close"], index_col="Date", parse_dates=["Date"])[start_date:end_date]

    # don't refresh any volatility indices, yahoo doesn't work for them
    if long_term_vol_symbol in config.broken_symbols or not utils.refresh(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=long_term_vol_symbol), refresh=refresh):
        long_term_vol = pd.read_csv(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=long_term_vol_symbol), index_col="Date", parse_dates=["Date"])[start_date:end_date]
    else:
        prices.download_data_from_yahoo(long_term_vol_symbol, start_date=start_date, end_date=end_date)
        long_term_vol = pd.read_csv(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=long_term_vol_symbol), usecols=["Date", "Close"], index_col="Date", parse_dates=["Date"])[start_date:end_date]

    '''
    volcont = pd.DataFrame({"VolContango": ((short_term_vol["Close"] / long_term_vol["Close"]) - 1),
                            "VolContangoSMA": ((short_term_vol["Close"] / long_term_vol["Close"]) - 1).rolling(period).mean(),
                            "LongTermVix": long_term_vol["Close"],
                            "ShortTermVix": short_term_vol["Close"]})
    '''

    # this get_signal_name() column name doesn't incorporate period
    if get_signal_name() not in df.columns:
        df[get_signal_name()] = ((short_term_vol["Close"] / long_term_vol["Close"]) - 1)
        df[get_signal_name() + "SMA" + str(period)] = ((short_term_vol["Close"] / long_term_vol["Close"]) - 1).rolling(period).mean()
    if short_term_vol_symbol not in df.columns:
        df[short_term_vol_symbol] = short_term_vol["Close"]
    if long_term_vol_symbol not in df.columns:
        df[long_term_vol_symbol] = long_term_vol["Close"]
    df.to_csv(utils.get_file_path(config.ta_data_path, table_filename, symbol=""))
    return df[get_signal_name()]


def plot_volcontango(symbol=default_symbols, period=default_period, refresh=False, start_date=config.start_date, end_date=config.end_date):

    short_vol_symbol = symbol[0]
    long_vol_symbol = symbol[1]

    if not utils.refresh(utils.get_file_path(config.ta_data_path, table_filename, symbol=""), refresh=refresh):
        df = pd.read_csv(utils.get_file_path(config.ta_data_path, table_filename, symbol=""), index_col="Date", parse_dates=["Date"])[start_date:end_date]
    else:
        volcontango(period=period, refresh=refresh, start_date=start_date, end_date=end_date)
        df = pd.read_csv(utils.get_file_path(config.ta_data_path, table_filename, symbol=""), index_col="Date", parse_dates=["Date"])[start_date:end_date]

    if not utils.refresh(utils.get_file_path(config.prices_data_path, table_filename, symbol=short_vol_symbol), refresh=refresh):
        short_vol = pd.read_csv(utils.get_file_path(config.prices_data_path, table_filename, symbol=short_vol_symbol), index_col="Date", parse_dates=["Date"])[start_date:end_date]
    else:
        prices.download_data_from_yahoo(short_vol_symbol, start_date=start_date, end_date=end_date)
        short_vol = pd.read_csv(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=short_vol_symbol), usecols=["Date", "Close"], index_col="Date", parse_dates=["Date"])[start_date:end_date]

    if not utils.refresh(utils.get_file_path(config.prices_data_path, table_filename, symbol=long_vol_symbol), refresh=refresh):
        long_vol = pd.read_csv(utils.get_file_path(config.prices_data_path, table_filename, symbol=long_vol_symbol), index_col="Date", parse_dates=["Date"])[start_date:end_date]
    else:
        prices.download_data_from_yahoo(long_vol_symbol, start_date=start_date, end_date=end_date)
        long_vol = pd.read_csv(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=long_vol_symbol), usecols=["Date", "Close"], index_col="Date", parse_dates=["Date"])[start_date:end_date]

    fig, ax = plt.subplots(3, figsize=config.figsize)
    ax[0].plot(df.index, df[get_signal_name()], label=get_signal_name())
    ax[1].plot(df.index, short_vol["Close"], label=short_vol_symbol)
    ax[2].plot(df.index, long_vol["Close"], label=long_vol_symbol)

    utils.prettify_ax(ax[0], title=get_signal_name(), center=True, start_date=start_date, end_date=end_date)
    utils.prettify_ax(ax[1], title=short_vol_symbol, start_date=start_date, end_date=end_date)
    utils.prettify_ax(ax[2], title=long_vol_symbol, start_date=start_date, end_date=end_date)

    utils.prettify_fig(fig)
    fig.savefig(utils.get_file_path(config.ta_graphs_path, graph_filename, symbol=get_signal_name()))
    utils.debug(fig)
    return fig, ax


def generate_signals(symbol=default_symbols, period=default_period, refresh=False, start_date=config.start_date, end_date=config.end_date):

    short_vol_symbol = symbol[0]
    long_vol_symbol = symbol[1]

    volcontango(period=period, refresh=refresh, start_date=start_date, end_date=end_date)
    df = pd.read_csv(utils.get_file_path(config.ta_data_path, table_filename, symbol=""), index_col="Date", parse_dates=["Date"])[start_date:end_date]

    if not utils.refresh(utils.get_file_path(config.ta_data_path, table_filename, symbol=short_vol_symbol), refresh=refresh):
        short_vol = pd.read_csv(utils.get_file_path(config.ta_data_path, table_filename, symbol=short_vol_symbol), index_col="Date", parse_dates=["Date"])[start_date:end_date]
    else:
        if utils.refresh(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=short_vol_symbol), refresh=refresh):
            prices.download_data_from_yahoo(short_vol_symbol, start_date=start_date, end_date=end_date)
        short_vol = pd.read_csv(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=short_vol_symbol), usecols=["Date", "Close"], index_col="Date", parse_dates=["Date"])[start_date:end_date]

    if not utils.refresh(utils.get_file_path(config.ta_data_path, table_filename, symbol=long_vol_symbol), refresh=refresh):
        long_vol = pd.read_csv(utils.get_file_path(config.ta_data_path, table_filename, symbol=long_vol_symbol), index_col="Date", parse_dates=["Date"])[start_date:end_date]
    else:
        if utils.refresh(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=long_vol_symbol), refresh=refresh):
            prices.download_data_from_yahoo(long_vol_symbol, start_date=start_date, end_date=end_date)
        long_vol = pd.read_csv(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=long_vol_symbol), usecols=["Date", "Close"], index_col="Date", parse_dates=["Date"])[start_date:end_date]

    signal_column_name = get_signal_name()
    if signal_column_name not in short_vol.columns or signal_column_name not in long_vol.columns:

        short_vol_conditions = [
                (((df[signal_column_name].shift(1) > 0) & (df[signal_column_name] < 0)) |
                    ((df[signal_column_name].shift(1) > df[signal_column_name + "SMA" + str(period)].shift(1)) & (df[signal_column_name] < df[signal_column_name + "SMA" + str(period)]))),  # short term volatility crossed below long term volatility, short VIX (VolContango crosses below zero, or VolContango crosses below the VolContangoSMA)
                (((df[signal_column_name].shift(1) < 0) & (df[signal_column_name] > 0) & (df[signal_column_name] >= df[signal_column_name + "SMA" + str(period)])) |
                    ((df[signal_column_name] > 0) & (df[signal_column_name].shift(1) < df[signal_column_name + "SMA" + str(period)].shift(1)) & (df[signal_column_name] > df[signal_column_name + "SMA" + str(period)]))),  # short term volatility crossed above long term volatility, long VIX (VolContango is crosses above zero and we are above the VolContangoSMA, or VolContango is above zero and VolContango crosses above the VolContangoSMA)
                False,
                False
            ]
    
        long_vol_conditions = [
                (((df[signal_column_name].shift(1) < 0) & (df[signal_column_name] > 0) & (df[signal_column_name] >= df[signal_column_name + "SMA" + str(period)])) |
                    ((df[signal_column_name] > 0) & (df[signal_column_name].shift(1) < df[signal_column_name + "SMA" + str(period)].shift(1)) & (df[signal_column_name] > df[signal_column_name + "SMA" + str(period)]))),  # short term volatility crossed above long term volatility, long VIX (VolContango is crosses above zero and we are above the VolContangoSMA, or VolContango is above zero and VolContango crosses above the VolContangoSMA)
                (((df[signal_column_name].shift(1) > 0) & (df[signal_column_name] < 0)) |
                    ((df[signal_column_name].shift(1) > df[signal_column_name + "SMA" + str(period)].shift(1)) & (df[signal_column_name] < df[signal_column_name + "SMA" + str(period)]))),  # short term volatility crossed below long term volatility, short VIX (VolContango crosses below zero, or VolContango crosses below the VolContangoSMA)
                False,
                False
            ]
        '''
        short_vol_conditions = [
            (((df[signal_column_name].shift(1) > 0) & (df[signal_column_name] < 0) & (df[signal_column_name] <= df[signal_column_name + "SMA" + str(period)])) |
                ((df[signal_column_name] < 0) & (df[signal_column_name].shift(1) > df[signal_column_name + "SMA" + str(period)].shift(1)) & (df[signal_column_name] < df[signal_column_name + "SMA" + str(period)]))),  # short term volatility crossed below long term volatility, short VIX (VolContango crosses below zero, or VolContango crosses below the VolContangoSMA)
            (((df[signal_column_name].shift(1) < 0) & (df[signal_column_name] > 0)) |
                ((df[signal_column_name].shift(1) < df[signal_column_name + "SMA" + str(period)].shift(1)) & (df[signal_column_name] > df[signal_column_name + "SMA" + str(period)]))),  # short term volatility crossed above long term volatility, long VIX (VolContango is crosses above zero and we are above the VolContangoSMA, or VolContango is above zero and VolContango crosses above the VolContangoSMA)
            False,
            False
        ]

        long_vol_conditions = [
                (((df[signal_column_name].shift(1) < 0) & (df[signal_column_name] > 0) & (df[signal_column_name] >= df[signal_column_name + "SMA" + str(period)])) |
                    ((df[signal_column_name] > 0) & (df[signal_column_name].shift(1) < df[signal_column_name + "SMA" + str(period)].shift(1)) & (df[signal_column_name] > df[signal_column_name + "SMA" + str(period)]))),  # short term volatility crossed above long term volatility, long VIX (VolContango is crosses above zero and we are above the VolContangoSMA, or VolContango is above zero and VolContango crosses above the VolContangoSMA)
                (((df[signal_column_name].shift(1) > 0) & (df[signal_column_name] < 0)) |
                    ((df[signal_column_name].shift(1) > df[signal_column_name + "SMA" + str(period)].shift(1)) & (df[signal_column_name] < df[signal_column_name + "SMA" + str(period)]))),  # short term volatility crossed below long term volatility, short VIX (VolContango crosses below zero, or VolContango crosses below the VolContangoSMA)
                False,
                False
            ]
        '''

        short_vol[signal_column_name] = np.select(short_vol_conditions, ta.signals, default=ta.default_signal)
        long_vol[signal_column_name] = np.select(long_vol_conditions, ta.signals, default=ta.default_signal)
        utils.debug(short_vol[signal_column_name])
        utils.debug(long_vol[signal_column_name])
        short_vol.to_csv(utils.get_file_path(config.ta_data_path, table_filename, symbol=short_vol_symbol))
        long_vol.to_csv(utils.get_file_path(config.ta_data_path, table_filename, symbol=long_vol_symbol))

    return short_vol[signal_column_name], long_vol[signal_column_name]


def plot_signals(symbol=default_symbols, period=default_period, refresh=False, start_date=config.start_date, end_date=config.end_date):

    short_vol_symbol = symbol[0]
    long_vol_symbol = symbol[1]

    generate_signals(symbol=symbol, period=period, refresh=refresh, start_date=start_date, end_date=end_date)
    fig, ax = plot_volcontango(period=period, refresh=refresh, start_date=start_date, end_date=end_date)
    df = pd.read_csv(utils.get_file_path(config.ta_data_path, table_filename, symbol=""), index_col="Date", parse_dates=["Date"])[start_date:end_date]
    short_vol = pd.read_csv(utils.get_file_path(config.ta_data_path, table_filename, symbol=short_vol_symbol), index_col="Date", parse_dates=["Date"])[start_date:end_date]
    long_vol = pd.read_csv(utils.get_file_path(config.ta_data_path, table_filename, symbol=long_vol_symbol), index_col="Date", parse_dates=["Date"])[start_date:end_date]
    signal_column_name = get_signal_name()

    buy_signals = short_vol.loc[short_vol[signal_column_name] == ta.buy_signal]
    ax[0].scatter(buy_signals.index, df.loc[df.index.isin(buy_signals.index)][signal_column_name], label=ta.buy_signal, color=ta.signal_colors[ta.buy_signal], marker=ta.signal_markers[ta.buy_signal], s=config.scatter_size, alpha=config.scatter_alpha)
    ax[1].scatter(buy_signals.index, short_vol.loc[short_vol.index.isin(buy_signals.index)]["Close"], label=ta.buy_signal, color=ta.signal_colors[ta.buy_signal], marker=ta.signal_markers[ta.buy_signal], s=config.scatter_size, alpha=config.scatter_alpha)
    buy_signals = long_vol.loc[long_vol[signal_column_name] == ta.buy_signal]
    ax[0].scatter(buy_signals.index, df.loc[df.index.isin(buy_signals.index)][signal_column_name], label=ta.buy_signal, color=ta.signal_colors[ta.buy_signal], marker=ta.signal_markers[ta.buy_signal], s=config.scatter_size, alpha=config.scatter_alpha)
    ax[2].scatter(buy_signals.index, long_vol.loc[long_vol.index.isin(buy_signals.index)]["Close"], label=ta.buy_signal, color=ta.signal_colors[ta.buy_signal], marker=ta.signal_markers[ta.buy_signal], s=config.scatter_size, alpha=config.scatter_alpha)

    sell_signals = short_vol.loc[short_vol[signal_column_name] == ta.sell_signal]
    ax[0].scatter(sell_signals.index, df.loc[df.index.isin(sell_signals.index)][signal_column_name], label=ta.sell_signal, color=ta.signal_colors[ta.sell_signal], marker=ta.signal_markers[ta.sell_signal], s=config.scatter_size, alpha=config.scatter_alpha)
    ax[1].scatter(sell_signals.index, short_vol.loc[short_vol.index.isin(sell_signals.index)]["Close"], label=ta.sell_signal, color=ta.signal_colors[ta.sell_signal], marker=ta.signal_markers[ta.sell_signal], s=config.scatter_size, alpha=config.scatter_alpha)
    sell_signals = long_vol.loc[long_vol[signal_column_name] == ta.sell_signal]
    ax[0].scatter(sell_signals.index, df.loc[df.index.isin(sell_signals.index)][signal_column_name], label=ta.sell_signal, color=ta.signal_colors[ta.sell_signal], marker=ta.signal_markers[ta.sell_signal], s=config.scatter_size, alpha=config.scatter_alpha)
    ax[2].scatter(sell_signals.index, long_vol.loc[long_vol.index.isin(sell_signals.index)]["Close"], label=ta.sell_signal, color=ta.signal_colors[ta.sell_signal], marker=ta.signal_markers[ta.sell_signal], s=config.scatter_size, alpha=config.scatter_alpha)

    utils.prettify_ax(ax[0], title=signal_column_name, center=True, start_date=start_date, end_date=end_date)
    utils.prettify_ax(ax[1], title=short_vol_symbol + "Price", start_date=start_date, end_date=end_date)
    utils.prettify_ax(ax[2], title=long_vol_symbol + "Price", start_date=start_date, end_date=end_date)

    utils.prettify_fig(fig)
    fig.savefig(utils.get_file_path(config.ta_graphs_path, signal_column_name + graph_filename, symbol=""))
    utils.debug(fig)

    return fig, ax


# symbol and period only to maintain interface
def get_signal_name(sterm_vol_symbol=None, lterm_vol_symbol=None, symbol=default_symbols, period=default_period):
    sterm_vol_symbol = short_term_vol_symbol if sterm_vol_symbol is None else sterm_vol_symbol
    lterm_vol_symbol = long_term_vol_symbol if lterm_vol_symbol is None else lterm_vol_symbol
    return "VolContango" + config.vol_dict.get(sterm_vol_symbol) + config.vol_dict.get(lterm_vol_symbol)