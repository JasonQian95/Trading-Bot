import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import prices
import config
import utils
import tautils as ta

import datetime

table_filename = "VolForecast.csv"
graph_filename = ".png"

long_vix_symbol = "SPXU"  # UVXY is 1.5x, VXX is 1x (0.5x in practice) # SPXU to short SPY when vol increases
short_vix_symbol = "UPRO"  # SVXY is 0.5x # UPRO to long SPY when vol decreases
default_symbols = [long_vix_symbol, short_vix_symbol]

default_periods = [2, 5]

long_term_vix_symbol = "^VIX6M"
historical_vix_symbol = "^GSPC"

# algo from: https://quantstrattrader.wordpress.com/2014/11/14/volatility-risk-premium-sharpe-2-return-to-drawdown-3/


def volforecast(period=default_periods, refresh=False, start_date=config.start_date, end_date=config.end_date):
    # if start_date < datetime.date(2018, 3, 1):
    #     raise ta.InsufficientDataException("UVXY and SVXY had their leveraged changes on Feb 27 2018, data before than will not apply now")
    # if start_date < datetime.date(2011, 11, 1):
    #     raise ta.InsufficientDataException("UVXY and SVXY inception on Oct 7 2011")
    # if start_date < datetime.date(2009, 10, 1):
    #     raise ta.InsufficientDataException("VIX3M inception on Sept 18 20091, VIX6M inception on Jan 3 2008")

    if not utils.refresh(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=long_term_vix_symbol), refresh=refresh):
        iv = pd.read_csv(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=long_term_vix_symbol), index_col="Date", parse_dates=["Date"])[start_date:end_date]
    else:
        prices.download_data_from_yahoo(long_term_vix_symbol, start_date=start_date, end_date=end_date)
        iv = pd.read_csv(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=long_term_vix_symbol), usecols=["Date", "Close"], index_col="Date", parse_dates=["Date"])[start_date:end_date]

    if not utils.refresh(utils.get_file_path(config.prices_data_path, table_filename, symbol=historical_vix_symbol), refresh=refresh):
        hv = pd.read_csv(utils.get_file_path(config.prices_data_path, table_filename, symbol=historical_vix_symbol), index_col="Date", parse_dates=["Date"])[start_date:end_date]
    else:
        prices.download_data_from_yahoo(historical_vix_symbol, start_date=start_date, end_date=end_date)
        hv = pd.read_csv(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=historical_vix_symbol), usecols=["Date", "Close"], index_col="Date", parse_dates=["Date"])[start_date:end_date]

    hv = pd.DataFrame({"Close": np.log(hv["Close"] / hv["Close"].shift(1)).rolling(period[0]).std() * 100 * np.sqrt(252)})
    voldiff = pd.DataFrame({"VolForecast": (hv["Close"] - iv["Close"]).rolling(period[1]).mean(),
                            "ImpliedVolatility": iv["Close"],
                            "HistoricalVolatility": hv["Close"]})

    if utils.refresh(utils.get_file_path(config.ta_data_path, table_filename, symbol=""), refresh=refresh):
        voldiff.to_csv(utils.get_file_path(config.ta_data_path, table_filename, symbol=""))
    return voldiff


def plot_volforecast(symbol=default_symbols, period=default_periods, refresh=False, start_date=config.start_date, end_date=config.end_date):

    long_vix_symbol = symbol[0]
    short_vix_symbol = symbol[1]

    if not utils.refresh(utils.get_file_path(config.ta_data_path, table_filename, symbol=""), refresh=refresh):
        df = pd.read_csv(utils.get_file_path(config.ta_data_path, table_filename, symbol=""), index_col="Date", parse_dates=["Date"])[start_date:end_date]
    else:
        volforecast(period=period, refresh=refresh, start_date=start_date, end_date=end_date)
        df = pd.read_csv(utils.get_file_path(config.ta_data_path, table_filename, symbol=""), usecols=["Date", "Close"], index_col="Date", parse_dates=["Date"])[start_date:end_date]

    if not utils.refresh(utils.get_file_path(config.prices_data_path, table_filename, symbol=long_vix_symbol), refresh=refresh):
        long_vix = pd.read_csv(utils.get_file_path(config.prices_data_path, table_filename, symbol=long_vix_symbol), index_col="Date", parse_dates=["Date"])[start_date:end_date]
    else:
        prices.download_data_from_yahoo(long_vix_symbol, start_date=start_date, end_date=end_date)
        long_vix = pd.read_csv(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=long_vix_symbol), usecols=["Date", "Close"], index_col="Date", parse_dates=["Date"])[start_date:end_date]

    if not utils.refresh(utils.get_file_path(config.prices_data_path, table_filename, symbol=short_vix_symbol), refresh=refresh):
        short_vix = pd.read_csv(utils.get_file_path(config.prices_data_path, table_filename, symbol=short_vix_symbol), index_col="Date", parse_dates=["Date"])[start_date:end_date]
    else:
        prices.download_data_from_yahoo(short_vix_symbol, start_date=start_date, end_date=end_date)
        short_vix = pd.read_csv(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=short_vix_symbol), usecols=["Date", "Close"], index_col="Date", parse_dates=["Date"])[start_date:end_date]

    fig, ax = plt.subplots(3, figsize=config.figsize)
    ax[0].plot(df.index, long_vix["Close"], label=long_vix_symbol)
    ax[1].plot(df.index, short_vix["Close"], label=short_vix_symbol)
    ax[2].plot(df.index, df["VolForecast"], label="VolForecast")

    utils.prettify_ax(ax[0], title=long_vix_symbol, start_date=start_date, end_date=end_date)
    utils.prettify_ax(ax[1], title=short_vix_symbol, start_date=start_date, end_date=end_date)
    utils.prettify_ax(ax[2], title="VolForecast", center=True, start_date=start_date, end_date=end_date)

    utils.prettify_fig(fig)
    fig.savefig(utils.get_file_path(config.ta_graphs_path, graph_filename, symbol="VolForecast"))
    utils.debug(fig)
    return fig, ax


def generate_signals(symbol=default_symbols, period=default_periods, refresh=False, start_date=config.start_date, end_date=config.end_date):

    long_vix_symbol = symbol[0]
    short_vix_symbol = symbol[1]

    volforecast(period=period, refresh=refresh, start_date=start_date, end_date=end_date)
    df = pd.read_csv(utils.get_file_path(config.ta_data_path, table_filename, symbol=""), index_col="Date", parse_dates=["Date"])[start_date:end_date]

    if not utils.refresh(utils.get_file_path(config.ta_data_path, table_filename, symbol=long_vix_symbol), refresh=refresh):
        long_vix = pd.read_csv(utils.get_file_path(config.ta_data_path, table_filename, symbol=long_vix_symbol), index_col="Date", parse_dates=["Date"])[start_date:end_date]
    else:
        if utils.refresh(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=long_vix_symbol), refresh=refresh):
            prices.download_data_from_yahoo(long_vix_symbol, start_date=start_date, end_date=end_date)
        long_vix = pd.read_csv(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=long_vix_symbol), usecols=["Date", "Close"], index_col="Date", parse_dates=["Date"])[start_date:end_date]

    if not utils.refresh(utils.get_file_path(config.ta_data_path, table_filename, symbol=short_vix_symbol), refresh=refresh):
        short_vix = pd.read_csv(utils.get_file_path(config.ta_data_path, table_filename, symbol=short_vix_symbol), index_col="Date", parse_dates=["Date"])[start_date:end_date]
    else:
        if utils.refresh(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=short_vix_symbol), refresh=refresh):
            prices.download_data_from_yahoo(short_vix_symbol, start_date=start_date, end_date=end_date)
        short_vix = pd.read_csv(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=short_vix_symbol), usecols=["Date", "Close"], index_col="Date", parse_dates=["Date"])[start_date:end_date]

    long_vix_conditions = [
            ((df["VolForecast"].shift(1) < 0) & (df["VolForecast"] > 0)),  # near term volatility crossed above expected volatility, long VIX
            ((df["VolForecast"].shift(1) > 0) & (df["VolForecast"] < 0)),  # near term volatility crossed below expected volatility, short VIX
            False,
            False
        ]

    short_vix_conditions = [
            ((df["VolForecast"].shift(1) > 0) & (df["VolForecast"] < 0)),  # near term volatility crossed below expected volatility, long VIX
            ((df["VolForecast"].shift(1) < 0) & (df["VolForecast"] > 0)),  # near term volatility crossed above expected volatility, short VIX
            False,
            False
        ]

    long_vix["VolForecast"] = np.select(long_vix_conditions, ta.signals, default=ta.default_signal)
    short_vix["VolForecast"] = np.select(short_vix_conditions, ta.signals, default=ta.default_signal)
    utils.debug(long_vix["VolForecast"])
    utils.debug(short_vix["VolForecast"])
    if utils.refresh(utils.get_file_path(config.ta_data_path, table_filename, symbol=long_vix_symbol), refresh=refresh):
        long_vix.to_csv(utils.get_file_path(config.ta_data_path, table_filename, symbol=long_vix_symbol))
    if utils.refresh(utils.get_file_path(config.ta_data_path, table_filename, symbol=short_vix_symbol), refresh=refresh):
        short_vix.to_csv(utils.get_file_path(config.ta_data_path, table_filename, symbol=short_vix_symbol))

    return long_vix["VolForecast"], short_vix["VolForecast"]


def plot_signals(symbol=default_symbols, period=default_periods, refresh=False, start_date=config.start_date, end_date=config.end_date):

    long_vix_symbol = symbol[0]
    short_vix_symbol = symbol[1]

    generate_signals(symbol=symbol, period=period, refresh=refresh, start_date=start_date, end_date=end_date)
    fig, ax = plot_volforecast(period=period, refresh=refresh, start_date=start_date, end_date=end_date)
    df = pd.read_csv(utils.get_file_path(config.ta_data_path, table_filename, symbol=""), index_col="Date", parse_dates=["Date"])[start_date:end_date]
    long_vix = pd.read_csv(utils.get_file_path(config.ta_data_path, table_filename, symbol=long_vix_symbol), index_col="Date", parse_dates=["Date"])[start_date:end_date]
    short_vix = pd.read_csv(utils.get_file_path(config.ta_data_path, table_filename, symbol=short_vix_symbol), index_col="Date", parse_dates=["Date"])[start_date:end_date]

    buy_signals = long_vix.loc[long_vix["VolForecast"] == ta.buy_signal]
    ax[0].scatter(buy_signals.index, long_vix.loc[long_vix.index.isin(buy_signals.index)]["Close"], label=ta.buy_signal, color=ta.signal_colors[ta.buy_signal], marker=ta.signal_markers[ta.buy_signal], s=config.scatter_size, alpha=config.scatter_alpha)
    ax[2].scatter(buy_signals.index, df.loc[df.index.isin(buy_signals.index)]["VolForecast"], label=ta.buy_signal, color=ta.signal_colors[ta.buy_signal], marker=ta.signal_markers[ta.buy_signal], s=config.scatter_size, alpha=config.scatter_alpha)
    buy_signals = short_vix.loc[short_vix["VolForecast"] == ta.buy_signal]
    ax[1].scatter(buy_signals.index, short_vix.loc[short_vix.index.isin(buy_signals.index)]["Close"], label=ta.buy_signal, color=ta.signal_colors[ta.buy_signal], marker=ta.signal_markers[ta.buy_signal], s=config.scatter_size, alpha=config.scatter_alpha)
    ax[2].scatter(buy_signals.index, df.loc[df.index.isin(buy_signals.index)]["VolForecast"], label=ta.buy_signal, color=ta.signal_colors[ta.buy_signal], marker=ta.signal_markers[ta.buy_signal], s=config.scatter_size, alpha=config.scatter_alpha)

    sell_signals = long_vix.loc[long_vix["VolForecast"] == ta.sell_signal]
    ax[0].scatter(sell_signals.index, long_vix.loc[long_vix.index.isin(sell_signals.index)]["Close"], label=ta.sell_signal, color=ta.signal_colors[ta.sell_signal], marker=ta.signal_markers[ta.sell_signal], s=config.scatter_size, alpha=config.scatter_alpha)
    ax[2].scatter(sell_signals.index, df.loc[df.index.isin(sell_signals.index)]["VolForecast"], label=ta.sell_signal, color=ta.signal_colors[ta.sell_signal], marker=ta.signal_markers[ta.sell_signal], s=config.scatter_size, alpha=config.scatter_alpha)
    sell_signals = short_vix.loc[short_vix["VolForecast"] == ta.sell_signal]
    ax[1].scatter(sell_signals.index, short_vix.loc[short_vix.index.isin(sell_signals.index)]["Close"], label=ta.sell_signal, color=ta.signal_colors[ta.sell_signal], marker=ta.signal_markers[ta.sell_signal], s=config.scatter_size, alpha=config.scatter_alpha)
    ax[2].scatter(sell_signals.index, df.loc[df.index.isin(sell_signals.index)]["VolForecast"], label=ta.sell_signal, color=ta.signal_colors[ta.sell_signal], marker=ta.signal_markers[ta.sell_signal], s=config.scatter_size, alpha=config.scatter_alpha)

    utils.prettify_ax(ax[0], title=long_vix_symbol + "Price", start_date=start_date, end_date=end_date)
    utils.prettify_ax(ax[1], title=short_vix_symbol + "Price", start_date=start_date, end_date=end_date)
    utils.prettify_ax(ax[2], title="VolForecast", center=True, start_date=start_date, end_date=end_date)

    utils.prettify_fig(fig)
    fig.savefig(utils.get_file_path(config.ta_graphs_path, get_signal_name() + graph_filename, symbol=""))
    utils.debug(fig)

    return fig, ax


def get_signal_name(symbol=default_symbols, period=default_periods):
    return "VolForecast"
