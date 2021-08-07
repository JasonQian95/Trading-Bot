import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import prices
import config
import utils
import stats as ta

table_filename = "VolForecast.csv"
graph_filename = ".png"

short_vol_symbol = "UPRO"  # SVXY is 0.5x # UPRO to long SPY when vol decreases
long_vol_symbol = "SPXU"  # UVXY is 1.5x, VXX is 1x (0.5x in practice) # SPXU to short SPY when vol increases
default_symbols = [short_vol_symbol, long_vol_symbol]

implied_vol_symbol = "^VIX6M"
historical_vol_symbol = "^GSPC"

default_periods = [2, 5]  # period[0] = 10 is also reasonable, and period[1] = 0 should be used if smoothing is not desired

# algo from: https://quantstrattrader.wordpress.com/2014/11/14/volatility-risk-premium-sharpe-2-return-to-drawdown-3/
# Using the actual S&P 500 index, compute the 2-day annualized historical volatility. Subtract that from the VXMT,
# Then, take the 5-day SMA of that difference. If this number is above 0, go long XIV, otherwise go long VXX

# Summary from http://www.naaim.org/wp-content/uploads/2013/10/00R_Easy-Volatility-Investing-+-Abstract-Tony-Cooper.pdf section 9.3 and 9.4
# The main aim of this strategy is to seek to maximize the roll yield by investing in XIV when the VIX
# term structure is in contango and in VXX when the term structure is in backwardation.
# One simply invests in XIV if VIX3M >= VIX and in VXX if VIX3M < VIX.
# Replace VIX3M with 10-day historical volatility with 5-day sma


def volforecast(period=default_periods, refresh=False, start_date=config.start_date, end_date=config.end_date):
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
    if implied_vol_symbol in config.broken_symbols or not utils.refresh(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=implied_vol_symbol), refresh=refresh):
        iv = pd.read_csv(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=implied_vol_symbol), index_col="Date", parse_dates=["Date"])[start_date:end_date]
    else:
        prices.download_data_from_yahoo(implied_vol_symbol, start_date=start_date, end_date=end_date)
        iv = pd.read_csv(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=implied_vol_symbol), usecols=["Date", "Close"], index_col="Date", parse_dates=["Date"])[start_date:end_date]

    if not utils.refresh(utils.get_file_path(config.prices_data_path, table_filename, symbol=historical_vol_symbol), refresh=refresh):
        hv = pd.read_csv(utils.get_file_path(config.prices_data_path, table_filename, symbol=historical_vol_symbol), index_col="Date", parse_dates=["Date"])[start_date:end_date]
    else:
        prices.download_data_from_yahoo(historical_vol_symbol, start_date=start_date, end_date=end_date)
        hv = pd.read_csv(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=historical_vol_symbol), usecols=["Date", "Close"], index_col="Date", parse_dates=["Date"])[start_date:end_date]

    hv = pd.DataFrame({"Close": np.log(hv["Close"] / hv["Close"].shift(1)).rolling(period[0]).std() * 100 * np.sqrt(252)})
    '''
    voldiff = pd.DataFrame({"VolForecast": (hv["Close"] - iv["Close"]).rolling(period[1]).mean(),
                            "ImpliedVolatility": iv["Close"],
                            "HistoricalVolatility": hv["Close"]})
    '''

    # this get_signal_name() column name doesn't incorporate period[0] and period[1]
    if get_signal_name() not in df.columns:
        df[get_signal_name()] = (hv["Close"] - iv["Close"]).rolling(period[1]).mean()
    if implied_vol_symbol + "SMA" + str(period[1]) not in df.columns:
        df[implied_vol_symbol + "SMA" + str(period[1])] = iv["Close"]
    if "HistoricalVolatility" + "SMA" + str(period[0]) not in df.columns:
        df["HistoricalVolatility" + "SMA" + str(period[0])] = hv["Close"]
    df.to_csv(utils.get_file_path(config.ta_data_path, table_filename, symbol=""))
    return df[get_signal_name()]


def plot_volforecast(symbol=default_symbols, period=default_periods, refresh=False, start_date=config.start_date, end_date=config.end_date):

    short_vol_symbol = symbol[0]
    long_vol_symbol = symbol[1]

    if not utils.refresh(utils.get_file_path(config.ta_data_path, table_filename, symbol=""), refresh=refresh):
        df = pd.read_csv(utils.get_file_path(config.ta_data_path, table_filename, symbol=""), index_col="Date", parse_dates=["Date"])[start_date:end_date]
    else:
        volforecast(period=period, refresh=refresh, start_date=start_date, end_date=end_date)
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


def generate_signals(symbol=default_symbols, period=default_periods, refresh=False, start_date=config.start_date, end_date=config.end_date):

    short_vol_symbol = symbol[0]
    long_vol_symbol = symbol[1]

    volforecast(period=period, refresh=refresh, start_date=start_date, end_date=end_date)
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
                ((df[signal_column_name].shift(1) > 0) & (df[signal_column_name] < 0)),  # near term volatility crossed below expected volatility, short VIX
                ((df[signal_column_name].shift(1) < 0) & (df[signal_column_name] > 0)),  # near term volatility crossed above expected volatility, long VIX
                False,
                False
            ]

        long_vol_conditions = [
                ((df[signal_column_name].shift(1) < 0) & (df[signal_column_name] > 0)),  # near term volatility crossed above expected volatility, long VIX
                ((df[signal_column_name].shift(1) > 0) & (df[signal_column_name] < 0)),  # near term volatility crossed below expected volatility, short VIX
                False,
                False
            ]

        short_vol[signal_column_name] = np.select(short_vol_conditions, ta.signals, default=ta.default_signal)
        long_vol[signal_column_name] = np.select(long_vol_conditions, ta.signals, default=ta.default_signal)
        utils.debug(short_vol[signal_column_name])
        utils.debug(long_vol[signal_column_name])

        short_vol.to_csv(utils.get_file_path(config.ta_data_path, table_filename, symbol=short_vol_symbol))
        long_vol.to_csv(utils.get_file_path(config.ta_data_path, table_filename, symbol=long_vol_symbol))

    return short_vol[signal_column_name], long_vol[signal_column_name]


def plot_signals(symbol=default_symbols, period=default_periods, refresh=False, start_date=config.start_date, end_date=config.end_date):

    short_vol_symbol = symbol[0]
    long_vol_symbol = symbol[1]

    generate_signals(symbol=symbol, period=period, refresh=refresh, start_date=start_date, end_date=end_date)
    fig, ax = plot_volforecast(period=period, refresh=refresh, start_date=start_date, end_date=end_date)
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
def get_signal_name(ivol_symbol=None, symbol=default_symbols, period=default_periods):
    ivol_symbol = implied_vol_symbol if ivol_symbol is None else ivol_symbol
    return "VolForecast" + config.vol_dict.get(ivol_symbol)