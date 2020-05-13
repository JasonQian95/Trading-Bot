import pandas as pd
import matplotlib.pyplot as plt
import prices
import config
import utils

sma_name = "SMA"
ema_name = "EMA"
sma_graph_filename = sma_name + ".png"
ema_graph_filename = ema_name + ".png"

sma_column_name = sma_name
ema_column_name = ema_name

default_periods = [20, 50, 200]
default_colors = ["red", "green", "yellow"]


def sma(symbol, period, backfill=False, refresh=False, start_date=config.start_date, end_date=config.end_date):
    if utils.refresh(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=symbol), refresh=refresh):
        prices.download_data_from_yahoo(symbol, backfill=backfill, start_date=start_date, end_date=end_date)
    df = pd.read_csv(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=symbol), index_col="Date", parse_dates=["Date"])[start_date:end_date]
    if (sma_column_name + str(period)) not in df.columns:
        df[sma_column_name + str(period)] = df["Close"].rolling(period).mean()
        df.to_csv(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=symbol))
    return df[sma_column_name + str(period)]


# TODO: 200 day ema is slightly off. 50 and 20 day emas have no issue. Diverence seems to start around 70
# Data matches 20 and 50 ema from Yahoo Finance, TradingView, and MarketWatch. 200 ema matches TradingView but not the others
def ema(symbol, period, backfill=False, refresh=False, start_date=config.start_date, end_date=config.end_date):
    if utils.refresh(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=symbol), refresh=refresh):
        prices.download_data_from_yahoo(symbol, backfill=backfill, start_date=start_date, end_date=end_date)
    df = pd.read_csv(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=symbol), index_col="Date", parse_dates=["Date"])[start_date:end_date]
    if (ema_column_name + str(period)) not in df.columns:
        df[ema_column_name + str(period)] = df["Close"].ewm(span=period, min_periods=period, adjust=False).mean()
        df.to_csv(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=symbol))
    return df[ema_column_name + str(period)]


def plot_sma(symbol, period=default_periods, backfill=False, refresh=False, start_date=config.start_date, end_date=config.end_date):
    if utils.refresh(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=symbol), refresh=refresh):
        prices.download_data_from_yahoo(symbol, backfill=backfill, start_date=start_date, end_date=end_date)
    df = pd.read_csv(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=symbol), index_col="Date", parse_dates=["Date"])[start_date:end_date]
    fig, ax = plt.subplots(figsize=config.figsize)
    # TODO: replace this with a call to prices.plot_prices?
    ax.plot(df.index, df["Close"], label="Price")
    for p in period:
        column_name = sma_column_name + str(p)
        if column_name not in df.columns:
            df = df.join(sma(symbol, p, backfill=backfill, refresh=refresh, start_date=start_date, end_date=end_date))
        ax.plot(df.index, df[column_name], label=column_name)
        #utils.debug(fig)  # doesn't work
    utils.prettify(ax, title=symbol+"-".join(str(p) for p in period)+sma_name)
    fig.savefig(utils.get_file_path(config.prices_data_path, "-".join(str(p) for p in period) + sma_graph_filename, symbol=symbol))
    utils.debug(fig)
    return fig, ax


def plot_ema(symbol, period=default_periods, backfill=False, refresh=False, start_date=config.start_date, end_date=config.end_date):
    if utils.refresh(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=symbol), refresh=refresh):
        prices.download_data_from_yahoo(symbol, backfill=backfill, start_date=start_date, end_date=end_date)
    df = pd.read_csv(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=symbol), index_col="Date", parse_dates=["Date"])[start_date:end_date]
    fig, ax = plt.subplots(figsize=config.figsize)
    # TODO: replace this with a call to prices.plot_prices?
    ax.plot(df.index, df["Close"], label="Price")
    for p in period:
        column_name = ema_column_name + str(p)
        if column_name not in df.columns:
            df = df.join(ema(symbol, p, backfill=backfill, refresh=refresh, start_date=start_date, end_date=end_date))
        ax.plot(df.index, df[column_name], label=column_name)
        #utils.debug(fig)  # doesn't work
    utils.prettify(ax, title=symbol+"-".join(str(p) for p in period)+ema_name)
    fig.savefig(utils.get_file_path(config.prices_data_path, "-".join(str(p) for p in period) + ema_graph_filename, symbol=symbol))
    utils.debug(fig)
    return fig, ax
