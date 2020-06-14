import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader.data as dr
import yfinance as yf

import config
import utils

from pandas_datareader._utils import RemoteDataError

price_table_filename = "Price.csv"
price_graph_filename = "Price.png"
daily_return_name = "DailyReturn"
daily_return_table_filename = daily_return_name + ".csv"
daily_return_graph_filename = daily_return_name + ".png"
after_hours_daily_return_name = "AfterHours" + daily_return_name
after_hours_daily_return_graph_filename = after_hours_daily_return_name + ".png"
during_hours_daily_return_name = "DuringHours" + daily_return_name
during_hours_daily_return_graph_filename = during_hours_daily_return_name + ".png"

avg_name = "Average"
total_return_name = "TotalReturn"
total_return_graph_filename = total_return_name + ".png"
total_after_hours_name = "TotalAfterHoursReturn"
total_during_hours_name = "TotalDuringHoursReturn"
total_after_hours_normalized_name = "TotalAfterHoursNormalizedReturn"
total_during_hours_normalized_name = "TotalDuringHoursNormalizedReturn"
during_and_after_hours_name = "AfterHoursAndDuringHoursReturns"
during_and_after_hours_table_filename = during_and_after_hours_name + ".csv"
during_and_after_hours_graph_filename = during_and_after_hours_name + ".png"
during_and_after_hours_normalized_name = "AfterHoursAndDuringHoursNormalizedReturns"


def download_data_from_yahoo(symbol, adjust=True, backfill=False, start_date=config.start_date, end_date=config.end_date):
    """Generates a .csv file containing the high, low, open, close, volume, and adjusted close by date from Yahoo for the given symbol
    For currency conversion rates, the format is like "USDCAD=X"
    For the S&P500 index, Yahoo uses the ticker GSPX. This function is hardcoded to query Yahoo using GSPX instead of SP500, but save the data to SP500.

    Parameters:
        symbol : str
        adjust : bool
            Whether to incluse the AdjClose column or not. Uses my implementation, not Yahoo's
        backfill : bool, optional
        start_date : date, optional
        end_date : date, optional

    Returns:
        dataframe
            A dataframe containing high, low, open, close, volume, and adjusted close by date for the given symbol
    """

    df = yf.Ticker(symbol).history(auto_adjust=False, start=start_date, end=end_date)

    # This causes an error with pandas, so we're using yfinance instead
    # if dividends causes errors, go to the error and remove the 'eval'
    # df = dr.get_data_yahoo(symbol, start_date, end_date)
    # df = df.merge(dr.DataReader(symbol, 'yahoo-dividends', start_date, end_date), how='outer', left_index=True, right_index=True)

    if symbol.upper() in yf.shared._ERRORS:
        raise RemoteDataError("No data fetched for symbol " + symbol + " using yfinance")

    df.drop("Adj Close", axis=1, inplace=True)
    df.drop("Stock Splits", axis=1, inplace=True)

    if adjust:
        df["AdjClose"] = df["Close"].add(df["Dividends"].cumsum())

    if backfill:
        utils.backfill(df)

    # df.sort_index(inplace=True)  # Yahoo data is sorted anyways

    utils.debug(df)
    df.to_csv(utils.get_file_path(config.prices_data_path, price_table_filename, symbol=symbol))
    return df


def get_dividend_adjusted_price(symbol, columns=["Close"], refresh=False, start_date=config.start_date, end_date=config.end_date):
    """Adjusted all ["Open", "High", "Low", "Close"] columns for dividends.

    Parameters:
        symbol : str
        columns : list of str, str
        The columns to use. The list of expected columns are ["Open", "High", "Low", "Close"],
            but fewer columns may be valid as well
        refresh : bool, optional
        start_date : date, optional
        end_date : date, optional

    Returns:
        dataframe
            A dataframe containing adjusted prices by date for the given symbol
    """

    if utils.refresh(utils.get_file_path(config.prices_data_path, price_table_filename, symbol=symbol), refresh=refresh):
        download_data_from_yahoo(symbol, start_date=start_date, end_date=end_date)
    df = pd.read_csv(utils.get_file_path(config.prices_data_path, price_table_filename, symbol=symbol), index_col="Date", parse_dates=["Date"])[start_date:end_date]

    if isinstance(columns, str):
        columns = [columns]

    for c in columns:
        df["Adj" + c] = df[c].add(df["Dividends"].cumsum())
        utils.debug(df["Adj" + c])
    df.to_csv(utils.get_file_path(config.prices_data_path, price_table_filename, symbol=symbol))
    return df[("Adj" + c for c in columns)]


def get_average_price(symbol, columns=["Open", "High", "Low", "Close"], refresh=False, start_date=config.start_date, end_date=config.end_date):
    """Creates a list of the average price by date for the given symbol, adds the data to the existing corresponding .csv file, and returns this data
    If no valid columns are provided in method, the data will be all nulls

    Parameters:
        symbol : str
        columns: list of str
            The columns to use. The list of expected columns are ["Open", "High", "Low", "Close"],
            but fewer columns may be valid as well
        refresh : bool, optional
        start_date : date, optional
        end_date : date, optional

    Returns:
        dataframe
            A dataframe containing average price by date for the given symbol
    """

    if utils.refresh(utils.get_file_path(config.prices_data_path, price_table_filename, symbol=symbol), refresh=refresh):
        download_data_from_yahoo(symbol, start_date=start_date, end_date=end_date)
    df = pd.read_csv(utils.get_file_path(config.prices_data_path, price_table_filename, symbol=symbol), index_col="Date", parse_dates=["Date"])[start_date:end_date]

    if avg_name not in df.columns:
        df[avg_name] = 0
        count = sum(c in df.columns for c in columns)
        for c in columns:
            if c in df.columns:
                df[avg_name] = df[avg_name].add(df[c])
        df[avg_name] = df[avg_name] / count  # Will leave null values if no methods were valid
        utils.debug(df[avg_name])
        df.to_csv(utils.get_file_path(config.prices_data_path, price_table_filename, symbol=symbol))
    return df[avg_name]


def plot_prices(symbol, refresh=False, start_date=config.start_date, end_date=config.end_date):
    """Plots a graph of the close prices for the given symbol

    Parameters:
        symbol : str
        refresh : bool, optional
        start_date : date, optional
        end_date : date, optional

    Returns:
        figure, axes
            A subplot containing the prices for the given symbol
    """

    if isinstance(symbol, str):
        symbol = [symbol]
    symbol.sort()

    fig, ax = plt.subplots(figsize=config.figsize)
    for s in symbol:
        if utils.refresh(utils.get_file_path(config.prices_data_path, price_table_filename, symbol=s), refresh=refresh):
            download_data_from_yahoo(s, start_date=start_date, end_date=end_date)
        df = pd.read_csv(utils.get_file_path(config.prices_data_path, price_table_filename, symbol=s), index_col="Date", parse_dates=["Date"])[start_date:end_date]

        ax.plot(df.index, df["Close"], label=s + "Price")
    utils.prettify_ax(ax, title="".join(str(s) for s in symbol) + "Price", start_date=start_date, end_date=end_date)

    utils.prettify_fig(fig)
    fig.savefig(utils.get_file_path(config.prices_graphs_path, price_graph_filename, symbol=",".join(str(s) for s in symbol)))
    utils.debug(fig)
    return fig, ax


def plot_percentage_gains(symbol, refresh=False, start_date=config.start_date, end_date=config.end_date):
    """Plots a graph of the percentage gains for the given symbol

    Parameters:
        symbol : str
        refresh : bool, optional
        start_date : date, optional
        end_date : date, optional

    Returns:
        figure, axes
            A subplot containing the percentage gains for the given symbol
    """

    if isinstance(symbol, str):
        symbol = [symbol]
    symbol.sort()

    fig, ax = plt.subplots(figsize=config.figsize)
    for s in symbol:
        if utils.refresh(utils.get_file_path(config.prices_data_path, price_table_filename, symbol=s), refresh=refresh):
            download_data_from_yahoo(s, start_date=start_date, end_date=end_date)
        df = pd.read_csv(utils.get_file_path(config.prices_data_path, price_table_filename, symbol=s), index_col="Date", parse_dates=["Date"])[start_date:end_date]

        ax.plot(df.index, df["Close"] / df["Close"][0], label=s + "Price")

    utils.prettify_ax(ax, title="".join(str(s) for s in symbol) + "Price", start_date=start_date, end_date=end_date)

    utils.prettify_fig(fig)
    fig.savefig(utils.get_file_path(config.prices_graphs_path, price_graph_filename, symbol=",".join(str(s) for s in symbol)))
    utils.debug(fig)
    return fig, ax


def get_daily_return(symbol, period=["daily", "after_hours", "during_hours"], refresh=False, start_date=config.start_date, end_date=config.end_date):
    """Plots a graph of the daily return for the given symbol, adds the data to the existing corresponding .csv file, and returns this data

    Parameters:
        symbol : str
        period : str
            Valid values are "daily", "after_hours", "during_hours"
        refresh : bool, optional
        start_date : date, optional
        end_date : date, optional

    Returns:
        dataframe
            A dataframe containing the daily return by date for the given symbol
    """

    if not utils.refresh(utils.get_file_path(config.prices_data_path, daily_return_table_filename, symbol=symbol), refresh=refresh):
        df = pd.read_csv(utils.get_file_path(config.prices_data_path, daily_return_table_filename, symbol=symbol), index_col="Date", parse_dates=["Date"])[start_date:end_date]
    else:
        if utils.refresh(utils.get_file_path(config.prices_data_path, price_table_filename, symbol=symbol), refresh=refresh):
            download_data_from_yahoo(symbol, start_date=start_date, end_date=end_date)
        df = pd.read_csv(utils.get_file_path(config.prices_data_path, price_table_filename, symbol=symbol), iusecols=["Date", "Open", "Close"], ndex_col="Date", parse_dates=["Date"])[start_date:end_date]

    if isinstance(period, str):
        period = [period]
    used_columns = []

    fig, ax = plt.subplots(len(period) + 1, figsize=config.figsize)  # + 1 to plot against vix

    for i, f in enumerate(period):
        if f == "daily":
            column_name = daily_return_name
            return_name = daily_return_name
        elif f == "after_hours":
            column_name = after_hours_daily_return_name
            return_name = after_hours_daily_return_name
        elif f == "during_hours":
            column_name = during_hours_daily_return_name
            return_name = during_hours_daily_return_name
        else:
            raise ValueError("Valid inputs are 'daily', 'after_hours', and 'during_hours'")

        used_columns.append(column_name)
        if column_name not in df.columns:
            if f == "daily":
                df[column_name] = (df["Close"] / df["Close"].shift(1)) - 1
            if f == "after_hours":
                df[column_name] = (df["Open"] / df["Close"].shift(1)) - 1
            if f == "during_hours":
                df[column_name] = (df["Close"] / df["Open"]) - 1
            utils.debug(df[column_name])
            df.to_csv(utils.get_file_path(config.prices_data_path, daily_return_table_filename, symbol=symbol))

        ax[i].plot(df.index, df[column_name], label=symbol + return_name)
        utils.prettify_ax(ax[i], title=symbol + return_name, center=True, start_date=start_date, end_date=end_date)

    if utils.refresh(utils.get_file_path(config.prices_data_path, price_table_filename, symbol=config.vix), refresh=refresh):
        download_data_from_yahoo(config.vix, start_date=start_date, end_date=end_date)
    vix_df = pd.read_csv(utils.get_file_path(config.prices_data_path, price_table_filename, symbol=config.vix), index_col="Date", parse_dates=["Date"])[start_date:end_date]
    ax[-1].plot(vix_df.index, vix_df["Close"], label=config.vix + "Price")
    utils.prettify_ax(ax[-1], title=config.vix + "Price", start_date=start_date, end_date=end_date)

    utils.prettify_fig(fig)
    fig.savefig(utils.get_file_path(config.prices_graphs_path, ("-".join(str(c) for c in used_columns)), symbol=symbol))
    utils.debug(fig)
    return df[used_columns]


def after_during_hours_returns(symbol, period=0, refresh=False, start_date=config.start_date, end_date=config.end_date):
    """Plots a graph of the after hours and daily hours return for the given symbol, saves this data in a .csv file, and returns this data

    Parameters:
        symbol : str
        period : int
            The number of trading days to look back. There are ~260 trading days in a year
        refresh : bool, optional
        start_date : date, optional
        end_date : date, optional

    Returns:
        dataframe
            A dataframe containing the after hours and daily hours return by date for the given symbol
    """
    if not utils.refresh(utils.get_file_path(config.prices_data_path, daily_return_table_filename, symbol=symbol), refresh=refresh):
        df = pd.read_csv(utils.get_file_path(config.prices_data_path, daily_return_table_filename, symbol=symbol), index_col="Date", parse_dates=["Date"])[start_date:end_date]
    else:
        if utils.refresh(utils.get_file_path(config.prices_data_path, price_table_filename, symbol=symbol), refresh=refresh):
            download_data_from_yahoo(symbol, start_date=start_date, end_date=end_date)
        df = pd.read_csv(utils.get_file_path(config.prices_data_path, price_table_filename, symbol=symbol), index_col="Date", parse_dates=["Date"])[start_date:end_date]

    if period != 0:
        period = -abs(period)
        start_date = df.index[period]
    df = df[start_date:end_date]

    after_hours_cum_sum = pd.Series(df["Open"] - df["Close"].shift(1)).cumsum()
    df[total_after_hours_name] = after_hours_cum_sum + df["Close"][0]
    during_hours_cum_sum = pd.Series(df["Close"] - df["Open"]).cumsum()
    df[total_during_hours_name] = during_hours_cum_sum + df["Close"][0]

    fig, ax = plt.subplots(2, figsize=config.figsize)

    ax[0].plot(df.index, df["Close"], label="Close")
    ax[0].plot(df.index, df[total_after_hours_name], label=total_after_hours_name)
    ax[0].plot(df.index, df[total_during_hours_name], label=total_during_hours_name)

    if config.debug:
        during_hours_cum_sum = pd.Series(df["Close"] - df["Close"].shift(1)).cumsum()
        df[total_return_name] = during_hours_cum_sum + df["Close"][0]
        ax[0].plot(df.index, df[total_return_name], label=total_return_name)
        df = df[["Open", "Close", total_after_hours_name, total_during_hours_name, total_return_name if total_return_name in df.columns else None]]
        df.to_csv(utils.get_file_path(config.prices_data_path, during_and_after_hours_table_filename, symbol=symbol))

    utils.prettify_ax(ax[0], title=symbol + during_and_after_hours_name, start_date=start_date, end_date=end_date)

    df[total_after_hours_normalized_name] = pd.Series(df[total_after_hours_name] - df["Close"])
    df[total_during_hours_normalized_name] = pd.Series(df[total_during_hours_name] - df["Close"])
    ax[1].plot(df.index, df[total_after_hours_normalized_name], label=total_after_hours_normalized_name)
    ax[1].plot(df.index, df[total_during_hours_normalized_name], label=total_during_hours_normalized_name)
    utils.prettify_ax(ax[1], title=symbol + during_and_after_hours_normalized_name, center=True, start_date=start_date, end_date=end_date)

    utils.prettify_fig(fig)
    fig.savefig(utils.get_file_path(config.prices_graphs_path, during_and_after_hours_graph_filename, symbol=symbol, dated=True, start_date=start_date, end_date=end_date))
    utils.debug(fig)
    return df[[total_after_hours_name, total_after_hours_name]]
