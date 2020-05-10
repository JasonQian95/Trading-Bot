import pandas as pd
import pandas_datareader.data as dr
import pandas_datareader.fred as fred
import matplotlib.pyplot as plt
import config
import utils

price_name = "Price"
price_table_filename = price_name + ".csv"
price_graph_filename = price_name + ".png"
daily_return_name = "DailyReturn"
daily_return_graph_filename = daily_return_name + ".png"
after_hours_daily_return_name = "AfterHours" + daily_return_name
after_hours_daily_return_graph_filename = after_hours_daily_return_name + ".png"
during_hours_daily_return_name = "DuringHours" + daily_return_name
during_hours_daily_return_graph_filename = during_hours_daily_return_name + ".png"

avg_column_name = "Average"
daily_return_column_name = daily_return_name
after_hours_daily_return_column_name = after_hours_daily_return_name
during_hours_daily_return_column_name = during_hours_daily_return_name


def download_data_from_fred(symbol, backfill=False, start_date=config.start_date, end_date=config.end_date):
    """Generates a .csv file containing the closing prices by date from FRED for the given symbol.
    As Fred does not have data on individual stock, the main use is get data on SP500 and VIXCLS

    Parameters:
        symbol : str
        backfill : bool, optional
        start_date : date, optional
        end_date : date, optional

    Returns:
        A dataframe containing closing prices by date for the given symbol to a .csv
    """
    
    df = fred.FredReader(symbol, start_date, end_date).read()
    # df = pandas_datareader.DataReader(["df"], "fred", start_date, end_date)
    df.index.rename("Date", inplace=True)
    df.rename(columns={symbol: "Close"}, inplace=True)
    utils.debug(df)

    if backfill:
        utils.backfill(df)

    df.to_csv(utils.get_file_path(config.prices_data_path, price_table_filename, symbol=symbol))
    return df


# TODO: get dividend data
# TODO: get data for other periods
def download_data_from_yahoo(symbol, backfill=False, start_date=config.start_date, end_date=config.end_date):
    """Generates a .csv file containing the high, low, open, close, volume, and adjusted close by date from Yahoo for the given symbol
    For currency conversion rates, the format is like "USDCAD=X"
    For the S&P500 index, Yahoo uses the ticker GSPX. This function is hardcoded to query Yahoo using GSPX instead of SP500, but save the data to SP500.

    Parameters:
        symbol : str
        backfill : bool, optional
        start_date : date, optional
        end_date : date, optional

    Returns:
        A dataframe containing high, low, open, close, volume, and adjusted close by date for the given symbol
    """
    
    df = dr.get_data_yahoo(symbol if symbol != config.sp500 else config.sp500_yahoo, start_date, end_date)
    utils.debug(df)

    if backfill:
        utils.backfill(df)

    df.to_csv(utils.get_file_path(config.prices_data_path, price_table_filename, symbol=symbol))
    return df


def get_average_price(symbol, method=["Open", "High", "Low", "Close"], backfill=False, refresh=False, start_date=config.start_date, end_date=config.end_date):
    """Creates a list of the average price by date for the given symbol, adds the data to the existing corresponding .csv file, and returns this data
    If no valid columns are provided in method, the data will be all nulls

    Parameters:
        symbol : str
        method: list of str
            The columns to use. The list of expected columns are ["Open", "High", "Low", "Close"],
            but other columns may be valid as well
        backfill : bool, optional
        refresh : bool, optional
        start_date : date, optional
        end_date : date, optional
    Returns:
        A dataframe containing average price by date for the given symbol
    """
    if utils.refresh(utils.get_file_path(config.prices_data_path, price_table_filename, symbol=symbol), refresh=refresh):
        download_data_from_yahoo(symbol, backfill=backfill, start_date=start_date, end_date=end_date)
    df = pd.read_csv(utils.get_file_path(config.prices_data_path, price_table_filename, symbol=symbol), index_col="Date")
    if avg_column_name not in df.columns:
        df[avg_column_name] = 0
        # df.insert(len(df.columns), avg_column_name, 0)
        count = sum(e in df.columns for e in method)
        for i in method:
            if i in df.columns:
                df[avg_column_name] = df[avg_column_name].add(df[i])
        df[avg_column_name] = df[avg_column_name] / count  # Will leave null values if no methods were valid
        # TODO: doesnt show column name in debug
        utils.debug(df[avg_column_name])
        df.to_csv(utils.get_file_path(config.prices_data_path, price_table_filename, symbol=symbol))
    return df[avg_column_name]


def plot_prices(symbol, backfill=False, refresh=False, start_date=config.start_date, end_date=config.end_date):
    """Plots a graph of the close prices for the given symbol

    Parameters:
        symbol : str
        backfill : bool, optional
        refresh : bool, optional
        start_date : date, optional
        end_date : date, optional
    """
    
    if utils.refresh(utils.get_file_path(config.prices_data_path, price_table_filename, symbol=symbol), refresh=refresh):
        download_data_from_yahoo(symbol, backfill=backfill, start_date=start_date, end_date=end_date)
    df = pd.read_csv(utils.get_file_path(config.prices_data_path, price_table_filename, symbol=symbol), index_col="Date")
    df["Close"].plot(y="Close", title=symbol+price_name)
    # df.plot(y="Close", title=symbol+price_graph_name)
    if plt.ylim()[0] < 0:
        plt.ylim(ymin=0)
    plt.savefig(utils.get_file_path(config.prices_data_path, price_graph_filename, symbol=symbol))
    utils.debug(plt.gcf())
    plt.close()


def get_daily_return(symbol, backfill=False, refresh=False, start_date=config.start_date, end_date=config.end_date):
    """Plots a graph of the daily return for the given symbol, adds the data to the existing corresponding .csv file, and returns this data

    Parameters:
        symbol : str
        backfill : bool, optional
        refresh : bool, optional
        start_date : date, optional
        end_date : date, optional
    Returns:
        A dataframe containing daily return by date for the given symbol
    """

    if utils.refresh(utils.get_file_path(config.prices_data_path, price_table_filename, symbol=symbol), refresh=refresh):
        download_data_from_yahoo(symbol, backfill=backfill, start_date=start_date, end_date=end_date)
    df = pd.read_csv(utils.get_file_path(config.prices_data_path, price_table_filename, symbol=symbol), index_col="Date")
    if daily_return_column_name not in df.columns:
        df[daily_return_column_name] = (df["Close"] / df["Close"].shift(1)) - 1
        utils.debug(df[daily_return_column_name])
        df.to_csv(utils.get_file_path(config.prices_data_path, price_table_filename, symbol=symbol))
    df[daily_return_column_name].plot(title=symbol+daily_return_name)
    # df.plot(y="daily_return_column_name", title=symbol+daily_returns_graph_filename)
    plt.savefig(utils.get_file_path(config.prices_data_path, daily_return_graph_filename, symbol=symbol))
    utils.debug(plt.gca())
    plt.close()
    return df[daily_return_column_name]


def get_after_hours_daily_return(symbol, backfill=False, refresh=False, start_date=config.start_date, end_date=config.end_date):
    """Plots a graph of the after hours daily return for the given symbol, adds the data to the existing corresponding .csv file, and returns this data

    Parameters:
        symbol : str
        backfill : bool, optional
        refresh : bool, optional
        start_date : date, optional
        end_date : date, optional
    Returns:
        A dataframe containing after hours daily return by date for the given symbol
    """

    if utils.refresh(utils.get_file_path(config.prices_data_path, price_table_filename, symbol=symbol), refresh=refresh):
        download_data_from_yahoo(symbol, backfill=backfill, start_date=start_date, end_date=end_date)
    df = pd.read_csv(utils.get_file_path(config.prices_data_path, price_table_filename, symbol=symbol), index_col="Date")
    if after_hours_daily_return_column_name not in df.columns:
        df[after_hours_daily_return_column_name] = (df["Open"] / df["Close"].shift(1)) - 1
        utils.debug(df[after_hours_daily_return_column_name])
        df.to_csv(utils.get_file_path(config.prices_data_path, price_table_filename, symbol=symbol))

    df[after_hours_daily_return_column_name].plot(title=symbol+after_hours_daily_return_name)
    # df.plot(y="after_hours_daily_return_column_name", title=symbol+after_hours_daily_returns_graph_filename)
    plt.savefig(utils.get_file_path(config.prices_data_path, after_hours_daily_return_graph_filename, symbol=symbol))
    utils.debug(plt.gca())
    plt.close()
    return df[after_hours_daily_return_column_name]


def get_during_hours_daily_return(symbol, backfill=False, refresh=False, start_date=config.start_date, end_date=config.end_date):
    """Plots a graph of the during hours daily return for the given symbol, adds the data to the existing corresponding .csv file, and returns this data

    Parameters:
        symbol : str
        backfill : bool, optional
        refresh : bool, optional
        start_date : date, optional
        end_date : date, optional
    Returns:
        A dataframe containing during hours daily return by date for the given symbol
    """

    if utils.refresh(utils.get_file_path(config.prices_data_path, price_table_filename, symbol=symbol), refresh=refresh):
        download_data_from_yahoo(symbol, backfill=backfill, start_date=start_date, end_date=end_date)
    df = pd.read_csv(utils.get_file_path(config.prices_data_path, price_table_filename, symbol=symbol), index_col="Date")
    if during_hours_daily_return_column_name not in df.columns:
        df[during_hours_daily_return_column_name] = (df["Close"] / df["Open"]) - 1
        utils.debug(df[during_hours_daily_return_column_name])
        df.to_csv(utils.get_file_path(config.prices_data_path, price_table_filename, symbol=symbol))

    df[during_hours_daily_return_column_name].plot(title=symbol+during_hours_daily_return_name)
    # df.plot(y="during_hours_daily_return_column_name", title=symbol+during_hours_daily_returns_graph_filename)
    plt.savefig(utils.get_file_path(config.prices_data_path, during_hours_daily_return_graph_filename, symbol=symbol))
    utils.debug(plt.gca())
    plt.close()
    return df[during_hours_daily_return_column_name]


# TODO: use this func instead of the above three
def get_daily_return_flex(symbol, func="daily", backfill=False, refresh=False, start_date=config.start_date, end_date=config.end_date):
    """Plots a graph of the daily return for the given symbol, adds the data to the existing corresponding .csv file, and returns this data

    Parameters:
        symbol : str
        func : str
            Valid values are "daily", "after_hours", "during_hours"
        backfill : bool, optional
        refresh : bool, optional
        start_date : date, optional
        end_date : date, optional
    Returns:
        A dataframe containing daily return by date for the given symbol
    """

    if utils.refresh(utils.get_file_path(config.prices_data_path, price_table_filename, symbol=symbol), refresh=refresh):
        download_data_from_yahoo(symbol, backfill=backfill, start_date=start_date, end_date=end_date)
    df = pd.read_csv(utils.get_file_path(config.prices_data_path, price_table_filename, symbol=symbol), index_col="Date")

    if func == "daily":
        return_name = daily_return_name
        graph_filename = daily_return_graph_filename
        column_name = daily_return_column_name
    if func == "after_hours":
        return_name = after_hours_daily_return_name
        graph_filename = after_hours_daily_return_graph_filename
        column_name = after_hours_daily_return_column_name
    if func == "during_hours":
        return_name = during_hours_daily_return_name
        graph_filename = during_hours_daily_return_graph_filename
        column_name = during_hours_daily_return_column_name

    if column_name not in df.columns:
        if func == "daily":
            df[column_name] = (df["Close"] / df["Close"].shift(1)) - 1
        if func == "after_hours":
            df[column_name] = (df["Open"] / df["Close"].shift(1)) - 1
        if func == "during_hours":
            df[column_name] = (df["Close"] / df["Open"]) - 1
        utils.debug(df[column_name])
        df.to_csv(utils.get_file_path(config.prices_data_path, price_table_filename, symbol=symbol))

    df[column_name].plot(title=symbol+return_name)
    # df.plot(y="column_name", title=symbol+graph_filename)
    plt.savefig(utils.get_file_path(config.prices_data_path, graph_filename, symbol=symbol))
    utils.debug(plt.gca())
    plt.close()
    return df[column_name]
