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
daily_return_table_filename = daily_return_name + ".csv"
daily_return_graph_filename = daily_return_name + ".png"


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


def download_data_from_yahoo(symbol, backfill=False, start_date=config.start_date, end_date=config.end_date):
    """Generates a .csv file containing the high, low, open, close, volume, and adjusted close by date from Yahoo for the given symbol
    For the S&P500 index, use GSPX, not SP500. For currency conversion rates, the format is like "USDCAD=X"

    Parameters:
        symbol : str
        backfill : bool, optional
        start_date : date, optional
        end_date : date, optional

    Returns:
        A dataframe containing high, low, open, close, volume, and adjusted close by date for the given symbol
    """
    
    df = dr.get_data_yahoo(symbol, start_date, end_date)
    utils.debug(df)

    if backfill:
        utils.backfill(df)

    df.to_csv(utils.get_file_path(config.prices_data_path, price_table_filename, symbol=symbol))  # If symbol = "GSPX" maybe pass "SP500" instead?
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
    if "Average" not in df.columns:
        df["Average"] = 0
        count = sum(e in df.columns for e in method)
        for i in method:
            if i in df.columns:
                df["Average"] = df["Average"].add(df[i])
        df["Average"] = df["Average"] / count  # Will leave null values if no methods were valid
        utils.debug(df["Average"])
        df.to_csv(utils.get_file_path(config.prices_data_path, price_table_filename, symbol=symbol))
        return df["Average"]


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
    df["Close"].plot(y="Close", title=symbol + price_name)
    # df.plot(y="Close", title=symbol+price_graph_name)
    plt.ylim(ymin=0)
    plt.savefig(utils.get_file_path(config.prices_data_path, price_graph_filename, symbol=symbol))
    utils.debug(plt.gca())
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

    if "DailyReturn" not in df.columns:
        df["DailyReturn"] = (df["Close"] / df["Close"].shift(1)) - 1
        df["DailyReturn"].plot(title=symbol + daily_return_name)
        # df.plot(y="DailyReturn", title=symbol+daily_returns_graph_name)
        utils.debug(df["DailyReturn"])
        df.to_csv(utils.get_file_path(config.prices_data_path, price_table_filename, symbol=symbol))
        plt.savefig(utils.get_file_path(config.prices_data_path, daily_return_graph_filename, symbol=symbol))
        utils.debug(plt.gca())
        plt.close()
        return df["DailyReturn"]
