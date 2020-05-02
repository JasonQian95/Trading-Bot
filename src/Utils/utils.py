import pandas as pd
from os.path import join, exists
import config


def backfill(df):
    """Fills in the null values in the dataframe. The fill method is "ffill"
    Parameters:
        df : dataframe
    """

    debug(df.isna().sum())
    debug(df[df.isna().any(axis=1)])
    df.fillna(method="ffill", inplace=True)


def reindex(df, backfill=True, start_date=config.start_date, end_date=config.end_date):
    """Reindexes the dataframe for all weekdays, by default leaving the inserted values null

    Parameters:
        df : dataframe
        backfill : bool, optional
        start_date : date, optional
        end_date : date, optional
    """
    weekdays = pd.date_range(start=start_date, end=end_date, freq='B')
    df.reindex(weekdays, inplace=True)
    if backfill:
        backfill(df)


def get_file_path(path, filename, symbol, dated=config.dated, start_date=config.start_date, end_date=config.end_date):
    """Returns a file path combining the given data

    Parameters:
        path : str
        filename : str
        symbol : str
        dated : bool, optional
        start_date : date, optional
        end_date : date, optional

    Returns:
        A file path combining the given data
    """

    return join(path, symbol + ((start_date.strftime(config.date_format) + "-" + end_date.strftime(config.date_format)) if dated else "") + filename)


def refresh(path, refresh=False):
    """Returns whether or not to recreate the file at the given path

    Parameters:
        path : str
            The path to check. It is expected that after checking this path, the user will create a file at this path.
        refresh : bool, optional
            Recreate the data file, regardless of whether or not it already exists

    Returns:
        Whether or not to recreate the file at the given path
    """

    return not exists(path) or refresh


# TODO: change this to use logging library
def debug(s):
    import inspect
    import pandas
    import matplotlib
    if config.debug:
        print(inspect.stack()[1].filename)
        if isinstance(s, pandas.DataFrame):
            print(s if config.verbose else s.head(), flush=True)
        elif isinstance(s, matplotlib.axes.Axes):
            matplotlib.pyplot.show()
            # TODO: close the grpah afterwards
            # TODO: set active axes/figure back to s. Until this is fixed, don't debug graphs
        else:
            print(s, flush=True)
