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


# Untested
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


def prettify(ax, title=""):
    if ax.get_ylim()[0] < 0:
        ax.set_ylim(ymin=0)
    ax.set_xlim(config.start_date, config.end_date)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend(loc="upper left")
    if title != "":
        ax.set_title(title)
    # fig.autofmt_xdate()
    # fig.set_size_inches(16, 9)


# Class that tries to run tests in order, isnt working
import unittest
class SequentialTestLoader(unittest.TestLoader):
    def getTestCaseNames(self, testCaseClass):
        test_names = super().getTestCaseNames(testCaseClass)
        testcase_methods = list(testCaseClass.__dict__.keys())
        test_names.sort(key=testcase_methods.index)
        return test_names


# TODO: change this to use logging library
def debug(s):
    import inspect
    import pandas
    import matplotlib
    if config.debug:
        print(inspect.stack()[1].filename)
        if isinstance(s, pandas.DataFrame) or isinstance(s, pandas.Series):
            print(s if config.verbose else s.head(), flush=True)
        elif isinstance(s, matplotlib.axes.Axes):
            pass
            # matplotlib.pyplot.show()
            # TODO: close the grpah afterwards
            # TODO: set active axes/figure back to s. Until this is fixed, don't debug graphs
        elif isinstance(s, matplotlib.pyplot.Figure):
            matplotlib.pyplot.show()
        else:
            print(s, flush=True)
