import matplotlib as mpl
from matplotlib.dates import YearLocator, MonthLocator, DateFormatter
import numpy as np
import pandas as pd

import config

import datetime
from enum import Enum
from functools import lru_cache
import inspect
import os
from os.path import join, exists
import sys


def backfill(df, limit=None):
    """Fills in the null values in the dataframe. The fill method is "ffill"
    Parameters:
        df : dataframe, series
        limit : int, optional
    """

    debug(df.isna().sum())
    debug(df[df.isna().any(axis=1)])
    df.fillna(method="ffill", limit=limit, inplace=True)


'''
# Untested, don't think I'd ever need this with Yahoo's data
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
'''


def get_file_path(path, filename, symbol="", dated=config.dated, start_date=config.start_date, end_date=config.end_date):
    """Returns a file path combining the given data

    Parameters:
        path : str
        filename : str
        symbol : str
        dated : bool, optional
        start_date : date, optional
        end_date : date, optional

    Returns:
        str
            A file path combining the given data
    """

    # symbol = symbol.upper()
    start_date = (start_date.replace("-", "_") if isinstance(start_date, str) else start_date.strftime(config.output_date_format))
    end_date = (end_date.replace("-", "_") if isinstance(end_date, str) else end_date.strftime(config.output_date_format))
    return join(path, symbol + ((start_date + ("-" if start_date != "" else "") + end_date) if dated else "") + filename)


def refresh(path, refresh=False):
    """Returns whether or not to recreate the file at the given path

    Parameters:
        path : str
            The path to check. It is expected that after checking this path, the user will create a file at this path.
        refresh : bool, optional
            Recreate the data file, regardless of whether or not it already exists

    Returns:
        bool
            Whether or not to recreate the file at the given path
    """

    # This is required because otherwise, the following will happen:
    # for refesh=True, plot_ema will call ema with refresh=False and ema will read the stale ma files
    # or plot_ema will call ema with refresh=True and ema will read from price files and discard the already generated ma
    if exists(path) and refresh:
        pass
        #os.remove(path)
    return not exists(path) or refresh


@lru_cache(maxsize=128)
def add_business_days(start_date, ndays):
    """Adds business days to a given date. Does not account for holidays.

    Parameters:
        start_date : str, datetime
        ndays : int

    Returns:
        date
            The new date
    """

    is_timestamp = False

    if isinstance(start_date, str):
        current_date = datetime.datetime.strptime(start_date, config.input_date_format).date()
    elif isinstance(start_date, pd.Timestamp):
        current_date = start_date.date()
        is_timestamp = True
    else:
        current_date = start_date

    if ndays != 0:
        sign = ndays/abs(ndays)
        ndays = abs(ndays)
        while ndays > 0:
            current_date += datetime.timedelta(days=sign * 1)
            if current_date.weekday() < 5:
                ndays -= 1

    return current_date if not is_timestamp else pd.Timestamp(current_date)


def prettify_ax(ax, title="", center=False, percentage=False, log_scale=False, start_date=config.start_date, end_date=config.end_date):
    """Makes matplotlib.pyplot.Axes look pretty

    Parameters:
        ax : Axes
        title : str
        center : bool
            x axis in center of graph instead of bottom
        start_date : date, optional
        end_date : date, optional
    """

    if title != "":
        ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend(loc="upper left", fontsize="small", labelspacing=0.2)
    '''
    matplotlib.rcParams["legend.loc"] = "upper left"  # matplotlib.rc("legend", loc="upper left")
    matplotlib.pyplot.rcParams["legend.fontsize"] = "small"  # matplotlib.pyplot.rc("legend", fontsize="small")
    matplotlib.pyplot.rcParams["legend.labelspacing"] = 0.2  # matplotlib.pyplot.rc("legend", labelspacing=0.2)
    '''

    # ax.spines["top"].set_visible(False)
    # ax.spines["right"].set_visible(False)

    ax.xaxis.set_major_locator(YearLocator())
    ax.xaxis.set_minor_locator(MonthLocator((1, 7)))
    ax.xaxis.set_major_formatter(DateFormatter("\n%Y"))
    ax.xaxis.set_minor_formatter(DateFormatter("%b"))

    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    # ax.tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="on", left="off", right="off", labelleft="on")  # Remove the tick marks

    # TODO: what causes smooshing
    # ax.set_xlim(start_date, end_date)  # this stops smooshing
    ax.margins(x=0)

    min_x = ax.get_xlim()[0].astype(int)
    max_x = ax.get_xlim()[1].astype(int)

    if not center and ax.get_ylim()[0] < 0:
        ax.set_ylim(ymin=0)
    if center:
        min_y = ax.get_ylim()[0].astype(float)  # should this be int?
        max_y = ax.get_ylim()[1].astype(float)  # should this be int?
        if abs(min_y) != abs(max_y):
            ax.set_ylim(ymin=-max(abs(min_y), abs(max_y)), ymax=max(abs(min_y), abs(max_y)))
        ax.plot(range(min_x, max_x), [0] * len(range(min_x, max_x)), "-", linewidth=0.5, color="black")
    if percentage:
        ax.set_ylim(ymin=0, ymax=100)
        # ax.yaxis.set_ticks(np.arange(0, 100, 10))
        # ax.set_yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

    if log_scale:
        ax.set_yscale("log")

    # TODO: do this for the dates too. Have to get locators, since there are too many ticks
    # Is this code below the same as this?: ax.xaxis.grid(color='gray', linestyle='dashed')
    # for y in range(ax.get_ylim()[0], ax.get_ylim()[1], 10):
    for i, y in enumerate(ax.get_yticks().astype(float)[1:-1]):
        ax.plot(range(min_x, max_x), [y] * len(range(min_x, max_x)), "--", linewidth=0.5, color="black", alpha=config.alpha)


def prettify_fig(fig, title=""):
    """Makes matplotlib.pyplot.Figure look pretty

    Parameters:
        fig : Figure
        title : str
    """

    # fig.autofmt_xdate()  # tilts dates
    fig.set_size_inches(config.figsize)  # currently I always set this when creating the fig
    fig.suptitle(title)
    fig.tight_layout()


# This class is to mute the annoying yfinance prints on errors.
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def debug(s, debug=False):
    if config.debug or debug:
        print(inspect.stack()[1].filename, flush=True)
        print("Func: " + inspect.stack()[1].function + " Line: " + str(inspect.stack()[1].lineno), flush=True)
        if isinstance(s, pd.DataFrame) or isinstance(s, pd.Series):
            print(s if config.verbose else s.head(), flush=True)
        elif isinstance(s, mpl.axes.Axes):
            # Currently I never debug axes instead of debugging figs
            pass
        elif isinstance(s, mpl.pyplot.Figure):
            mpl.pyplot.show(block=False)
            mpl.pyplot.pause(1)
            mpl.pyplot.close()
        else:
            print(s, flush=True)


class Mode(Enum):
    Long = 1
    Short = 2
    LongShort = 3

