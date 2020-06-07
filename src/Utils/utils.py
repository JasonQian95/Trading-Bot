import matplotlib as mpl
from matplotlib.dates import YearLocator, MonthLocator, DateFormatter
import pandas as pd

import config

import inspect
import os
from os.path import join, exists


def backfill(df):
    """Fills in the null values in the dataframe. The fill method is "ffill"
    Parameters:
        df : dataframe, series
    """

    debug(df.isna().sum())
    debug(df[df.isna().any(axis=1)])
    df.fillna(method="ffill", inplace=True)


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
    start_date = (start_date.replace("-", "_") if isinstance(start_date, str) else start_date.strftime(config.date_format))
    end_date = (end_date.replace("-", "_") if isinstance(end_date, str) else end_date.strftime(config.date_format))
    return join(path, symbol + ((start_date + "-" + end_date) if dated else "") + filename)


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
        os.remove(path)
    return not exists(path) or refresh


def prettify_ax(ax, title="", center=False, start_date=config.start_date, end_date=config.end_date):
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
        min_y = ax.get_ylim()[0].astype(float)
        max_y = ax.get_ylim()[1].astype(float)
        if abs(min_y) != abs(max_y):
            ax.set_ylim(ymin=-max(abs(min_y), abs(max_y)), ymax=max(abs(min_y), abs(max_y)))
        ax.plot(range(min_x, max_x), [0] * len(range(min_x, max_x)), "-", linewidth=0.5, color="black")

    # TODO: do this for the dates too. Have to get locators, since there are too many ticks
    # for y in range(ax.get_ylim()[0], ax.get_ylim()[1], 10):
    for i, y in enumerate(ax.get_yticks().astype(float)[1:-1]):
        ax.plot(range(min_x, max_x), [y] * len(range(min_x, max_x)), "--", linewidth=0.5, color="black", alpha=config.alpha)


def prettify_fig(fig, title="", start_date=config.start_date, end_date=config.end_date):
    # fig.autofmt_xdate()  # tilts dates
    # fig.set_size_inches(config.figsize)  # currently I always set this when creating the fig
    # fig.legend(loc="best")
    fig.tight_layout()


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
