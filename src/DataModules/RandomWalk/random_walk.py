import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import prices

import config
import utils

prob = [0.49, 0.51]
dates_table_filename = "Dates.csv"
table_filename = prices.price_table_filename
graph_filename = "RandomWalk.png"


def random_walk():
    df = pd.read_csv(utils.get_file_path(config.random_walk_data_path, dates_table_filename), index_col="Date", parse_dates=["Date"])

    start = 100
    positions = [start]

    rr = np.random.random(len(df.index) - 1)
    downp = rr < prob[0]
    upp = rr > prob[1]

    for idownp, iupp in zip(downp, upp):
        down = idownp and positions[-1] > 1
        up = iupp and positions[-1] < 200
        positions.append(positions[-1] - down + up)

    df["Close"] = positions
    df["Dividends"] = 0
    df.to_csv(utils.get_file_path(config.random_walk_data_path, table_filename, symbol="RandomWalk"))

    fig, ax = plt.subplots(figsize=config.figsize)
    ax.plot(df.index, df["Close"], label="Price")
    utils.prettify_ax(ax, title="RandomWalk", start_date=df.index[0], end_date=df.index[-1])
    fig.savefig(utils.get_file_path(config.random_walk_graphs_path, graph_filename))
    utils.debug(fig)
    return df


def get_num_increase_decrease(symbol, refresh=False, start_date=config.start_date, end_date=config.end_date):
    if utils.refresh(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=symbol), refresh=refresh):
        prices.download_data_from_yahoo(symbol, start_date=start_date, end_date=end_date)
    df = pd.read_csv(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=symbol), index_col="Date", parse_dates=["Date"])[start_date:end_date]

    return {"Increase": (df["Close"].diff() > 0).sum(), "Decrease": (df["Close"].diff() < 0).sum()}


def get_num_conseq_increase_decrease(symbol, refresh=False, start_date=config.start_date, end_date=config.end_date):
    if utils.refresh(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=symbol), refresh=refresh):
        prices.download_data_from_yahoo(symbol, start_date=start_date, end_date=end_date)
    df = pd.read_csv(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=symbol), index_col="Date", parse_dates=["Date"])[start_date:end_date]

    return {"ConseqIncrease": ((df["Close"].diff() > 0) & (df["Close"].diff().shift(1) > 0)).sum(),
            "ConseqDecrease": ((df["Close"].diff() < 0) & (df["Close"].diff().shift(1) < 0)).sum(),
            "Reversal": ((df["Close"].diff() < 0) & (df["Close"].diff().shift(1) > 0)).sum() + ((df["Close"].diff() > 0) & (df["Close"].diff().shift(1) < 0)).sum()}


def get_longest_conseq_increase_decrease(symbol, refresh=False, start_date=config.start_date, end_date=config.end_date):
    if utils.refresh(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=symbol), refresh=refresh):
        prices.download_data_from_yahoo(symbol, start_date=start_date, end_date=end_date)
    df = pd.read_csv(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=symbol), index_col="Date", parse_dates=["Date"])[start_date:end_date]

    df["Diff"] = df["Close"].diff()
    df["Shift"] = df["Diff"].shift()
    longest_counts = {"LongestConseqIncrease": 0, "LongestConseqDecrease": 0}
    increasing = False
    count = 0

    for index, row in df.iterrows():
        if row["Diff"] > 0 and row["Shift"] > 0 and increasing == True:
            count += 1
            if count > longest_counts["LongestConseqIncrease"]:
                longest_counts["LongestConseqIncrease"] = count
        if row["Diff"] > 0 and row["Shift"] > 0 and increasing == False:
            increasing = True
            count = 1
        elif row["Diff"] < 0 and row["Shift"] < 0 and increasing == False:
            count += 1
            if count > longest_counts["LongestConseqDecrease"]:
                longest_counts["LongestConseqDecrease"] = count
        elif row["Diff"] < 0 and row["Shift"] < 0 and increasing == True:
            increasing = False
            count = 1

    return longest_counts
