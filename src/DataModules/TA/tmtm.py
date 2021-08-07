import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import prices
import config
import utils
import sp500_wiki_scrapper as sp

from collections import OrderedDict
import datetime
from functools import lru_cache
from timeit import default_timer as timer

# Today's move = Tmrw's move
table_filename = "TMTM.csv"
graph_filename = ".png"

dates_table_filename = "Dates.csv"
dates = pd.read_csv(utils.get_file_path(config.simulation_data_path, "Dates.csv"), index_col="Date", parse_dates=["Date"])[config.start_date:config.end_date].index

price_files = {}
initial_balance = 100


@lru_cache(maxsize=128)
def get_price_on_date(symbol, date, time="Close", start_date=config.start_date, end_date=config.end_date):
    if symbol in price_files:
        df = price_files[symbol]
    else:
        # below 3 lines commented out/modified for testing (moved price files to "\SP500AndOthers" for organization)
        #if utils.refresh(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=symbol), refresh=False):
        #    prices.download_data_from_yahoo(symbol, start_date=start_date, end_date=end_date)
        df = pd.read_csv(utils.get_file_path(config.prices_data_path + "\SP500AndOthers", prices.price_table_filename, symbol=symbol), index_col="Date", parse_dates=["Date"])[start_date:end_date]
        price_files[symbol] = df
    price = df[time][date] if date in df.index else get_price_on_date(symbol, add_trading_days(date, -1), time=time)
    return price


def is_date_in_bounds(symbol, date, start_date=config.start_date, end_date=config.end_date):
    if symbol in price_files:
        df = price_files[symbol]
    else:
        # below 3 lines commented out/modified for testing (moved price files to "\SP500AndOthers" for organization)
        #if utils.refresh(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=symbol), refresh=False):
        #    prices.download_data_from_yahoo(symbol, start_date=self.start_date, end_date=self.end_date)
        df = pd.read_csv(utils.get_file_path(config.prices_data_path + "\SP500AndOthers", prices.price_table_filename, symbol=symbol), index_col="Date", parse_dates=["Date"])[start_date:end_date]
        price_files[symbol] = df
    if df.index[0] <= date <= df.index[-1]:
        return True
    return False


@lru_cache(maxsize=128)
def add_trading_days(start_date, ndays):
    is_timestamp = False

    if isinstance(start_date, str):
        current_date = datetime.datetime.strptime(start_date, config.input_date_format).date()
    elif isinstance(start_date, pd.Timestamp):
        current_date = start_date.date()
        is_timestamp = True
    else:
        current_date = start_date

    if ndays != 0:
        # Use SPYPrice file instead if refresh support is required
        current_date = dates[dates.get_loc(start_date) + ndays]

    return current_date if not is_timestamp else pd.Timestamp(current_date)


def get_cgar(df, start_date=config.start_date, end_date=config.end_date):
    # Formula normally has a -1 at the end
    return (df["Portfolio Value"][-1] / df["Portfolio Value"][0]) ** (1 / (df.index[-1] - df.index[0]).days / 252)


def get_sharpe_ratio(df):
    return df["Portfolio Value"].pct_change().mean() / df["Portfolio Value"].pct_change().std() * np.sqrt(252)


def get_beta(df, bench):
    a = df["Portfolio Value"].pct_change()[3:]
    b = bench["Close"].pct_change()[3:] #.add(bench["Dividends"].cumsum()) dont have dividend data
    beta = np.cov(a, b)[0][1] / np.var(b)
    return beta


def get_beta_ports(dfa, dfb):
    a = dfa["Portfolio Value"].pct_change()[3:]
    b = dfb["Portfolio Value"].pct_change()[3:]
    beta = np.cov(a, b)[0][1] / np.var(b)
    return beta


def get_max_drawdown(df):
    rolling_max = df["Portfolio Value"].cummax()
    drawdown = df["Portfolio Value"] / rolling_max - 1.0
    return drawdown.cummin().min() * 100


def tmtm(refresh=False, mode=utils.Mode.long, start_date=config.start_date, end_date=config.end_date):
    dates = pd.read_csv(utils.get_file_path(config.simulation_data_path, dates_table_filename), index_col="Date", parse_dates=["Date"])[start_date:end_date].index
    sp500 = sp.get_sp500()
    if mode == utils.Mode.long:
        strat_names = ["MomentumBuyOnOpen", "MomentumBuyOnClose", "MomentumBuyGap", "MeanReversionBuyOnOpen", "MeanReversionBuyOnClose", "MeanReversionBuyGap"]
    else: #if mode == utils.Mode.short:
        strat_names = ["MomentumShortOnOpen", "MomentumShortOnClose", "MomentumShortGap", "MeanReversionShortOnOpen", "MeanReversionShortOnClose", "MeanReversionShortGap"]
    num_strats = len(strat_names)
    dfs = [pd.DataFrame()] * num_strats
    balances = [0] * num_strats
    symbol_counts = [{}, {}, {}, {}, {}, {}]  # * num_strats; this results in all the dicts being a copy of a single dict

    if not refresh:
        for i in range(num_strats):
            dfs[i] = pd.read_csv(utils.get_file_path(config.prices_data_path, "TMTM" + strat_names[i] + ".csv", symbol=""), index_col="Date", parse_dates=["Date"])[start_date:end_date]
            balances[i] = dfs[i]["Portfolio Value"][-1]
            symbol_counts[i] = dict(eval(dfs[i]["SymbolCount"][-1]))
    else:
        for i in range(num_strats):
            dfs[i] = pd.DataFrame(columns=["Portfolio Value", "Symbol", "Move", "PrevMove", "SymbolCount", "Stats"], index=dates)
            balances[i] = initial_balance

        for date in dates[2:]:
            # could turn these into lists to be able to rank
            biggest_gain = 0
            biggest_gainer = ""
            biggest_gainer_next_buy_on_open = 1
            biggest_gainer_next_buy_on_close = 1
            biggest_gainer_next_buy_gap = 1
            biggest_gainer_next_short_on_open = 1
            biggest_gainer_next_short_on_close = 1
            biggest_gainer_next_short_gap = 1
            biggest_lose = 2
            biggest_loser = ""
            biggest_loser_next_buy_on_open = 1
            biggest_loser_next_buy_on_close = 1
            biggest_loser_next_buy_gap = 1
            biggest_loser_next_short_on_open = 1
            biggest_loser_next_short_on_close = 1
            biggest_loser_next_short_gap = 1
            for symbol in sp500:
                if is_date_in_bounds(symbol, add_trading_days(date, -2), start_date, end_date):
                    price1 = get_price_on_date(symbol, add_trading_days(date, -2), "Close", start_date, end_date)
                    price2 = get_price_on_date(symbol, add_trading_days(date, -1), "Close", start_date, end_date)
                    price3 = get_price_on_date(symbol, date, "Open", start_date, end_date)
                    price4 = get_price_on_date(symbol, date, "Close", start_date, end_date)
                    if (price2 / price1) > biggest_gain:
                        biggest_gain = price2 / price1
                        biggest_gainer = symbol
                        biggest_gainer_next_buy_on_open = price4 / price3
                        biggest_gainer_next_buy_on_close = price4 / price2
                        biggest_gainer_next_buy_gap = price3 / price2
                        biggest_gainer_next_short_on_open = price3 / price4
                        biggest_gainer_next_short_on_close = price2 / price4
                        biggest_gainer_next_short_gap = price2 / price3
                    if (price2 / price1) < biggest_lose:
                        biggest_lose = price2 / price1
                        biggest_loser = symbol
                        biggest_loser_next_buy_on_open = price4 / price3
                        biggest_loser_next_buy_on_close = price4 / price2
                        biggest_loser_next_buy_gap = price3 / price2
                        biggest_loser_next_short_on_open = price3 / price4
                        biggest_loser_next_short_on_close = price2 / price4
                        biggest_loser_next_short_gap = price2 / price3
            # if biggest lose >= 100 (biggest loser gained) what do?

            biggest_moves = [biggest_gain, biggest_gain,  biggest_gain, biggest_lose, biggest_lose, biggest_lose]
            biggest_movers = [biggest_gainer, biggest_gainer, biggest_gainer, biggest_loser, biggest_loser, biggest_loser]
            if mode == utils.Mode.long:
                biggest_movers_next = [biggest_gainer_next_buy_on_open, biggest_gainer_next_buy_on_close, biggest_gainer_next_buy_gap, biggest_loser_next_buy_on_open, biggest_loser_next_buy_on_close, biggest_loser_next_buy_gap]
            else: #if mode == utils.Mode.short:
                biggest_movers_next = [biggest_gainer_next_short_on_open, biggest_gainer_next_short_on_close, biggest_gainer_next_short_gap, biggest_loser_next_short_on_open, biggest_loser_next_short_on_close, biggest_loser_next_short_gap]

            if '' in biggest_movers or biggest_gain == 1 or biggest_lose == 1:
                raise Exception()
            for i in range(num_strats):
                balances[i] *= biggest_movers_next[i]
                dfs[i]["Portfolio Value"][date] = balances[i]
                dfs[i]["Symbol"][date] = biggest_movers[i]
                dfs[i]["Move"][date] = biggest_movers_next[i]
                dfs[i]["PrevMove"][date] = biggest_moves[i]
                symbol_counts[i][biggest_movers[i]] = symbol_counts[i].get(biggest_movers[i], 0) + 1
                # is this too slow? Seems ok
                dfs[i]["SymbolCount"][date] = OrderedDict(sorted(symbol_counts[i].items(), key=lambda x: x[1], reverse=True))

        for i in range(num_strats):
            benchmark = pd.read_csv(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol="SPY"), index_col="Date", parse_dates=["Date"])[start_date:end_date]

            print(strat_names[i])
            print("Performance: " + "{:.2f}".format(balances[i]))
            print("CGAR: " + "{:.2f}".format(get_cgar(dfs[i])))
            print("Sharpe: " + "{:.2f}".format(get_sharpe_ratio(dfs[i])))
            print("Beta: " + "{:.2f}".format(get_beta(dfs[i], benchmark)))
            print("MaxDrawdown: " + "{:.2f}".format(get_max_drawdown(dfs[i])))
            # gives a warning due to null values
            print("Winners/Losers: " + "{:.2f}".format(dfs[i][dfs[i]["Move"] > 1.0]["Move"].count() / dfs[i][dfs[i]["Move"] < 1.0]["Move"].count()))
            # print(OrderedDict(sorted(symbol_counts[i].items(), key=lambda x: x[1], reverse=True))) #print(dfs[i]["SymbolCount"][dfs[i].index[-1]])

            dfs[i]["Stats"][-1] = "Final: Performance: {:.2f} CGAR {:.2f} Sharpe Ratio {:.2f} Beta: {:.2f} Max Drawdown {:.2f}% Winners/Losers {:.2f}".format(balances[i], get_cgar(dfs[i]), get_sharpe_ratio(dfs[i]), get_beta(dfs[i], benchmark), get_max_drawdown(dfs[i]), dfs[i][dfs[i]["Move"] > 1.0]["Move"].count() / dfs[i][dfs[i]["Move"] < 1.0]["Move"].count())
            dfs[i].to_csv(utils.get_file_path(config.ta_data_path, "TMTM" + strat_names[i] + ".csv", symbol=""))
        print("Beta of gap strats: " + "{:.2f}".format(get_beta_ports(dfs[2], dfs[5])))

    fig, ax = plt.subplots(2, figsize=config.figsize)
    benchmark = pd.read_csv(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol="SPY"), index_col="Date", parse_dates=["Date"])[start_date:end_date]
    for i in range(num_strats):
        ax[0].plot(dfs[i].index, dfs[i]["Portfolio Value"], label=strat_names[i])
    ax[0].plot(benchmark.index, benchmark["Close"] * initial_balance / benchmark["Close"][0], label="Benchmark")
    utils.prettify_ax(ax[0], title="TodaysMove=TommorrowsMove", start_date=start_date, end_date=config.end_date)

    for i in range(num_strats):
        ax[1].plot(dfs[i].index, dfs[i]["Portfolio Value"], label=strat_names[i])
    ax[1].plot(benchmark.index, benchmark["Close"] * initial_balance / benchmark["Close"][0], label="Benchmark")
    utils.prettify_ax(ax[1], title="TodaysMove=TommorrowsMove: LogScale", log_scale=True, start_date=start_date, end_date=end_date)
    # COMMENT OUT FOR SHORT
    #ax[1].legend(loc="lower left", fontsize="small", labelspacing=0.2)

    utils.prettify_fig(fig)
    fig.savefig(utils.get_file_path(config.ta_graphs_path, graph_filename, symbol="TMTM" + "A" if mode == utils.Mode.long else "B"))
    utils.debug(fig)


start_time = timer()
tmtm(refresh=True, start_date=datetime.date(2005, 1, 1), end_date=datetime.date(2020, 6, 1))
time = timer() - start_time
hours, rem = divmod(time, 3600)
minutes, seconds = divmod(rem, 60)
print("Total Time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
