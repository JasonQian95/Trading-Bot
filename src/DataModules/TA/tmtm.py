import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import prices
import config
import utils
import sp500_wiki_scrapper as sp

from collections import OrderedDict
import datetime
from enum import Enum
from functools import lru_cache
from timeit import default_timer as timer

# Today's move = Tmrw's move
table_filename = "TMTM.csv"
graph_filename = ".png"

dates_table_filename = "Dates.csv"
dates = pd.read_csv(utils.get_file_path(config.simulation_data_path, "Dates.csv"), index_col="Date", parse_dates=["Date"])[config.start_date:config.end_date].index

price_files = {}
initial_balance = 100


class Time(Enum):
    Open = 1
    Close = 2
    Gap = 3
    All = 4


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
    # df["Portfolio Value"][0] is blank, change to df["Portfolio Value"][2]?
    return (df["Portfolio Value"][-1] / df["Portfolio Value"][0]) ** (1 / ((df.index[-1] - df.index[0]).days / 365)) # 252 trading days


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


def tmtm(mode=utils.Mode.Long, refresh=False, start_date=config.start_date, end_date=config.end_date):
    dates = pd.read_csv(utils.get_file_path(config.simulation_data_path, dates_table_filename), index_col="Date", parse_dates=["Date"])[start_date:end_date].index
    sp500 = sp.get_sp500()
    strat_names = []
    if mode == utils.Mode.Long or mode == utils.Mode.LongShort:
        strat_names += ["MomentumBuyOnOpen", "MomentumBuyOnClose", "MomentumBuyOnGap", "MeanReversionBuyOnOpen", "MeanReversionBuyOnClose", "MeanReversionBuyOnGap"]
    if mode == utils.Mode.Short or mode == utils.Mode.LongShort:
        strat_names += ["MomentumShortOnOpen", "MomentumShortOnClose", "MomentumShortOnGap", "MeanReversionShortOnOpen", "MeanReversionShortOnClose", "MeanReversionShortOnGap"]
    num_strats = len(strat_names)
    dfs = [pd.DataFrame()] * num_strats
    balances = [initial_balance] * num_strats
    symbol_counts = [{} for i in range(num_strats)] #[{}, {}, {}, {}, {}, {}]  # * num_strats; this results in all the dicts being a copy of a single dict

    if not refresh:
        for i in range(num_strats):
            dfs[i] = pd.read_csv(utils.get_file_path(config.prices_data_path, "TMTM" + strat_names[i] + ".csv", symbol=""), index_col="Date", parse_dates=["Date"])[start_date:end_date]
            balances[i] = dfs[i]["Portfolio Value"][-1]
            symbol_counts[i] = dict(eval(dfs[i]["SymbolCount"][-1]))
    else:
        for i in range(num_strats):
            dfs[i] = pd.DataFrame(columns=["Portfolio Value", "Symbol", "Move", "PrevMove", "SymbolCount", "Stats"], index=dates)

        for date in dates[0:2]:
            for i in range(num_strats):
                # to make cagr work
                dfs[i]["Portfolio Value"][date] = balances[i]
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
            biggest_movers_next = []
            if mode == utils.Mode.Long or mode == utils.Mode.LongShort:
                biggest_movers_next += [biggest_gainer_next_buy_on_open, biggest_gainer_next_buy_on_close, biggest_gainer_next_buy_gap, biggest_loser_next_buy_on_open, biggest_loser_next_buy_on_close, biggest_loser_next_buy_gap]
            if mode == utils.Mode.Short or mode == utils.Mode.LongShort:
                biggest_movers_next += [biggest_gainer_next_short_on_open, biggest_gainer_next_short_on_close, biggest_gainer_next_short_gap, biggest_loser_next_short_on_open, biggest_loser_next_short_on_close, biggest_loser_next_short_gap]

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

        benchmark = pd.read_csv(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol="SPY"), index_col="Date", parse_dates=["Date"])[start_date:end_date]
        for i in range(num_strats):
            print(strat_names[i])
            print("Performance: " + "{:.2f}".format(balances[i]))
            print("CGAR: " + "{:.2f}".format(get_cgar(dfs[i])))
            print("Sharpe: " + "{:.2f}".format(get_sharpe_ratio(dfs[i])))
            print("Beta: " + "{:.2f}".format(get_beta(dfs[i], benchmark)))
            print("MaxDrawdown: " + "{:.2f}".format(get_max_drawdown(dfs[i])))
            # gives a warning due to null values
            print("Winners/Losers: " + "{:.2f}".format(dfs[i][dfs[i]["Move"] > 1.0]["Move"].count() / dfs[i][dfs[i]["Move"] < 1.0]["Move"].count()))
            # print(OrderedDict(sorted(symbol_counts[i].items(), key=lambda x: x[1], reverse=True))) #print(dfs[i]["SymbolCount"][dfs[i].index[-1]])
            print()

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
    if mode == utils.Mode.Long:
        ax[1].legend(loc="lower left", fontsize="small", labelspacing=0.2)

    utils.prettify_fig(fig)
    fig.savefig(utils.get_file_path(config.ta_graphs_path, graph_filename, symbol="TMTMTest" + "Long" if mode == utils.Mode.Long else "Short"))
    utils.debug(fig)


def tmtm_multi_strat_sma(periods=[1, 5, 10, 20, 30], minimum_avg=0, mode=utils.Mode.Long, time=Time.All, refresh=False, start_date=config.start_date, end_date=config.end_date):
    dates = pd.read_csv(utils.get_file_path(config.simulation_data_path, dates_table_filename), index_col="Date", parse_dates=["Date"])[start_date:end_date].index
    base_strat_names = []
    if mode == utils.Mode.Long or mode == utils.Mode.LongShort:
        if time == Time.Open:
            base_strat_names += ["MomentumBuyOnOpen", "MeanReversionBuyOnOpen"]
        elif time == Time.Close:
            base_strat_names += ["MomentumBuyOnClose", "MeanReversionBuyOnClose"]
        elif time == Time.Gap:
            base_strat_names += ["MomentumBuyOnGap", "MeanReversionBuyOnGap"]
        elif time == Time.All:
            base_strat_names += ["MomentumBuyOnOpen", "MomentumBuyOnClose", "MomentumBuyOnGap", "MeanReversionBuyOnOpen", "MeanReversionBuyOnClose", "MeanReversionBuyOnGap"]
    if mode == utils.Mode.Short or mode == utils.Mode.LongShort:
        if time == Time.Open:
            base_strat_names += ["MomentumShortOnOpen", "MeanReversionShortOnOpen"]
        elif time == Time.Close:
            base_strat_names += ["MomentumShortOnClose", "MeanReversionShortOnClose"]
        elif time == Time.Gap:
            base_strat_names += ["MomentumShortOnGap", "MeanReversionShortOnGap"]
        elif time == Time.All:
            base_strat_names += ["MomentumShortOnOpen", "MomentumShortOnClose", "MomentumShortOnGap", "MeanReversionShortOnOpen", "MeanReversionShortOnClose", "MeanReversionShortOnGap"]
    num_base_strats = len(base_strat_names)
    base_dfs = [pd.DataFrame()] * num_base_strats

    strat_names = [mode.name + "On" + time.name + "SMA" + str(period) for period in periods]
    num_strats = len(strat_names)
    dfs = [pd.DataFrame()] * num_strats
    balances = [initial_balance] * num_strats
    strat_counts = [{} for i in range(num_strats)] #[{}, {}, {}, {}, {}]  # * num_strats; this results in all the dicts being a copy of a single dict
    symbol_counts = [{} for i in range(num_strats)] #[{}, {}, {}, {}, {}]  # * num_strats; this results in all the dicts being a copy of a single dict

    if not refresh:
        for i in range(num_strats):
            dfs[i] = pd.read_csv(utils.get_file_path(config.prices_data_path, "TMTM" + strat_names[i] + ".csv", symbol=""), index_col="Date", parse_dates=["Date"])[start_date:end_date]
    else:
        for i in range(num_base_strats):
            #tmtm(refresh=refresh, mode=mode, start_date=start_date, end_date=end_date)
            base_dfs[i] = pd.read_csv(utils.get_file_path(config.prices_data_path, "TMTM" + base_strat_names[i] + ".csv", symbol=""), index_col="Date", parse_dates=["Date"])[start_date:end_date]

        for i in range(num_strats):
            dfs[i] = pd.DataFrame(columns=["Portfolio Value", "Move", "Strat", "StratCount", "Symbol", "SymbolCount", "Stats"], index=dates)

        for i in range(num_base_strats):
            for period in periods:
                base_dfs[i]["SMA" + str(period)] = base_dfs[i]["Move"].rolling(period).mean()

        for i in range(num_strats):
            for date in dates[0:3]:
                # to make cagr work
                dfs[i]["Portfolio Value"][date] = balances[i]
            for date in dates[3:]: # + periods[i]
                best_recent_strat = 0
                best_recent_performance = 0
                for j in range(num_base_strats):
                    if base_dfs[j]["SMA" + str(periods[i])][add_trading_days(date, -1)] > best_recent_performance:
                        best_recent_strat = j
                        best_recent_performance = base_dfs[j]["SMA" + str(periods[i])][add_trading_days(date, -1)]

                if best_recent_performance > minimum_avg:
                    balances[i] *= base_dfs[best_recent_strat]["Move"][date]
                    dfs[i]["Portfolio Value"][date] = balances[i]
                    dfs[i]["Move"][date] = base_dfs[best_recent_strat]["Move"][date]
                    dfs[i]["Strat"][date] = base_strat_names[best_recent_strat]
                    strat_counts[i][base_strat_names[best_recent_strat]] = strat_counts[i].get(base_strat_names[best_recent_strat], 0) + 1
                    dfs[i]["StratCount"][date] = OrderedDict(sorted(strat_counts[i].items(), key=lambda x: x[1], reverse=True))
                    dfs[i]["Symbol"][date] = base_dfs[best_recent_strat]["Symbol"][date]
                    symbol_counts[i][base_dfs[best_recent_strat]["Symbol"][date]] = symbol_counts[i].get(base_dfs[best_recent_strat]["Symbol"][date], 0) + 1
                    # is this too slow? Seems ok
                    dfs[i]["SymbolCount"][date] = OrderedDict(sorted(symbol_counts[i].items(), key=lambda x: x[1], reverse=True))
                else:
                    balances[i] *= 1
                    dfs[i]["Portfolio Value"][date] = balances[i]
                    dfs[i]["Move"][date] = 1
                    dfs[i]["Strat"][date] = "None"
                    strat_counts[i]["None"] = strat_counts[i].get("None", 0) + 1
                    dfs[i]["StratCount"][date] = OrderedDict(sorted(strat_counts[i].items(), key=lambda x: x[1], reverse=True))
                    dfs[i]["Symbol"][date] = "None"
                    symbol_counts[i]["None"] = symbol_counts[i].get("None", 0) + 1
                    # is this too slow? Seems ok
                    dfs[i]["SymbolCount"][date] = OrderedDict(sorted(symbol_counts[i].items(), key=lambda x: x[1], reverse=True))

        for i in range(num_strats):
            benchmark = pd.read_csv(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol="SPY"), index_col="Date", parse_dates=["Date"])[start_date:end_date]
            vol = pd.read_csv(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol="^VIX"), index_col="Date", parse_dates=["Date"])[start_date:end_date]

            print(strat_names[i])
            print("Performance: " + "{:.2f}".format(balances[i]))
            print("CGAR: " + "{:.2f}".format(get_cgar(dfs[i])))
            print("Sharpe: " + "{:.2f}".format(get_sharpe_ratio(dfs[i])))
            print("Beta: " + "{:.2f}".format(get_beta(dfs[i], benchmark)))
            print("Beta to vol: " + "{:.2f}".format(get_beta(dfs[i], vol)))
            print("MaxDrawdown: " + "{:.2f}".format(get_max_drawdown(dfs[i])))
            # gives a warning due to null values
            print("Winners/Losers: " + "{:.2f}".format(dfs[i][dfs[i]["Move"] > 1.0]["Move"].count() / dfs[i][dfs[i]["Move"] < 1.0]["Move"].count()))
            print(OrderedDict(sorted(strat_counts[i].items(), key=lambda x: x[1], reverse=True)))
            # print(OrderedDict(sorted(symbol_counts[i].items(), key=lambda x: x[1], reverse=True))) #print(dfs[i]["SymbolCount"][dfs[i].index[-1]])
            print()

            dfs[i]["Stats"][-1] = "Final: Performance: {:.2f} CGAR {:.2f} Sharpe Ratio {:.2f} Beta: {:.2f} Max Drawdown {:.2f}% Winners/Losers {:.2f}".format(balances[i], get_cgar(dfs[i]), get_sharpe_ratio(dfs[i]), get_beta(dfs[i], benchmark), get_max_drawdown(dfs[i]), dfs[i][dfs[i]["Move"] > 1.0]["Move"].count() / dfs[i][dfs[i]["Move"] < 1.0]["Move"].count())
            dfs[i].to_csv(utils.get_file_path(config.ta_data_path, "TMTM" + strat_names[i] + ".csv", symbol=""))

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

    vol = pd.read_csv(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol="^VIX"), index_col="Date", parse_dates=["Date"])[start_date:end_date]
    ax[0].twinx().plot(vol.index, vol["Close"], label="Vol")
    ax[1].plot(vol.index, vol["Close"] * 100 / vol["Close"][0], label="Vol")

    utils.prettify_fig(fig)
    fig.savefig(utils.get_file_path(config.ta_graphs_path, graph_filename, symbol="TMTMTest" + mode.name + "On" + time.name + "SMA" + "-".join(str(p) for p in periods)))
    utils.debug(fig)


start_time = timer()
'''
tmtm(mode=utils.Mode.Long, refresh=False, start_date=datetime.date(2005, 1, 1), end_date=datetime.date(2020, 6, 1))
tmtm(mode=utils.Mode.Short, refresh=False, start_date=datetime.date(2005, 1, 1), end_date=datetime.date(2020, 6, 1))

for mode in utils.Mode:
    for time in Time:
        tmtm_multi_strat_sma(mode=mode, time=time, refresh=True, start_date=datetime.date(2005, 1, 1), end_date=datetime.date(2020, 6, 1))
'''
tmtm_multi_strat_sma(mode=utils.Mode.Long, time=Time.Gap, minimum_avg=1, refresh=True, start_date=datetime.date(2005, 1, 1), end_date=datetime.date(2020, 6, 1))

time = timer() - start_time
hours, rem = divmod(time, 3600)
minutes, seconds = divmod(rem, 60)
print("Total Time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
