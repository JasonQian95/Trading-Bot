import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import moving_average as ma
import prices
import sp500_wiki_scrapper as sp
import config
import utils
import tautils as ta

import datetime
from functools import lru_cache
from pandas_datareader._utils import RemoteDataError
from timeit import default_timer as timer


default_cash = 100000
default_bet_size = default_cash / 10
per_share_commission = "per_share"
minimum_commission = "minimum"
maximum_commission = "maximum"
default_commission = {per_share_commission: 0.01,
                      minimum_commission: 4.95,
                      maximum_commission: 9.95}
default_short_sell = False
default_soft_signals = False


simulation_table_filename = "SimulationResults.csv"
simulation_actions_only_table_filename = "ActionsOnly" + simulation_table_filename
simulation_graph_filename = "SimulationResults.png"

cash_column_name = "Cash"
portfolio_column_name = "Portofolio"
actions_column_name = "Actions"
portfolio_value_column_name = "PortfolioValue"
log_columns = [cash_column_name, portfolio_column_name, actions_column_name, portfolio_value_column_name]

download_data_time = "DownloadDataTime"
generate_signals_time = "GenerateSignalsTime"
read_signals_time = "ReadSignalsTime"
get_price_time = "GetPriceTime"
get_dividend_time = "GetDividendTime"
total_time = "TotalTime"
times = {download_data_time: 0.0,
         generate_signals_time: 0.0,
         read_signals_time: 0.0,
         get_price_time: 0.0,
         get_dividend_time: 0.0,
         total_time: 0.0}


class Simulation:

    def __init__(self, initial_cash=default_cash, symbols=["SPY"], benchmark=["SPY"],
                 bet_size=default_bet_size, commission=default_commission, short_sell=default_short_sell, soft_signals=default_soft_signals,
                 filename="", fail_gracefully=False, refresh=False, start_date=config.start_date, end_date=config.end_date):
        self.cash = initial_cash
        self.initial_cash = initial_cash
        self.symbols = symbols
        self.benchmark = benchmark
        self.bet_size = initial_cash // min(bet_size, (len(symbols) // 5) + 1)
        self.commission = commission
        self.short_sell = short_sell
        self.soft_signals = soft_signals
        self.filename = filename
        self.fail_gracefully = fail_gracefully
        self.refresh = refresh
        self.start_date = start_date
        self.end_date = end_date

        self.portfolio = {}
        self.signal_files = {}
        self.price_files = {}
        self.times = times
        self.total_dividends = 0
        self.total_commissions = 0
        self.dates = pd.read_csv(utils.get_file_path(config.simulation_data_path, "Dates.csv"), index_col="Date", parse_dates=["Date"])[self.start_date:self.end_date].index  # Probably faster to do .index before [start_date:end_date]
        self.log = pd.DataFrame(index=self.dates, columns=log_columns)
        self.log[actions_column_name] = ""

        self.run()

    def buy(self, symbol, date, amount, partial_shares=False):
        if self.cash < amount:
            # TODO: current behavior can lead to 'races' between stocks
            amount = self.cash
        if symbol not in self.portfolio:
            # TODO: buy at close price, or next day's open price?
            shares = (amount // self.get_price_on_date(symbol, date, time="Close")) if not partial_shares else (amount / self.get_price_on_date(symbol, date, time="Close"))
            if shares != 0:
                self.portfolio[symbol] = shares
                self.cash -= shares * self.get_price_on_date(symbol, date, time="Close")
                self.cash -= self.calculate_commission(shares)
                self.log.loc[date][actions_column_name] = self.log.loc[date][actions_column_name] + "{} {} {}  Shares at {} totaling {}".format(ta.buy_signal, symbol, shares, self.get_price_on_date(symbol, date), shares * self.get_price_on_date(symbol, date))
        else:
            # buy more
            pass

    def sell(self, symbol, date, amount, partial_shares=False, short_sell=False):
        if symbol in self.portfolio:
            # TODO: sell at close price, or next day's open price?
            if amount == 0:  # sell all
                self.cash += self.portfolio[symbol] * self.get_price_on_date(symbol, date)
                self.cash -= self.calculate_commission(self.portfolio[symbol])
                self.log.loc[date][actions_column_name] = self.log.loc[date][actions_column_name] + "{} {} {}  Shares at {} totalling {}".format(ta.sell_signal, symbol, self.portfolio[symbol], self.get_price_on_date(symbol, date), self.portfolio[symbol] * self.get_price_on_date(symbol, date))
                self.portfolio.pop(symbol)
            else:
                shares = (amount // self.get_price_on_date(symbol, date)) if not partial_shares else (amount / self.get_price_on_date(symbol, date))
                shares = shares if shares < self.portfolio[symbol] else self.portfolio[symbol]
                if shares != 0:
                    self.cash += shares * self.get_price_on_date(symbol, date)
                    self.cash -= self.calculate_commission(shares)
                    self.log.loc[date][actions_column_name] = self.log.loc[date][actions_column_name] + "{} {} {}  Shares at {} totalling {}".format(ta.sell_signal, symbol, shares, self.get_price_on_date(symbol, date), shares * self.get_price_on_date(symbol, date))
                    self.portfolio[symbol] -= shares
                    if self.portfolio[symbol] <= 0:
                        self.portfolio.pop(symbol)
        else:
            if self.short_sell:
                pass

    def calculate_commission(self, shares, func="stock"):
        if shares * self.commission[per_share_commission] < self.commission[minimum_commission]:
            self.commision += self.commission[minimum_commission]
            return self.commission[minimum_commission]
        if shares * self.commission[per_share_commission] > self.commission[maximum_commission]:
            self.commision += self.commission[maximum_commission]
            return self.commission[maximum_commission]
        self.commision += shares * self.commission[per_share_commission]
        return shares * self.commission[per_share_commission]

    @lru_cache(maxsize=128)
    def get_price_on_date(self, symbol, date, time="Close"):
        start_time = timer()

        if symbol in self.price_files:
            df = self.price_files[symbol]
        else:
            if utils.refresh(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=symbol), refresh=False):
                prices.download_data_from_yahoo(symbol, backfill=False, start_date=self.start_date, end_date=self.end_date)
            df = pd.read_csv(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=symbol), index_col="Date", parse_dates=["Date"])[self.start_date:self.end_date]
            self.price_files[symbol] = df

        if date in df.index:
            price = round(df.loc[date][time], 2)
        else:
            print(symbol + " " + str(date))
            price = self.get_price_on_date(symbol, pd.Timestamp(date.date() - datetime.timedelta(days=1)), time=time)

        self.times[get_price_time] = self.times[get_price_time] + timer() - start_time
        return price

    def get_dividends(self, symbol, date):
        start_time = timer()

        if symbol in self.price_files:
            df = self.price_files[symbol]
        else:
            if utils.refresh(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=symbol), refresh=False):
                prices.download_data_from_yahoo(symbol, backfill=False, start_date=self.start_date, end_date=self.end_date)
            df = pd.read_csv(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=symbol), index_col="Date", parse_dates=["Date"])[self.start_date:self.end_date]
            self.price_files[symbol] = df

        dividend = self.portfolio[symbol] * df.loc[date]["Dividends"] if date in df.index and "Dividends" in df.columns else 0
        self.cash += dividend
        self.total_dividends += dividend
        if dividend != 0:
            self.log.loc[date][actions_column_name] = self.log.loc[date][actions_column_name] + "Dividend: {} {} Shares totaling {} ".format(symbol, self.portfolio[symbol], dividend)

        self.times[get_dividend_time] = self.times[get_dividend_time] + timer() - start_time
        return dividend

    def total_value(self, date):
        total_value = self.cash
        for position in self.portfolio:
            total_value += self.portfolio[position] * self.get_price_on_date(position, date, time="Close")
        return total_value

    def generate_signals(self, symbols=None, start_date=None, end_date=None):
        start_time = timer()

        if symbols is None:
            symbols = self.symbols
        if isinstance(symbols, str):
            symbols = [symbols]
        if start_date is None:
            start_date = self.start_date
        if end_date is None:
            end_date = self.end_date

        for symbol in symbols:
            # TODO: strategy is hardcoded
            print("Generating signals for " + symbol, flush=True)
            try:
                ma.generate_signals(symbol, func=ma.ema_name, period=[50, 200], refresh=self.refresh, start_date=start_date, end_date=end_date)
            except ta.InsufficientDataException:
                print("Insufficient data for " + symbol)
                self.symbols.remove(symbol)
        self.times[generate_signals_time] = self.times[generate_signals_time] + timer() - start_time

    def read_signal(self, symbol, date):
        start_time = timer()

        if symbol in self.signal_files:
            df = self.signal_files[symbol]
        else:
            df = pd.read_csv(utils.get_file_path(config.ta_data_path, ma.ma_table_filename, symbol=symbol), index_col="Date", parse_dates=["Date"])[self.start_date:self.end_date]
            self.signal_files[symbol] = df

        signal = df.loc[date][ta.signal_name] if date in df.index else ta.default_signal
        self.times[read_signals_time] = self.times[read_signals_time] + timer() - start_time
        return signal

    def plot_against_benchmark(self, log, benchmark=None):
        if benchmark is None:
            benchmark = self.benchmark

        fig, ax = plt.subplots(2, 2, figsize=config.figsize)

        # Simulation performance
        ax[0][0].plot(log.index, log[portfolio_value_column_name], label=portfolio_value_column_name)
        utils.prettify_ax(ax[0][0], title=portfolio_value_column_name, start_date=self.start_date, end_date=self.end_date)

        # Benchmark performance
        for bench in benchmark:
            df = pd.read_csv(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=bench), index_col="Date", parse_dates=["Date"])[self.start_date:self.end_date]
            ax[1][0].plot(df.index, df["Close"], label=bench + "Price")
        utils.prettify_ax(ax[1][0], title="Benchmarks", start_date=self.start_date, end_date=self.end_date)

        # Simulation compared to available symbols
        if len(self.symbols) < 4:
            for symbol in self.symbols:
                df = pd.read_csv(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=symbol), index_col="Date", parse_dates=["Date"])[self.start_date:self.end_date]
                # ax[0][1].plot(df.index, (log[log.index in df.index][total_value_column_name] / log[log.index in df.index][cash_column_name][0]) / (df["Close"] / df["Close"][0]), label="Portfolio/" + symbol)
                ax[0][1].plot(df.index, df["Close"] / df["Close"][0])
            ax[0][1].plot(log.index, log[portfolio_value_column_name] / log[portfolio_value_column_name][0], label=portfolio_value_column_name)
            utils.prettify_ax(ax[0][1], title="PortfolioVSSymbols", start_date=self.start_date, end_date=self.end_date)

        # Simulation compared to benchmarks
        for bench in benchmark:
            df = pd.read_csv(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=bench), index_col="Date", parse_dates=["Date"])[self.start_date:self.end_date]
            ax[1][1].plot(df.index, (log[portfolio_value_column_name] / log[cash_column_name][0]) / (df["Close"] / df["Close"][0]), label="Portfolio/" + bench)
        utils.prettify_ax(ax[1][1], title="PortfolioVSBenchmarks", start_date=self.start_date, end_date=self.end_date)

        utils.prettify_fig(fig)
        fig.savefig(utils.get_file_path(config.simulation_graphs_path, simulation_graph_filename, symbol=self.symbols[0] if len(self.symbols) == 1 else ""))
        utils.debug(fig)

    def run(self):
        start_time = timer()

        for symbol in self.symbols:
            try:
                if utils.refresh(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=symbol), refresh=self.refresh):
                    prices.download_data_from_yahoo(symbol, backfill=False, start_date=self.start_date, end_date=self.end_date)
            except RemoteDataError:
                print("Invalid symbol: " + symbol)
                self.symbols.remove(symbol)
        for symbol in self.benchmark:
            try:
                if utils.refresh(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=symbol), refresh=self.refresh):
                    prices.download_data_from_yahoo(symbol, backfill=False, start_date=self.start_date, end_date=self.end_date)
            except RemoteDataError:
                print("Invalid symbol: " + symbol)
                self.symbols.remove(symbol)
        self.times[download_data_time] = self.times[download_data_time] + timer() - start_time

        self.generate_signals()

        self.log.loc[self.dates[0]][cash_column_name, portfolio_column_name, actions_column_name, portfolio_value_column_name] = [self.cash, self.portfolio, "Initial", self.total_value(self.dates[0])]

        try:
            for date in self.dates:
                print(date, flush=True)
                for symbol in self.portfolio:
                    self.get_dividends(symbol, date)
                for symbol in self.symbols:
                    signal = self.read_signal(symbol, date)
                    # unsure if I want to enforce self.cash > self.bet_size or not
                    if signal == ta.buy_signal and self.cash > self.bet_size:
                        self.buy(symbol, date, self.bet_size)
                        self.log.loc[date][cash_column_name, portfolio_column_name, portfolio_value_column_name] = [self.cash, self.portfolio, self.total_value(date)]
                    if signal == ta.sell_signal:
                        self.sell(symbol, date, amount=0)
                        self.log.loc[date][cash_column_name, portfolio_column_name, portfolio_value_column_name] = [self.cash,self.portfolio, self.total_value(date)]
                    if signal == ta.soft_sell_signal and self.soft_signals:
                        self.buy(symbol, date, self.bet_size)
                        self.log.loc[date][cash_column_name, portfolio_column_name, portfolio_value_column_name] = [self.cash, self.portfolio, self.total_value(date)]
                    if signal == ta.soft_sell_signal and self.soft_signals and self.short_sell:
                        self.sell(symbol, date, self.bet_size)
                        self.log.loc[date][cash_column_name, portfolio_column_name, portfolio_value_column_name] = [self.cash, self.portfolio, self.total_value(date)]

                self.log.loc[date][cash_column_name, portfolio_column_name, portfolio_value_column_name] = [self.cash, self.portfolio, self.total_value(date)]
        except (AttributeError, KeyError) as e:
            if self.fail_gracefully:
                print(e)
                self.log = self.log.loc[self.log[self.log.index] < date]
                # self.log = self.log[:self.log.index.get_loc(date) - 1]
            else:
                raise

        self.plot_against_benchmark(self.log, self.benchmark)
        self.log.loc[date][cash_column_name, portfolio_column_name, actions_column_name, portfolio_value_column_name] = [self.cash, self.portfolio, "Final", self.total_value(date)]
        self.log.to_csv(utils.get_file_path(config.simulation_data_path, simulation_table_filename, symbol=self.symbols[0] if len(self.symbols) == 1 else ""))
        self.log = self.log.loc[self.log[actions_column_name] != ""]  # self.log.dropna(subset=[actions_column_name], inplace=True)
        self.log.to_csv(utils.get_file_path(config.simulation_data_path, simulation_actions_only_table_filename, symbol=self.symbols[0] if len(self.symbols) == 1 else self.filename))

        self.times[total_time] = self.times[total_time] + timer() - start_time
        print(self.times)
        print(self.get_price_on_date.cache_info())
        print(self.total_dividends)
        print(self.total_commissions)


if __name__ == '__main__':
    # sim = Simulation(symbols=["SPY"], soft_signals=True)  # , soft_signals=True
    # sim = Simulation(symbols=["AAPL"], soft_signals=True)
    Simulation(symbols=sp.get_sp500(), refresh=False, soft_signals=True)

# ["ABT", "ATVI", "AMD", "GOOG", "AMZN", "AAL", "AAPL", "T", "BAC", "BA", "CSCO", "KO", "ED", "COST", "CVS", "DAL", "DLR", "DFS", "DLTR", "DUK", "DRE", "EBAY", "EA", "EXPE", "XOM", "FB", "F", "FOX", "GE", "GM", "GILD", "GS", "HRB", "HOG", "HAS", "HP", "INTC", "IBM", "MA", "MCD", "MGM", "MSFT", "NFLX", "NVDA", "OXY", "PYPL", "PG", "QCOM", "O", "RCL", "CRM", "LUV", "TGT", "TWTR", "UAL", "UPS", "V", "WMT", "DIS", "WFC"]

# TODO Thread this
# TODO pass in the function used to generate signals instead of hardcoding
# TODO plot rate of change of portfolio vs benchmark
# number of shares to buy is calculated before commission. Commission can then push cash into negatives
# include return percentage vs benchmark return percentage in log
