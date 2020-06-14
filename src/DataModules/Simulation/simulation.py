import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import ema
import macd
import rsi
import sma

import prices
import sp500_wiki_scrapper as sp
import config
import utils
import tautils as ta

from collections import OrderedDict
import datetime
from functools import lru_cache
import inspect
import os
from pandas_datareader._utils import RemoteDataError
import psutil
from threading import Thread
from timeit import default_timer as timer


per_share_commission = "per_share"
minimum_commission = "minimum"
maximum_commission = "maximum"
default_commission = {per_share_commission: 0.01,
                      minimum_commission: 4.95,
                      maximum_commission: 9.95}

simulation_table_filename = "SimulationResults.csv"
simulation_actions_only_table_filename = "ActionsOnly" + simulation_table_filename
simulation_graph_filename = "SimulationResults.png"
dates_table_filename = "Dates.csv"

cash_column_name = "Cash"
portfolio_column_name = "Portofolio"
actions_column_name = "Actions"
portfolio_value_column_name = "PortfolioValue"
total_commission_column_name = "TotalCommission"
total_dividend_column_name = "TotalDividend"
log_columns = [cash_column_name, portfolio_column_name, actions_column_name, portfolio_value_column_name, total_commission_column_name, total_dividend_column_name]

download_data_time = "DownloadDataTime"
generate_signals_time = "GenerateSignalsTime"
read_signals_time = "ReadSignalsTime"
get_price_time = "GetPricesTime"
get_dividend_time = "GetDividendsTime"
total_time = "TotalTime"
times = {download_data_time: 0.0,
         generate_signals_time: 0.0,
         read_signals_time: 0.0,
         get_price_time: 0.0,
         get_dividend_time: 0.0,
         total_time: 0.0}


class Simulation:

    def __init__(self, filename="", initial_cash=100000, symbols=["SPY"], benchmark=["SPY"],
                 commission=default_commission, max_portfolio_size=100, short_sell=False, soft_signals=False, partial_shares=False,
                 fail_gracefully=False, refresh=False, start_date=config.start_date, end_date=config.end_date,
                 signal_func=None, signal_func_args=[], signal_func_kwargs={}):

        if len(symbols) == 0:
            raise ValueError("Requires at least one symbol")
        if signal_func is None:
            raise ValueError("Requires a signal function")

        self.filename = filename
        self.cash = initial_cash
        self.initial_cash = initial_cash
        self.symbols = symbols
        self.benchmark = benchmark
        self.commission = commission
        self.max_portfolio_size = max_portfolio_size
        self.purchase_size = (initial_cash // min(max_portfolio_size, len(symbols) // 5 + 1)) - (self.commission[maximum_commission] if maximum_commission in self.commission else 0)
        self.short_sell = short_sell
        self.soft_signals = soft_signals
        self.partial_shares = partial_shares
        self.fail_gracefully = fail_gracefully
        self.refresh = refresh
        self.start_date = start_date
        self.end_date = end_date
        self.signal_func = signal_func
        self.signal_func_args = signal_func_args
        self.signal_func_kwargs = signal_func_kwargs

        self.portfolio = {}
        self.signal_files = {}
        self.price_files = {}
        self.times = times
        self.total_dividends = 0
        self.total_commissions = 0
        self.total_trades = 0
        self.dates = pd.read_csv(utils.get_file_path(config.simulation_data_path, dates_table_filename), index_col="Date", parse_dates=["Date"])[self.start_date:self.end_date].index
        self.log = pd.DataFrame(index=self.dates, columns=log_columns)
        self.log[actions_column_name] = ""
        self.signal_table_filename = inspect.getmodule(self.signal_func).table_filename
        self.signal_name = inspect.getmodule(self.signal_func).get_signal_name(**signal_func_kwargs)
        # TODO make this work for multiple signals
        '''
        self.signal_names = []
        self.signal_table_filename = []
        for func in signal_func:
            self.signal_name.append(inspect.getmodule(func).signal_name)
            self.signal_table_filename.append(inspect.getmodule(func).table_filename)
        '''

        # self.get_price_on_date.cache_clear()
        self.run()

    def buy(self, symbol, date, purchase_size):
        """Simulates buying a stock

        Parameters:
            symbol : str
            date : datetime
            purchase_size : float, optional
                How much to buy. If 0, buy the default amount
            partial_shares : bool
                Whether partial shares are supported. If True, the amount bought will always equal amount, even if that number isn't reachable in a number of whole shares
        """

        if purchase_size == 0:
            purchase_size = self.purchase_size

        if self.cash < purchase_size:
            self.log.loc[date][actions_column_name] = self.log.loc[date][actions_column_name] + "Unable to buy {} ".format(symbol)
            # purchase_size = self.cash
        elif symbol not in self.portfolio:
            # TODO: buy at close price, or next day's open price?
            shares = (purchase_size // self.get_price_on_date(symbol, date, time="Close")) if not self.partial_shares else (purchase_size / self.get_price_on_date(symbol, date, time="Close"))
            if shares < 0:
                # Sometimes shares is -1. When variables are printed, math does not add up to -1??
                print("Symbol {} self.purchase_size {} Purchase size {} Price {} Shares {}".format(symbol, self.purchase_size, purchase_size, self.get_price_on_date(symbol, date, time="Close"), shares), flush=True)
                raise Exception
            if shares != 0:
                self.portfolio[symbol] = shares
                self.cash -= shares * self.get_price_on_date(symbol, date, time="Close")
                self.cash -= self.calculate_commission(shares)
                self.log.loc[date][actions_column_name] = self.log.loc[date][actions_column_name] + "{} {} {} Shares at {} totaling {:.2f} ".format(ta.buy_signal, symbol, shares, self.get_price_on_date(symbol, date), shares * self.get_price_on_date(symbol, date))
        else:
            if self.short_sell and self.portfolio[symbol] < 0:
                pass
            elif self.portfolio[symbol] > 0:
                # buy more
                pass

    def sell(self, symbol, date, sell_size=0, short_sell=False):
        """Simulates selling a stock

        Parameters:
            symbol : str
            date : datetime
            sell_size : float, optional
                How much to sell. If 0, sell all
            partial_shares : bool
                Whether partial shares are supported. If True, the amount sold will always equal amount, even if that number isn't reachable in a number of whole shares
            short_sell : bool
        """

        if symbol in self.portfolio:
            if self.portfolio[symbol] > 0:
                # TODO: sell at close price, or next day's open price?
                if sell_size == 0:  # sell all
                    self.cash += self.portfolio[symbol] * self.get_price_on_date(symbol, date)
                    self.cash -= self.calculate_commission(self.portfolio[symbol])
                    self.log.loc[date][actions_column_name] = self.log.loc[date][actions_column_name] + "{} {} {} Shares at {} totalling {:.2f} ".format(ta.sell_signal, symbol, self.portfolio[symbol], self.get_price_on_date(symbol, date), self.portfolio[symbol] * self.get_price_on_date(symbol, date))
                    self.portfolio.pop(symbol)
                else:
                    shares = (sell_size // self.get_price_on_date(symbol, date)) if not self.partial_shares else (sell_size / self.get_price_on_date(symbol, date))
                    shares = shares if shares < self.portfolio[symbol] else self.portfolio[symbol]
                    if shares < 0:
                        print("Amount {} Price {} Shares {}".format(sell_size, self.get_price_on_date(symbol, date, time="Close"), shares), flush=True)
                        raise Exception
                    if shares != 0:
                        self.cash += shares * self.get_price_on_date(symbol, date)
                        self.cash -= self.calculate_commission(shares)
                        self.log.loc[date][actions_column_name] = self.log.loc[date][actions_column_name] + "{} {} {} Shares at {} totalling {:.2f} ".format(ta.sell_signal, symbol, shares, self.get_price_on_date(symbol, date), shares * self.get_price_on_date(symbol, date))
                        self.portfolio[symbol] -= shares
                        if self.portfolio[symbol] < 0:
                            print("Amount {} Price {} Shares {} Shares in portfolio {}".format(sell_size, self.get_price_on_date(symbol, date, time="Close"), shares, self.portfolio[symbol]), flush=True)
                            raise Exception
                        if self.portfolio[symbol] == 0:
                            self.portfolio.pop(symbol)
                self.update_purchase_size(date)
            else:
                # short sell more
                pass
        else:
            if self.short_sell:
                pass

    def calculate_commission(self, shares):
        """Calculates the commissions to buy a stock using the previously set commission dictionary

        Parameters:
            shares : int

        Returns:
            float
                The commission required to buy a stock
        """

        self.total_trades += 1
        if shares * self.commission[per_share_commission] < self.commission[minimum_commission]:
            self.total_commissions += self.commission[minimum_commission]
            return self.commission[minimum_commission]
        if shares * self.commission[per_share_commission] > self.commission[maximum_commission]:
            self.total_commissions += self.commission[maximum_commission]
            return self.commission[maximum_commission]
        self.total_commissions += shares * self.commission[per_share_commission]
        return shares * self.commission[per_share_commission]

    @lru_cache(maxsize=128)
    def get_price_on_date(self, symbol, date, time="Close"):
        """Gets the price of the given symbol on the given date

        Parameters:
            symbol : str
            date : datetime
            time : str
                Which column to use to determine price. Valid times are "Open" and "Close"

        Returns:
            float
                The price of the given symbol on the given date
        """

        start_time = timer()

        if symbol in self.price_files:
            df = self.price_files[symbol]
        else:
            if utils.refresh(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=symbol), refresh=False):
                prices.download_data_from_yahoo(symbol, start_date=self.start_date, end_date=self.end_date)
            df = pd.read_csv(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=symbol), index_col="Date", parse_dates=["Date"])[self.start_date:self.end_date]
            self.price_files[symbol] = df

        price = df.loc[date][time] if date in df.index else self.get_price_on_date(symbol, pd.Timestamp(date.date() - datetime.timedelta(days=1)), time=time)

        self.times[get_price_time] = self.times[get_price_time] + timer() - start_time
        return price

    def get_dividends(self, symbol, date):
        """Adds dividends to the portfolio for the given symbol on the given date

        Parameters:
            symbol : str
            date : datetime

        Returns:
            float
                The dividends added
        """

        start_time = timer()

        if symbol in self.price_files:
            df = self.price_files[symbol]
        else:
            if utils.refresh(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=symbol), refresh=False):
                prices.download_data_from_yahoo(symbol, start_date=self.start_date, end_date=self.end_date)
            df = pd.read_csv(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=symbol), index_col="Date", parse_dates=["Date"])[self.start_date:self.end_date]
            self.price_files[symbol] = df

        dividend = self.portfolio[symbol] * df.loc[date]["Dividends"] if date in df.index and "Dividends" in df.columns else 0
        self.cash += dividend
        self.total_dividends += dividend
        if dividend != 0:
            self.log.loc[date][actions_column_name] = self.log.loc[date][actions_column_name] + "Dividend: {} {} Shares totaling {:.2f} ".format(symbol, self.portfolio[symbol], dividend)

        self.times[get_dividend_time] = self.times[get_dividend_time] + timer() - start_time
        return dividend

    def total_value(self, date):
        """Gets the total value of the portfolio on the given date

        Parameters:
            date : datetime

        Returns:
            float
                The total value of the portfolio on the given date
        """

        total_value = self.cash
        for position in self.portfolio:
            total_value += self.portfolio[position] * self.get_price_on_date(position, date, time="Close")
        return total_value

    def update_purchase_size(self, date):
        """Updates purchase size

        Parameters:
            date : datetime
            purchase_size : float, optional
        """

        self.purchase_size = (self.total_value(date) // min(self.max_portfolio_size, len(self.symbols) // 5 + 1)) - (self.commission[maximum_commission] if maximum_commission in self.commission else 0)

    def generate_signals(self):
        """Generates signals for all symbols
        """

        start_time = timer()

        for symbol in self.symbols:
            print("Generating signals for " + symbol, flush=True)
            try:
                self.signal_func(symbol, *self.signal_func_args, **self.signal_func_kwargs, refresh=self.refresh, start_date=self.start_date, end_date=self.end_date)
                # TODO make this work for multiple signal funcs, this will require some backfill
                '''
                for i, func in enumerate(signal_func):
                    func(symbol, *args[i], refresh=refresh, start_date=start_date, end_date=end_date, **kwargs[i])
                '''
            except ta.InsufficientDataException:
                print("Insufficient data for " + symbol)
                self.symbols.remove(symbol)
            except RemoteDataError:
                print("Invalid symbol: " + symbol)
                self.symbols.remove(symbol)

        self.times[generate_signals_time] = self.times[generate_signals_time] + timer() - start_time

    def read_signal(self, symbol, date):
        """Reads the signal file to determine whether to buy, sell, or hold

        Parameters:
            symbol : str
            date : datetime

        Returns:
            str
                The signal read
        """

        start_time = timer()

        if symbol in self.signal_files:
            df = self.signal_files[symbol]
        else:
            try:
                df = pd.read_csv(utils.get_file_path(config.ta_data_path, self.signal_table_filename, symbol=symbol), usecols=["Date", self.signal_name], index_col="Date", parse_dates=["Date"])[self.start_date:self.end_date]
                self.signal_files[symbol] = df
            except FileNotFoundError:  # pd.errors.EmptyDataError
                # When start_date is too recent to generate data, no files are generated
                print("No signals found for {} {}".format(symbol, inspect.getmodule(self.signal_func).__name__))
                self.symbols.remove(symbol)
                self.times[read_signals_time] = self.times[read_signals_time] + timer() - start_time
                return ta.default_signal

        signal = df.loc[date][self.signal_name] if date in df.index else ta.default_signal
        # TODO make this work for multiple signals
        '''
        signal = df.loc[date][self.signal_names[0]] if date in df.index else ta.default_signal
        for signal_name in self.signal_names:
            if signal != df.loc[date][signal_name] if date in df.index else ta.default_signal:
                signal = ta.default_signal
        '''
        self.times[read_signals_time] = self.times[read_signals_time] + timer() - start_time
        return signal

    def plot_against_benchmark(self, log, benchmark=None):
        """Plots data comparing the simulation to the benchmarks and available symbols

        Parameters:
            log : dataframe
            benchmark : list of str
        """

        if benchmark is None:
            benchmark = self.benchmark

        fig, ax = plt.subplots(2, 2, figsize=config.figsize)

        # Portfolio performance
        ax[0][0].plot(log.index, log[portfolio_value_column_name], label=portfolio_value_column_name)
        ax[0][0].plot(log.index, log[cash_column_name], label=cash_column_name)
        ax[0][0].plot(log.index, log[total_commission_column_name], label=total_commission_column_name)
        ax[0][0].plot(log.index, log[total_dividend_column_name], label=total_dividend_column_name)
        utils.prettify_ax(ax[0][0], title=portfolio_value_column_name, start_date=self.start_date, end_date=self.end_date)

        # Benchmark performance
        for bench in benchmark:
            df = pd.read_csv(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=bench), index_col="Date", parse_dates=["Date"])[self.start_date:self.end_date]
            ax[1][0].plot(df.index, df["AdjClose"], label=bench + "Price")
        utils.prettify_ax(ax[1][0], title="Benchmarks", start_date=self.start_date, end_date=self.end_date)

        # Portfolio compared to benchmarks
        ax[0][1].plot(log.index, (log[portfolio_value_column_name] / log[portfolio_value_column_name][0]), label="Portfolio")
        for bench in benchmark:
            df = pd.read_csv(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=bench), index_col="Date", parse_dates=["Date"])[self.start_date:self.end_date]
            ax[0][1].plot(df.index, (df["AdjClose"] / df["Close"][0]), label=bench)
            if len(benchmark) > 1:
                ax[0][1].plot(df.index, (log[portfolio_value_column_name] / log[portfolio_value_column_name][0]) / (df["Close"] / df["Close"][0]), label="PortfolioVS" + bench)
        utils.prettify_ax(ax[0][1], title="PortfolioVSBenchmarks", start_date=self.start_date, end_date=self.end_date)

        # Simulation compared to available symbols
        if len(self.symbols) < 6:
            ax[1][1].plot(log.index, (log[portfolio_value_column_name] / log[portfolio_value_column_name][0]), label="Portfolio")
            for symbol in self.symbols:
                df = pd.read_csv(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=symbol), index_col="Date", parse_dates=["Date"])[self.start_date:self.end_date]
                # ax[1][1].plot(df.index, (log[log.index in df.index][total_value_column_name] / log[log.index in df.index][cash_column_name][0]) / (df["Close"] / df["Close"][0]), label="PortfolioVS" + symbol)
                ax[1][1].plot(df.index, (df["AdjClose"] / df["Close"][0]), label=symbol)
            utils.prettify_ax(ax[1][1], title="PortfolioVSSymbols", start_date=self.start_date, end_date=self.end_date)

        # TODO: fix this for when not all symbols start at the same date. Fill_value=1 flattens before dates where the return can be calculated, and is inaccurate for dates after the return has already been calculated
        # Average return of available symbols
        if len(self.symbols) > 1:
            avg_return = pd.Series(0, index=self.dates)
            for symbol in self.symbols:
                df = pd.read_csv(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=symbol), index_col="Date", parse_dates=["Date"])[self.start_date:self.end_date]
                avg_return = avg_return.add((df["AdjClose"] / df["Close"][0]), fill_value=1)
            avg_return = avg_return / len(self.symbols)
            # add this to the PortfolioVSBenchmarks graph, don't reprettify
            ax[0][1].plot(avg_return.index, avg_return, label="AverageReturnOfSymbols")
            # add this to the PortfolioVSSymbols graph, don't reprettify
            ax[1][1].plot(avg_return.index, avg_return, label="AverageReturnOfSymbols")

        utils.prettify_fig(fig, title=self.filename)
        fig.savefig(utils.get_file_path(config.simulation_graphs_path, self.filename + simulation_graph_filename))
        utils.debug(fig)

    def run(self):
        """Runs the simulation
        """

        start_time = timer()

        for symbol in self.symbols:
            try:
                if utils.refresh(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=symbol), refresh=self.refresh):
                    prices.download_data_from_yahoo(symbol, start_date=self.start_date, end_date=self.end_date)
            except RemoteDataError:
                print("Invalid symbol: " + symbol)
                self.symbols.remove(symbol)
        for bench in self.benchmark:
            try:
                if utils.refresh(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=bench), refresh=self.refresh):
                    prices.download_data_from_yahoo(bench, start_date=self.start_date, end_date=self.end_date)
            except RemoteDataError:
                print("Invalid symbol: " + bench)
                self.symbols.remove(bench)
        self.times[download_data_time] = self.times[download_data_time] + timer() - start_time

        self.generate_signals()

        self.log.loc[self.dates[0]][cash_column_name, portfolio_column_name, actions_column_name, portfolio_value_column_name] = [self.cash, self.portfolio, "Initial ", self.total_value(self.dates[0])]

        try:
            for date in self.dates:
                print(date, flush=True)
                for symbol in self.portfolio:
                    self.get_dividends(symbol, date)
                for symbol in self.symbols:
                    signal = self.read_signal(symbol, date)
                    if signal == ta.buy_signal:
                        self.buy(symbol, date, self.purchase_size)
                    if signal == ta.sell_signal:
                        self.sell(symbol, date, sell_size=0)
                    if signal == ta.soft_buy_signal and self.soft_signals:
                        self.buy(symbol, date, self.purchase_size)
                    if signal == ta.soft_sell_signal and self.soft_signals and self.short_sell:
                        self.sell(symbol, date, self.purchase_size)
                self.log.loc[date][cash_column_name, portfolio_column_name, portfolio_value_column_name, total_commission_column_name, total_dividend_column_name] \
                    = [self.cash, self.portfolio_value(date), self.total_value(date), self.total_commissions, self.total_dividends]
                # Why does this line require str(self.portfolio)?
                # self.log.loc[date][cash_column_name, portfolio_column_name, portfolio_value_column_name] = [self.cash, str(self.portfolio), self.total_value(date)]
        except (AttributeError, KeyError) as e:
            if self.fail_gracefully:
                print(e)
                self.log = self.log.loc[self.log[self.log.index] < date]
            else:
                raise

        self.plot_against_benchmark(self.log, self.benchmark)

        self.times[total_time] = self.times[total_time] + timer() - start_time
        print(self.filename)
        print("Times: {}".format(self.format_times()))
        print(self.get_price_on_date.cache_info())
        print(str(psutil.Process(os.getpid()).memory_info().rss / float(2 ** 20)) + "mb")
        print("Dividends: {:.2f}".format(self.total_dividends))
        print("Commissions: {:.2f}".format(self.total_commissions))
        print("Total trades: {}".format(self.total_trades))
        print("Performance: {}".format(self.get_performance()))
        print("Sharpe ratios: {}".format(self.get_sharpe_ratios()))

        self.log.loc[date][cash_column_name, portfolio_column_name, actions_column_name, portfolio_value_column_name] = \
            [self.cash, self.portfolio_value(date), "Final: Performance: {} Times: {} Cache: {} Total Dividends: {:.2f} Total Commissions: {:.2f} Total Trades: {} Sharpe Ratios {}"
                .format(self.get_performance(), self.format_times(), self.get_price_on_date.cache_info(), self.total_dividends, self.total_commissions, self.total_trades, self.get_sharpe_ratios()), self.total_value(date)]
        self.log.to_csv(utils.get_file_path(config.simulation_data_path, self.filename + simulation_table_filename))
        self.log = self.log.loc[self.log[actions_column_name] != ""]  # self.log.dropna(subset=[actions_column_name], inplace=True)
        self.log.to_csv(utils.get_file_path(config.simulation_data_path, self.filename + simulation_actions_only_table_filename))

    # Functions for logging

    def portfolio_value(self, date):
        portfolio = {}
        for position in self.portfolio:
            portfolio[position] = "{:.2f}".format(self.portfolio[position] * self.get_price_on_date(position, date, time="Close"))
        return OrderedDict(sorted(portfolio.items()))

    def format_times(self):
        str_times = {}
        for entry in self.times:
            hours, rem = divmod(times[entry], 3600)
            minutes, seconds = divmod(rem, 60)
            str_times[entry] = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)
        return OrderedDict(sorted(str_times.items()))

    def get_performance(self):
        performances = {"Portfolio": "{:.2f}".format(self.log[portfolio_value_column_name][-1] / self.log[portfolio_value_column_name][0])}
        for bench in self.benchmark:
            performances[bench] = "{:.2f}".format(ta.get_performance(bench, self.start_date, self.end_date))
        return performances

    def get_sharpe_ratios(self):
        sharpe_ratios = {"Portfolio": "{:.2f}".format((self.log[portfolio_value_column_name] / self.log[portfolio_value_column_name].shift(1)).mean() / (self.log[portfolio_value_column_name] / self.log[portfolio_value_column_name].shift(1)).std())}
        for bench in self.benchmark:
            sharpe_ratios[bench] = "{:.2f}".format(ta.get_sharpe_ratio(bench, self.start_date, self.end_date))
        return sharpe_ratios


def update_dates_file(start_date=config.start_date, end_date=config.end_date):
    prices.download_data_from_yahoo("SPY", start_date=start_date, end_date=end_date)
    df = pd.read_csv(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol="SPY"), index_col="Date", parse_dates=["Date"])[start_date:end_date]
    df.to_csv(utils.get_file_path(config.simulation_data_path, dates_table_filename), columns=[])


if __name__ == '__main__':
    '''
    Simulation(symbols=sp.get_sp500(),
               refresh=False, filename="SP500EMA20-50",
               signal_func=ema.generate_signals, signal_func_kwargs={"period": [20, 50]})
    Simulation(symbols=sp.get_sp500(),
               refresh=False, filename="SP500EMA20-200",
               signal_func=ema.generate_signals, signal_func_kwargs={"period": [20, 200]})
    # This is the 'benchmark'/'default'
    Simulation(symbols=sp.get_sp500(),
               refresh=False, filename="SP500EMA50-200",
               signal_func=ema.generate_signals, signal_func_kwargs={"period": [50, 200]})
    Simulation(symbols=sp.get_sp500(),
               refresh=False, filename="SP500EMA50-200SoftSignals", soft_signals=True,
               signal_func=ema.generate_signals, signal_func_kwargs={"period": [50, 200]})
    #Simulation(symbols=sp.get_sp500(),
    #           refresh=False, filename="SP500EMA50-200ShortSell", short_sell=True,
    #           signal_func=ema.generate_signals, signal_func_kwargs={"period": [50, 200]})
    Simulation(symbols=sp.get_sp500(),
               refresh=False, filename="SP500EMA50-200PortSize20", max_portfolio_size=20,
               signal_func=ema.generate_signals, signal_func_kwargs={"period": [50, 200]})
    Simulation(symbols=sp.get_removed_sp500(),
               refresh=False, filename="RemovedSP500EMA50-200",
               signal_func=ema.generate_signals, signal_func_kwargs={"period": [50, 200]})
    Simulation(symbols=sp.get_sp500() + sp.get_removed_sp500(),
               refresh=False, filename="CurrentAndRemovedSP500EMA50-200",
               signal_func=ema.generate_signals, signal_func_kwargs={"period": [50, 200]})
    Simulation(symbols=sp.get_sp500(),
               refresh=False, filename="SP500SMA50-200",
               signal_func=sma.generate_signals, signal_func_kwargs={"period": [50, 200]})
    Simulation(symbols=sp.get_sp500(),
               refresh=False, filename="SP500MACD",
               signal_func=macd.generate_signals, signal_func_kwargs={"period": ema.default_periods})
    Simulation(symbols=sp.get_sp500(),
               refresh=False, filename="SP500RSI14",
               signal_func=rsi.generate_signals, signal_func_kwargs={"period": rsi.default_period})
    Simulation(symbols=["SPY"],
               refresh=False, filename="SPYEMA50-200",
               signal_func=ema.generate_signals, signal_func_kwargs={"period": [50, 200]})
    Simulation(symbols=["MSFT", "GOOG", "FB", "AAPL", "AMZN"],
               refresh=False, filename="BigNEMA50-200",
               signal_func=ema.generate_signals, signal_func_kwargs={"period": [50, 200]})
    '''

# Threading causes issues where, if reading shortly after file is generated, pandas will read an empty file and throw pandas.errors.EmptyDataError: No columns to parse from file
