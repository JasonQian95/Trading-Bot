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
import inspect
from pandas_datareader._utils import RemoteDataError
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
log_columns = [cash_column_name, portfolio_column_name, actions_column_name, portfolio_value_column_name]

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

    def __init__(self, initial_cash=100000, symbols=["SPY"], benchmark=["SPY"],
                 max_port_size=20, purchase_size=0, commission=default_commission, short_sell=False, soft_signals=False,
                 filename="", fail_gracefully=False, refresh=False, start_date=config.start_date, end_date=config.end_date,
                 signal_func=None, *args, **kwargs):
        if len(symbols) == 0:
            raise ValueError("Requires at least one symbol")
        if signal_func is None:
            raise ValueError("Requires a signal function")

        self.cash = initial_cash
        self.initial_cash = initial_cash
        self.symbols = symbols
        self.benchmark = benchmark
        self.purchase_size = initial_cash // min(max_port_size, len(symbols) // 5) if purchase_size == 0 else purchase_size
        self.commission = commission
        self.short_sell = short_sell
        self.soft_signals = soft_signals
        self.filename = filename
        self.fail_gracefully = fail_gracefully
        self.refresh = refresh
        self.start_date = start_date
        self.end_date = end_date
        self.signal_func = signal_func

        self.portfolio = {}
        self.signal_files = {}
        self.price_files = {}
        self.times = times
        self.total_dividends = 0
        self.total_commissions = 0
        self.dates = pd.read_csv(utils.get_file_path(config.simulation_data_path, dates_table_filename), index_col="Date", parse_dates=["Date"])[self.start_date:self.end_date].index  # Probably faster to do .index before [start_date:end_date]
        self.log = pd.DataFrame(index=self.dates, columns=log_columns)
        self.log[actions_column_name] = ""
        self.signal_table_filename = None

        self.run(self.signal_func, *args, **kwargs)

    def buy(self, symbol, date, purchase_size, partial_shares=False):
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
            # TODO: current behavior can lead to 'races' between stocks, where after buying the first stock there is no money for the second
            purchase_size = self.cash
        if symbol not in self.portfolio:
            # TODO: buy at close price, or next day's open price?
            shares = (purchase_size // self.get_price_on_date(symbol, date, time="Close")) if not partial_shares else (purchase_size / self.get_price_on_date(symbol, date, time="Close"))
            if shares < 0:
                return
                # TODO: Sometimes shares is -1. When variables are printed, math does not add up to -1??
                # Symbol SLB self.purchase_size 5000 Purchase size -1.7605099999366214 Price 39.79 shares -1.0
                # print("Symbol {} self.purchase_size {} Purchase size {} Price {} shares {}".format(symbol, self.purchase_size, purchase_size, self.get_price_on_date(symbol, date, time="Close"), shares), flush=True)
            if shares != 0:
                self.portfolio[symbol] = shares
                self.cash -= shares * self.get_price_on_date(symbol, date, time="Close")
                self.cash -= self.calculate_commission(shares)
                self.log.loc[date][actions_column_name] = self.log.loc[date][actions_column_name] + "{} {} {}  Shares at {} totaling {} ".format(ta.buy_signal, symbol, shares, self.get_price_on_date(symbol, date), shares * self.get_price_on_date(symbol, date))
        else:
            # buy more
            pass

    def sell(self, symbol, date, amount=0, partial_shares=False, short_sell=False):
        """Simulates selling a stock

        Parameters:
            symbol : str
            date : datetime
            amount : float, optional
                How much to sell. If 0, sell all
            partial_shares : bool
                Whether partial shares are supported. If True, the amount sold will always equal amount, even if that number isn't reachable in a number of whole shares
            short_sell : bool
        """

        if symbol in self.portfolio:
            # TODO: sell at close price, or next day's open price?
            if amount == 0:  # sell all
                self.cash += self.portfolio[symbol] * self.get_price_on_date(symbol, date)
                self.cash -= self.calculate_commission(self.portfolio[symbol])
                self.log.loc[date][actions_column_name] = self.log.loc[date][actions_column_name] + "{} {} {}  Shares at {} totalling {} ".format(ta.sell_signal, symbol, self.portfolio[symbol], self.get_price_on_date(symbol, date), self.portfolio[symbol] * self.get_price_on_date(symbol, date))
                self.portfolio.pop(symbol)
            else:
                shares = (amount // self.get_price_on_date(symbol, date)) if not partial_shares else (amount / self.get_price_on_date(symbol, date))
                shares = shares if shares < self.portfolio[symbol] else self.portfolio[symbol]
                if shares < 0:
                    print("Amount {} Price {} shares {}".format(amount, self.get_price_on_date(symbol, date, time="Close"), shares), flush=True)
                    raise Exception
                if shares != 0:
                    self.cash += shares * self.get_price_on_date(symbol, date)
                    self.cash -= self.calculate_commission(shares)
                    self.log.loc[date][actions_column_name] = self.log.loc[date][actions_column_name] + "{} {} {}  Shares at {} totalling {} ".format(ta.sell_signal, symbol, shares, self.get_price_on_date(symbol, date), shares * self.get_price_on_date(symbol, date))
                    self.portfolio[symbol] -= shares
                    if self.portfolio[symbol] <= 0:
                        self.portfolio.pop(symbol)
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

    def generate_signals(self, symbols=None, refresh=None, start_date=None, end_date=None, signal_func=None, *args, **kwargs):
        """Generates signals for the given symbol

        Parameters:
            symbols : list of str
            start_date : datetime, optional
            end_date : datetime, optional
        """

        start_time = timer()

        self.signal_table_filename = inspect.getmodule(signal_func).table_filename

        if symbols is None:
            symbols = self.symbols
        if isinstance(symbols, str):
            symbols = [symbols]
        if signal_func is None:
            signal_func = self.signal_func
        if refresh is None:
            refresh = self.refresh
        if start_date is None:
            start_date = self.start_date
        if end_date is None:
            end_date = self.end_date

        for symbol in symbols:
            print("Generating signals for " + symbol, flush=True)
            try:
                # TODO this must be run is signal_func has changed, but can be skipped if it hasn't
                signal_func(symbol, *args, refresh=refresh, start_date=start_date, end_date=end_date, **kwargs)
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

        if self.signal_table_filename is None:
            raise Exception

        if symbol in self.signal_files:
            df = self.signal_files[symbol]
        else:
            try:
                # TODO this is hardcoded
                df = pd.read_csv(utils.get_file_path(config.ta_data_path, self.signal_table_filename, symbol=symbol), index_col="Date", parse_dates=["Date"])[self.start_date:self.end_date]
                self.signal_files[symbol] = df
            except FileNotFoundError:
                print("No signals found for " + symbol)
                self.symbols.remove(symbol)
                self.times[read_signals_time] = self.times[read_signals_time] + timer() - start_time
                return ta.default_signal

        signal = df.loc[date][ta.signal_name] if date in df.index else ta.default_signal
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
        utils.prettify_ax(ax[0][0], title=portfolio_value_column_name, start_date=self.start_date, end_date=self.end_date)

        # Benchmark performance
        for bench in benchmark:
            df = pd.read_csv(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=bench), index_col="Date", parse_dates=["Date"])[self.start_date:self.end_date]
            ax[1][0].plot(df.index, df["Close"], label=bench + "Price")
        utils.prettify_ax(ax[1][0], title="Benchmarks", start_date=self.start_date, end_date=self.end_date)

        # Portfolio compared to benchmarks
        for bench in benchmark:
            df = pd.read_csv(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=bench), index_col="Date", parse_dates=["Date"])[self.start_date:self.end_date]
            ax[0][1].plot(df.index, (log[portfolio_value_column_name] / log[portfolio_value_column_name][0]) / (df["Close"] / df["Close"][0]), label="PortfolioVS" + bench)
        utils.prettify_ax(ax[0][1], title="PortfolioVSBenchmarks", start_date=self.start_date, end_date=self.end_date)

        '''
        # Slope of portfolio compared to benchmarks
        for bench in benchmark:
            df = pd.read_csv(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=bench), index_col="Date", parse_dates=["Date"])[self.start_date:self.end_date]
            ax[1][1].plot(df.index, ((log[portfolio_value_column_name] / log[cash_column_name][0]) / (df["Close"] / df["Close"][0]) / (log[portfolio_value_column_name].shift(1) / log[cash_column_name][0]) / (df["Close"].shift(1) / df["Close"][0])), label="SlopeOfPortfolioVS" + bench)
        utils.prettify_ax(ax[1][1], title="SlopeOfPortfolioVSBenchmarks", start_date=self.start_date, end_date=self.end_date)
        
        # Simulation compared to available symbols
        if len(self.symbols) < 4:
            for symbol in self.symbols:
                df = pd.read_csv(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=symbol), index_col="Date", parse_dates=["Date"])[self.start_date:self.end_date]
                # ax[2][0].plot(df.index, (log[log.index in df.index][total_value_column_name] / log[log.index in df.index][cash_column_name][0]) / (df["Close"] / df["Close"][0]), label="PortfolioVS" + symbol)
                ax[2][0].plot(df.index, df["Close"] / df["Close"][0])
            ax[2][0].plot(log.index, log[portfolio_value_column_name] / log[portfolio_value_column_name][0], label=portfolio_value_column_name)
            utils.prettify_ax(ax[2][0], title="PortfolioVSSymbols", start_date=self.start_date, end_date=self.end_date)
        '''

        # TODO: fix this
        # Average return of available symbols
        avg_return = pd.Series(0, index=self.dates)
        for symbol in self.symbols:
            df = pd.read_csv(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=symbol), index_col="Date", parse_dates=["Date"])[self.start_date:self.end_date]
            avg_return = avg_return.add(df["Close"] / df["Close"][0], fill_value=1)
        avg_return = avg_return / len(self.symbols)
        ax[1][1].plot(avg_return.index, avg_return, label="AverageReturnOfSymbols")
        utils.prettify_ax(ax[1][1], title="AverageReturnOfSymbols", start_date=self.start_date, end_date=self.end_date)

        utils.prettify_fig(fig)
        fig.savefig(utils.get_file_path(config.simulation_graphs_path, self.filename + simulation_graph_filename))
        utils.debug(fig)

    def run(self, signal_func, *args, **kwargs):
        """Runs the simulation
        """

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

        self.generate_signals(self.symbols, refresh=self.refresh, signal_func=signal_func, *args, **kwargs)

        self.log.loc[self.dates[0]][cash_column_name, portfolio_column_name, actions_column_name, portfolio_value_column_name] = [self.cash, self.portfolio, "Initial ", self.total_value(self.dates[0])]

        try:
            for date in self.dates:
                print(date, flush=True)
                for symbol in self.portfolio:
                    self.get_dividends(symbol, date)
                for symbol in self.symbols:
                    signal = self.read_signal(symbol, date)
                    # Instead of buying as much as possible, don't buy if we cannot make a full purchase
                    if signal == ta.buy_signal and self.cash > self.purchase_size:
                        self.buy(symbol, date, self.purchase_size)
                        self.log.loc[date][cash_column_name, portfolio_column_name, portfolio_value_column_name] = [self.cash, self.portfolio, self.total_value(date)]
                    # logging only
                    if signal == ta.buy_signal and self.cash < self.purchase_size:
                        self.log.loc[date][actions_column_name] = self.log.loc[date][actions_column_name] + "Unable to buy {} ".format(symbol)
                    if signal == ta.sell_signal:
                        self.sell(symbol, date, amount=0)
                        self.log.loc[date][cash_column_name, portfolio_column_name, portfolio_value_column_name] = [self.cash, self.portfolio, self.total_value(date)]
                    # Instead of buying as much as possible, don't buy if we cannot make a full purchase
                    if signal == ta.soft_buy_signal and self.soft_signals and self.cash > self.purchase_size:
                        self.buy(symbol, date, self.purchase_size)
                        self.log.loc[date][cash_column_name, portfolio_column_name, portfolio_value_column_name] = [self.cash, self.portfolio, self.total_value(date)]
                    # logging only
                    if signal == ta.soft_buy_signal and self.soft_signals and self.cash < self.purchase_size:
                        self.log.loc[date][actions_column_name] = self.log.loc[date][actions_column_name] + "Unable to buy {} ".format(symbol)
                    if signal == ta.soft_sell_signal and self.soft_signals and self.short_sell:
                        self.sell(symbol, date, self.purchase_size)
                        self.log.loc[date][cash_column_name, portfolio_column_name, portfolio_value_column_name] = [self.cash, self.portfolio, self.total_value(date)]
                # TODO: Why does this line require str(self.portfolio)?
                self.log.loc[date][cash_column_name, portfolio_column_name, portfolio_value_column_name] = [self.cash, str(self.portfolio), self.total_value(date)]
        except (AttributeError, KeyError) as e:
            if self.fail_gracefully:
                print(e)
                self.log = self.log.loc[self.log[self.log.index] < date]
                # self.log = self.log[:self.log.index.get_loc(date) - 1]
            else:
                raise

        self.plot_against_benchmark(self.log, self.benchmark)

        self.times[total_time] = self.times[total_time] + timer() - start_time
        print(self.times)
        print(self.get_price_on_date.cache_info())
        print(self.total_dividends)
        print(self.total_commissions)
        print(self.get_sharpe_ratios(self.log))
        print(self.get_sharpe_ratios2(self.log))

        self.log.loc[date][cash_column_name, portfolio_column_name, actions_column_name, portfolio_value_column_name] = [self.cash, self.portfolio, "Final: Performance: {} Times: {} Cache: {} Total Dividends: {} Total Commissions: {} Sharpe Ratios {} Sharpe Ratios 2 {}".format(self.get_performance(self.log), self.times, self.get_price_on_date.cache_info(), self.total_dividends, self.total_commissions, self.get_sharpe_ratios(self.log), self.get_sharpe_ratios2(self.log)), self.total_value(date)]
        self.log.to_csv(utils.get_file_path(config.simulation_data_path, self.filename + simulation_table_filename))
        self.log = self.log.loc[self.log[actions_column_name] != ""]  # self.log.dropna(subset=[actions_column_name], inplace=True)
        self.log.to_csv(utils.get_file_path(config.simulation_data_path, self.filename + simulation_actions_only_table_filename))

    def get_performance(self, log):
        performances = {}
        performances["Portfolio"] = log[portfolio_value_column_name][-1] / log[portfolio_value_column_name][0]
        for bench in self.benchmark:
            df = pd.read_csv(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=bench), index_col="Date", parse_dates=["Date"])[self.start_date:self.end_date]
            performances[bench] = df["Close"][-1] / df["Close"][0]
        return performances

    # TODO Move this out
    def get_sharpe_ratios(self, log):
        sharpe_ratios = {}
        # multiply sharpe ratios by (252 ^0.5) to annualize
        sharpe_ratios["Portfolio"] = (log[portfolio_value_column_name] / log[portfolio_value_column_name].shift(1)).mean() / (log[portfolio_value_column_name] / log[portfolio_value_column_name].shift(1)).std()
        for bench in self.benchmark:
            df = pd.read_csv(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=bench), index_col="Date", parse_dates=["Date"])[self.start_date:self.end_date]
            sharpe_ratios[bench] = (df["Close"] / df["Close"].shift(1)).mean() / (df["Close"] / df["Close"].shift(1)).std()
        return sharpe_ratios

    def get_sharpe_ratios2(self, log):
        sharpe_ratios = {}
        # multiply sharpe ratios by (252 ^0.5) to annualize
        big_r = log[portfolio_value_column_name].cumsum()
        small_r = (big_r - big_r.shift(1)) / big_r.shift(1)
        sharpe_ratios["Portfolio"] = small_r.mean() / small_r.std()
        for bench in self.benchmark:
            df = pd.read_csv(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=bench), index_col="Date", parse_dates=["Date"])[self.start_date:self.end_date]
            big_r = df["Close"].cumsum()
            small_r = (big_r - big_r.shift(1)) / big_r.shift(1)
            sharpe_ratios[bench] = small_r.mean() / small_r.std()
        return sharpe_ratios


def update_dates_file(start_date=config.start_date, end_date=config.end_date):
    prices.download_data_from_yahoo("SPY", start_date=start_date, end_date=end_date)
    df = pd.read_csv(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol="SPY"), index_col="Date", parse_dates=["Date"])[start_date:end_date]
    df.to_csv(utils.get_file_path(config.simulation_data_path, dates_table_filename), columns=[])


if __name__ == '__main__':
    #Simulation(symbols=["ABT", "ATVI", "AMD", "GOOG", "AMZN", "AAL", "AAPL", "T", "BAC", "BA", "CSCO", "KO", "ED", "COST", "CVS", "DAL", "DLR", "DFS", "DLTR", "DUK", "DRE", "EBAY", "EA", "EXPE", "XOM", "FB", "F", "FOX", "GE", "GM", "GILD", "GS", "HRB", "HOG", "HAS", "HP", "INTC", "IBM", "MA", "MCD", "MGM", "MSFT", "NFLX", "NVDA", "OXY", "PYPL", "PG", "QCOM", "O", "RCL", "CRM", "LUV", "TGT", "TWTR", "UAL", "UPS", "V", "WMT", "DIS", "WFC"],
    Simulation(symbols=sp.get_sp500(),
               refresh=False, filename="SP500MACD", max_port_size=100,
               signal_func=ma.generate_macd_signals)
    Simulation(symbols=sp.get_sp500(),
               refresh=False, soft_signals=True, filename="SP500EMA50-200SoftSignals20PortSize",
               signal_func=ma.generate_signals, ma_type=ma.ema_name, period=[50, 200])
    Simulation(symbols=sp.get_sp500(),
               refresh=False, soft_signals=True, filename="SP500EMA50-200SoftSignals", max_port_size=100,
               signal_func=ma.generate_signals, ma_type=ma.ema_name, period=[50, 200])

# ["ABT", "ATVI", "AMD", "GOOG", "AMZN", "AAL", "AAPL", "T", "BAC", "BA", "CSCO", "KO", "ED", "COST", "CVS", "DAL", "DLR", "DFS", "DLTR", "DUK", "DRE", "EBAY", "EA", "EXPE", "XOM", "FB", "F", "FOX", "GE", "GM", "GILD", "GS", "HRB", "HOG", "HAS", "HP", "INTC", "IBM", "MA", "MCD", "MGM", "MSFT", "NFLX", "NVDA", "OXY", "PYPL", "PG", "QCOM", "O", "RCL", "CRM", "LUV", "TGT", "TWTR", "UAL", "UPS", "V", "WMT", "DIS", "WFC"]

# TODO Thread this
# TODO for sp500, 2020-1-1, it tries to open ma files that weren't generated
# number of shares to buy is calculated before commission. Commission can then push cash into negatives
