import pandas as pd

import prices
import sp500_wiki_scrapper as sp

import ema
import sma
import macd
import rsi
import bb

import config
import tautils as ta
import utils

import schedule
import time
import datetime

table_filename = ".csv"


def job():
    for func in [ema, sma, macd, rsi, bb]:
        buy_list = []
        sell_list = []
        for symbol in sp.get_sp500():
            # TODO: needs a refresh
            func.generate_signals(symbol, start_date=utils.add_business_days(datetime.date.today(), -200), end_date=datetime.date.today())
            df = pd.read_csv(utils.get_file_path(config.ta_data_path, func.table_filename, symbol=symbol), index_col="Date", parse_dates=["Date"])
            if df.index[-1].date() != datetime.date.today():
                print("Last date in {} file does not equal todays date. Is today a weekend or a holiday?".format(symbol))
            if df[func.get_signal_name()][-1] == ta.buy_signal:
                buy_list.append(symbol)
            if df[func.get_signal_name()][-1] == ta.sell_signal:
                sell_list.append(symbol)

        df = pd.DataFrame({"Buy": pd.Series(buy_list),
                          "Sell": pd.Series(sell_list)})
        df.to_csv(utils.get_file_path(config.ta_data_path, func.get_signal_name() + table_filename, dated=True, start_date="", end_date=datetime.date.today()), index=False)


schedule.every().monday.at("16:05").do(job)
schedule.every().tuesday.at("16:05").do(job)
schedule.every().wednesday.at("16:05").do(job)
schedule.every().thursday.at("16:05").do(job)
schedule.every().friday.at("16:05").do(job)

while True:
    schedule.run_pending()
    time.sleep(1)
