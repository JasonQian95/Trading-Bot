import numpy as np
import pandas as pd

import prices
import config
import utils

# strategy from:
# https://logical-invest.com/universal-investment-strategy/
# https://seekingalpha.com/article/2714185-the-spy-tlt-universal-investment-strategy
# https://seekingalpha.com/article/2823336-spy-tlt-universal-investment-strategy-20-year-backtest
# https://seekingalpha.com/article/3050016-hell-on-fire-the-3x-leveraged-universal-investment-strategy-part-i
# https://seekingalpha.com/article/4299701-leveraged-etfs-for-long-term-investing
# https://seekingalpha.com/article/4115470-strategy-beats-vanguard-mutual-funds-and-etfs
# https://seekingalpha.com/article/3548056-simple-spy-top-off-portfolio
# https://www.bogleheads.org/forum/viewtopic.php?t=272007

index = "UPRO"
bonds = "TMF"
default_symbols = [index, bonds]

increments = 10
lookback = 72
modified = False  # this is basically a volatility tax (raises the denominator of the sharpe ratio to the 2.5 power)


def rebalance(symbol=default_symbols, refresh=False, start_date=config.start_date, end_date=config.end_date):

    if utils.refresh(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=symbol[0]), refresh=refresh):
        prices.download_data_from_yahoo(symbol[0], start_date=start_date, end_date=end_date)
    index_df = pd.read_csv(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=symbol[0]), usecols=["Date", "Close", "Dividends"], index_col="Date", parse_dates=["Date"])[start_date:end_date]
    if utils.refresh(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=symbol[1]), refresh=refresh):
        prices.download_data_from_yahoo(symbol[1], start_date=start_date, end_date=end_date)
    bonds_df = pd.read_csv(utils.get_file_path(config.prices_data_path, prices.price_table_filename, symbol=symbol[1]), usecols=["Date", "Close", "Dividends"], index_col="Date", parse_dates=["Date"])[start_date:end_date]

    best_sharpe_ratio = -100  # technically should be -infi
    best_portfolio_balance = -1
    for i in range(0, increments + 1):
        portfolio = pd.DataFrame({
            "Close": (index_df["Close"].add(index_df["Dividends"].cumsum()).pct_change().tail(lookback) * i / increments).add((bonds_df["Close"].add(bonds_df["Dividends"].cumsum()).pct_change().tail(lookback) * (increments - i) / increments))
            })
        sharpe_ratio = portfolio["Close"].pct_change().mean() / ((portfolio["Close"].pct_change().std() * np.sqrt(252)) ** (2.5 if modified else 1)) * (1000000 if modified else 100)  # the last multiplier is to make the numbers more readable
        print("The ratio of {:.2f} {}, {:.2f} {} had a sharpe ratio of {:.10f}".format(i / increments, index, (increments - i) / increments, bonds, sharpe_ratio))
        if sharpe_ratio > best_sharpe_ratio:
            best_sharpe_ratio = sharpe_ratio
            best_portfolio_balance = i
    print("The best ratio was: {} {}, {} {} with a sharpe ratio of {}".format(best_portfolio_balance / increments, index, (increments - best_portfolio_balance) / increments, bonds, best_sharpe_ratio))
    print("Date was: " + index_df.last_valid_index().strftime("%Y-%m-%d"))
    return best_portfolio_balance / increments


rebalance()
