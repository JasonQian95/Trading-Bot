import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import prices
import config
import utils


days_before_merger = -5
days_after_merger = 5

# list of spacs that have known issues with adding/subtracting business days due to holidays. Also OPEN is missing a day randomly
fail_list = ["RMO", "VINC", "PRCH", "ARKO", "OPEN", "AVCT", "PAE", "VRT", "RPAY", "WTRH"]
# list of spacs with missing data
data_issues_list = ["AGLY", "CLVR", "HUNTF", "KLDI", "LAZY", "SONG"]

spac_data = pd.read_csv(utils.get_file_path(config.spac_data_path, "SPAC.csv", symbol=""), index_col=False, parse_dates=["Completion Date"])
pre_and_post_merger_prices = []
failed_count = 0
failed_symbols = []
for index, row in spac_data.iterrows():
    symbol = row["Post-SPAC Ticker Symbol"]
    start_date = utils.add_business_days(row["Completion Date"], days_before_merger)
    end_date = utils.add_business_days(row["Completion Date"], days_after_merger)

    if symbol not in data_issues_list:
        # Never refresh for spacs, Yahoo consistently missing data around mergers
        df = pd.read_csv(utils.get_file_path(config.spac_data_path, prices.price_table_filename, symbol=symbol), index_col="Date", parse_dates=["Date"])
        try:
            # Preferably we would use the below line and also filter dfs for [start_date:end_date]
            # but there are issues where the dates are not as expected due to holdidays. Also OPEN is missing a day randomly
            pre_and_post_merger_prices.append((df.loc[start_date]["Close"], df.loc[end_date]["Close"]))
        except KeyError:
            if symbol in fail_list:
                try:
                    # Produces wrong data when dates are missing, and Yahoo seems to consistently have missing data around mergers
                    # However the data has been manually verified, and this gets around the issue with holidays
                    pre_and_post_merger_prices.append((df["Close"][0], df["Close"][-1]))
                except KeyError as e:
                    failed_count += 1
                    failed_symbols.append(symbol)
            else:
                failed_count += 1
                failed_symbols.append(symbol)

print(pre_and_post_merger_prices)
print(failed_count)
print(failed_symbols)

pre_and_post_merger_prices = np.array(pre_and_post_merger_prices)
fig, ax = plt.subplots(1, figsize=config.figsize)
ax.scatter(pre_and_post_merger_prices[:, 0], pre_and_post_merger_prices[:, 1], label="SpacPreAndPostMergerPrice")

ax.set_title("SPACPreAndPostMergerPrice")
ax.set_xlabel("PreMergerPrice")
ax.set_ylabel("PostMergerPrice")
ax.set_xlim(xmin=0)
ax.set_ylim(ymin=0)
if ax.get_xlim()[1].astype(int) < ax.get_ylim()[1].astype(int):
    ax.set_xlim(xmax=ax.get_ylim()[1].astype(int))
elif ax.get_xlim()[1].astype(int) > ax.get_ylim()[1].astype(int):
    ax.set_ylim(ymax=ax.get_xlim()[1].astype(int))
max_x = ax.get_xlim()[1].astype(int)
max_y = ax.get_ylim()[1].astype(int)
ax.plot([0, max_x], [0, max_y], transform=ax.transAxes, linestyle="--")

ax.set_axisbelow(True)
ax.xaxis.grid(color="black", linestyle="--")
ax.yaxis.grid(color="black", linestyle="--")
fig.savefig(utils.get_file_path(config.prices_graphs_path, "SPACPreAndPostMergerPrice", symbol=""))
