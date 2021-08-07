import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import prices
import config
import utils

days_before_merger = -5
days_after_merger = 5

# list of spacs that have known issues with adding/subtracting business days due to holidays. Also OPEN and AVCT are missing a day randomly
fail_list = ["RMO", "VINC", "PRCH", "ARKO", "OPEN", "AVCT", "PAE", "VRT", "RPAY", "WTRH"]
# list of spacs with missing data
data_issues_list = ["AGLY", "CLVR", "HUNTF", "KLDI", "LAZY", "SONG"]

spac_data = pd.read_csv(utils.get_file_path(config.spac_data_path, "SPAC.csv", symbol=""), index_col=False, parse_dates=["Completion Date"])
pre_and_post_merger_prices = []
failed_count = 0
failed_symbols = []

near_floor_lower_bound = 9.5
near_floor_upper_bound = 12
above_floor_lower_bound = 18
above_floor_upper_bound = 25
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
                except KeyError:
                    failed_count += 1
                    failed_symbols.append(symbol)
            else:
                failed_count += 1
                failed_symbols.append(symbol)

# debug
print("Failed symbols count: " + str(failed_count))
print("Failed symbols: " + str(failed_symbols))

# stats
pre_list = []
post_list = []
for pre, post in pre_and_post_merger_prices:
    pre_list.append(pre)
    post_list.append(post)
print()
print("The average pre-merger value of spacs is " + str(sum(pre_list) / len(pre_list)))
print("The average post-merger value of spacs is " + str(sum(post_list) / len(post_list)))
print("Sample size is " + str(len(pre_list)))

near_floor_pre_list = []
near_floor_post_list = []
for pre, post in pre_and_post_merger_prices:
    if near_floor_lower_bound < pre < near_floor_upper_bound:
        near_floor_pre_list.append(pre)
        near_floor_post_list.append(post)
print()
print("The average pre-merger value of spacs near {:.2f} to {:.2f} is {:.2f}".format(near_floor_lower_bound, near_floor_upper_bound, sum(near_floor_pre_list) / len(near_floor_pre_list)))
print("The average post-merger value of spacs near {:.2f} to {:.2f} is {:.2f}".format(near_floor_lower_bound, near_floor_upper_bound, sum(near_floor_post_list) / len(near_floor_post_list)))
print("Sample size is " + str(len(near_floor_pre_list)))

above_floor_pre_list = []
above_floor_post_list = []
for pre, post in pre_and_post_merger_prices:
    if above_floor_lower_bound < pre < above_floor_upper_bound:
        above_floor_pre_list.append(pre)
        above_floor_post_list.append(post)
above_floor_deviation = [max(a - b, 0) for a, b in zip(above_floor_pre_list, above_floor_post_list)]
above_floor_deviation_as_percentage = [max(a - b, 0) / a for a, b in zip(above_floor_pre_list, above_floor_post_list)]
print()
print("The average pre-merger value of spacs near {:.2f} to {:.2f} is {:.2f}".format(above_floor_lower_bound, above_floor_upper_bound, sum(above_floor_pre_list) / len(above_floor_pre_list)))
print("The average post-merger value of spacs near {:.2f} to {:.2f} is {:.2f}".format(above_floor_lower_bound, above_floor_upper_bound, sum(above_floor_post_list) / len(above_floor_post_list)))
print("The standard deviation of spacs near {:.2f} to {:.2f} is {:.2f}".format(above_floor_lower_bound, above_floor_upper_bound, np.std(above_floor_deviation)))
print("The standard deviation as a perrcentage of spacs near {:.2f} to {:.2f} is {:.2f}".format(above_floor_lower_bound, above_floor_upper_bound, np.std(above_floor_deviation_as_percentage)))
print("Sample size is " + str(len(above_floor_pre_list)))

# charting
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
