from os.path import dirname, join
import datetime
import matplotlib

# Debug settings
debug = False
verbose = False  # more debug info
dated = False  # dated filenames, turn off for testing
refresh = False  # refresh files used for testing
skip_test = True

# Frequently used symbols
sp500 = "SP500"
index = "SPY"
vix = "VIXCLS"
sp500_yahoo = "^GSPC"
vix_yahoo = "^VIX"
test_symbol = "AAPL"

# Date range for data, and formatting of dates for saved csvs
start_date = datetime.date(2005, 1, 1)
end_date = datetime.date.today()
date_format = "%Y%m%d"

# matplotlib settings
matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(color=["blue", "green", "red", "cyan", "magenta", "red"])
figsize = (16, 9)

# Project structure absolute paths
project_root = dirname(__file__)

data_folder_name = "data"
prices_folder_name = "Prices"
symbols_folder_name = "SymbolLists"
sp500_folder_name = "SP500"

data_path = join(project_root, data_folder_name)
prices_data_path = join(data_path, prices_folder_name)
symbols_data_path = join(data_path, symbols_folder_name)
sp500_symbols_data_path = join(symbols_data_path, sp500_folder_name)
