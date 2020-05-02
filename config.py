from os.path import dirname, join
import datetime

debug = False
verbose = False  # more debug info
dated = False  # dated filenames, turn off for testing
refresh = False  # refresh files used for testing

sp500 = "SP500"
index = "SPY"
vix = "VIXCLS"

start_date = datetime.date(2005, 1, 1)
end_date = datetime.date.today()
date_format = "%Y%m%d"

project_root = dirname(__file__)

data_folder_name = "data"
prices_folder_name = "Prices"
symbols_folder_name = "SymbolLists"
sp500_folder_name = "SP500"

data_path = join(project_root, data_folder_name)
prices_data_path = join(data_path, prices_folder_name)
symbols_data_path = join(data_path, symbols_folder_name)
sp500_symbols_data_path = join(symbols_data_path, sp500_folder_name)
