import matplotlib
import pandas

import datetime
import warnings
from os.path import dirname, join

# Debug settings
debug = False
verbose = False  # more debug info
dated = False  # dated filenames, turn off for testing
refresh = False  # refresh files used for testing
skip_test = False  # skip some tests that generate excess files
trash_data_files = True

# Frequently used symbols
index = "SPY"
sp500_yahoo = "^GSPC"
vix_yahoo = "^VIX"
test_symbol = "AAPL"
broken_symbols = ["HWM", "TT"]

# Date range for data, and formatting of dates for saved csvs
# start_date = datetime.date(2003, 6, 19)  # SPY closed at 100.02 on this date
start_date = datetime.date(2005, 1, 1)
end_date = datetime.date.today()
date_format = "%Y_%m_%d"

# pandas settings
pandas.plotting.register_matplotlib_converters()
date_parser = pandas.to_datetime  # add data_parser = config.data_parser to all read_csv, or try df.index = df.index.astype(datetime)

# matplotlib settings
figsize = (16, 9)
scatter_size = 100
scatter_alpha = 0.7
alpha = 0.3
# matplotlib.rcParams['legend.loc'] = "best"
matplotlib.rc("figure", max_open_warning=0)
# Probbaly shouldn't use green or red
matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(color=["blue", "green", "red", "cyan", "magenta"])
# These are the "Tableau 20" colors as RGB. Curently unused
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
for i in range(len(tableau20)):
    r, g, b = tableau20[i]
    tableau20[i] = (r / 255., g / 255., b / 255.)

# Ignore warnings
warnings.filterwarnings(action="ignore", message="unclosed", category=ResourceWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)

# Project structure absolute paths
project_root = dirname(__file__)

data_folder_name = "data"

junk_folder_name = "Junk"
prices_folder_name = "Prices"
simulation_folder_name = "Simulation"
ta_folder_name = "TA"
symbols_folder_name = "SymbolLists"
sp500_folder_name = "SP500"

graphs_folder_name = "Graphs"

data_path = join(project_root, data_folder_name)
junk_data_path = join(data_path, junk_folder_name)
junk_graphs_path = join(junk_data_path, graphs_folder_name)
prices_data_path = join(data_path, prices_folder_name) if not trash_data_files else junk_data_path
prices_graphs_path = join(prices_data_path, graphs_folder_name) if not trash_data_files else junk_graphs_path
simulation_data_path = join(data_path, simulation_folder_name)
simulation_graphs_path = join(simulation_data_path, graphs_folder_name)
ta_data_path = join(data_path, ta_folder_name) if not trash_data_files else junk_data_path
ta_graphs_path = join(ta_data_path, graphs_folder_name) if not trash_data_files else junk_graphs_path
symbols_data_path = join(data_path, symbols_folder_name)
sp500_symbols_data_path = join(symbols_data_path, sp500_folder_name)
