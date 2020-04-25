import os.path as path
from os.path import dirname, join
from pathlib import Path

index = "SPY"

data_folder_name = "data"
symbols_folder_name = "SymbolLists"
sp500_folder_name = "SP500"
index_folder_name = "Index"

'''
project_root = Path(__file__).parent
data_path = Path(project_root, data_folder_name)
symbols_path = Path(data_path, symbols_folder_name)
sp500_symbols_path = Path(symbols_path, sp500_folder_name)
'''

project_root = dirname(__file__)
data_path = join(project_root, data_folder_name)
symbols_path = join(data_path, symbols_folder_name)
sp500_symbols_path = join(symbols_path, sp500_folder_name)
index_symbols_path = join(symbols_path, index_folder_name)
