import os.path as path
from os.path import dirname, join


index = "SPY"

data_folder_name = "data"
symbols_folder_name = "SymbolLists"
sp500_folder_name = "SP500"


project_root = dirname(__file__)

data_path = join(project_root, data_folder_name)
symbols_data_path = join(data_path, symbols_folder_name)
sp500_symbols_data_path = join(symbols_data_path, sp500_folder_name)

'''
src_folder_name = "src"
src_path = join(project_root, src_folder_name)
symbols_src_path = join(src_path, symbols_folder_name)
sp500_symbols_src_path = join(symbols_src_path, sp500_folder_name)
'''
