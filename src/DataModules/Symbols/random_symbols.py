import pandas as pd

import config
import utils

import random


def get_random_symbols(num=100):
    """Returns a list of num random symbols from a list of all symbols

    Parameters:
        num : int, optional

    Returns:
        list of str
            A list of num random symbols
    """
    df = pd.read_csv(utils.get_file_path(config.symbols_data_path, config.all_symbols_table_filename))
    symbol_list = df["Symbol"].tolist()
    random_list = []
    for s in range(1, num):
        random_list.append(random.choice(symbol_list))
    return random_list

