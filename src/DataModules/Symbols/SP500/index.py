import pandas as pd
import config
import utils

index_filename = ".csv"
index_table_path = utils.get_file_path(config.sp500_symbols_data_path, index_filename, config.index)


def download_index():
    """Generates a csv file with containing the symbol of the index as define in config
    """

    df = pd.DataFrame(columns=["Symbol"], data=[[config.index]])
    # df = pd.DataFrame([{"Symbol":config.index}])

    utils.debug(df)

    df.to_csv(index_table_path, index=False)


def get_index(refresh=False):
    """Returns the symbol of the index as define in config

    Parameters:
        refresh : bool, optional
            Recreate the data file, regardless of whether or not it already exists
    Returns:
        str
            The symbol of the index as define in config
    """

    if utils.refresh(index_table_path, refresh):
        download_index()

    utils.debug(config.index)

    return config.index
