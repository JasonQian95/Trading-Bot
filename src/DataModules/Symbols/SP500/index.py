import pandas as pd
import config

index_filename = config.index + ".csv"


def download_index():
    """Generates a csv file with containing a column name and the symbol of the index as define in config
    """

    df = pd.DataFrame(columns=["Symbol"], data=[[config.index]])
    # df = pd.DataFrame([{"Symbol":config.index}])
    df.to_csv(config.join(config.sp500_symbols_data_path, index_filename), index=False)


def get_index():
    """Returns the symbol of the index as define in config

    Returns:
        The symbol of the index as define in config
    """

    return config.index
