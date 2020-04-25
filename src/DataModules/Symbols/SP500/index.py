import pandas as pd
import config

index_filename = config.index + ".csv"


def download_spy():
    df = pd.DataFrame(columns=["Symbol"], data=[[config.index]])
    # df = pd.DataFrame([{"Symbol":config.index}])
    df.to_csv(config.join(config.index_symbols_path, index_filename), index=False)


def get_spy():
    return config.index
