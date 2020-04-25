import pandas as pd
import config

sp500_full_filename = "SP500Full.csv"
sp500_symbols_filename = "SP500Symbols.csv"


def download_sp500():
    table = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    # df = table[0]
    # df = df.rename(columns=df.iloc[0]).drop(df.index[0])
    df = table[0][1:].rename(columns=table[0].iloc[0])
    df.to_csv(config.join(config.sp500_symbols_path, sp500_full_filename), index=False)
    df.to_csv(config.join(config.sp500_symbols_path, sp500_symbols_filename), columns=["Symbol"], index=False)


def get_sp500():
    table = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    df = table[0][1:].rename(columns=table[0].iloc[0])
    print(df["Symbol"].tolist())
