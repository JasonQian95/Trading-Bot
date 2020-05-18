import pandas as pd
import config
import utils

sp500_full_table_filename = "FullTable.csv"
sp500_symbols_table_filename = "Symbols.csv"
sp500_full_table_path = utils.get_file_path(config.sp500_symbols_data_path, sp500_full_table_filename, "SP500")
sp500_symbols_table_path = utils.get_file_path(config.sp500_symbols_data_path, sp500_symbols_table_filename, "SP500")

# Wikipedia table column names
symbol_column_name = "Symbol"
sector_column_name = "GICS Sector"
sub_sector_column_name = "GICS Sub Industry"


def download_sp500():
    """Generates two csv files, one containing the full S&P500 table from Wikipedia, and the other containing only the symbols
    """

    table = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    df = table[0]
    # df = df.rename(columns=df.iloc[0]).drop(df.index[0])
    # df = table[0][1:].rename(columns=table[0].iloc[0])
    utils.debug(df)
    df.to_csv(sp500_full_table_path, index=False)
    df.to_csv(sp500_symbols_table_path, columns=[symbol_column_name], index=False)


def get_sp500(refresh=False):
    """Returns a list of symbols contained in the SP500

    Parameters:
        refresh : bool, optional

    Returns:
        list of str
            A list of symbols contained in the SP500
    """

    if utils.refresh(sp500_symbols_table_path, refresh=refresh):
        download_sp500()
    df = pd.read_csv(sp500_symbols_table_path)
    utils.debug(df[symbol_column_name])
    return df[symbol_column_name].tolist()


def get_sp500_by_sector(sector):
    """Returns a list of symbols contained in the SP500 filtered by industry

    Parameters:
        sector : str or list of str
            The sector(s) to filter for
            List of sectors as of 26/4/2020: ['Communication Services', 'Consumer Discretionary', 'Consumer Staples', 'Energy', 'Financials', 'Health Care', 'Industrials', 'Information Technology', 'Materials', 'Real Estate', 'Utilities']

    Returns:
        list of str
            A list of symbols contained in the SP500 filtered by industry
    """

    if utils.refresh(sp500_symbols_table_path):
        download_sp500()
    df = pd.read_csv(sp500_full_table_path, usecols=[symbol_column_name, sector_column_name])
    if type(sector) is str:
        df = df.loc[df[sector_column_name] == sector]
    elif isinstance(sector, list):
        df = df.loc[df[sector_column_name].isin(sector)]
    utils.debug(df[symbol_column_name])
    return df[symbol_column_name].tolist()


def get_sp500_by_sub_sector(sector):
    """Returns a list of symbols contained in the SP500 filtered by sub sector

    Parameters:
        sector : str or list of str
            The sector(s) to filter for
            List of sectors as of 26/4/2020: ['Advertising', 'Aerospace & Defense', 'Agricultural & Farm Machinery', 'Agricultural Products', 'Air Freight & Logistics', 'Airlines', 'Alternative Carriers', 'Apparel Retail', 'Apparel, Accessories & Luxury Goods', 'Application Software', 'Asset Management & Custody Banks', 'Auto Parts & Equipment', 'Automobile Manufacturers', 'Automotive Retail', 'Biotechnology', 'Brewers', 'Broadcasting', 'Building Products', 'Cable & Satellite', 'Casinos & Gaming', 'Commodity Chemicals', 'Communications Equipment', 'Computer & Electronics Retail', 'Construction & Engineering', 'Construction Machinery & Heavy Trucks', 'Construction Materials', 'Consumer Electronics', 'Consumer Finance', 'Copper', 'Data Processing & Outsourced Services', 'Department Stores', 'Distillers & Vintners', 'Distributors', 'Diversified Banks', 'Diversified Chemicals', 'Diversified Support Services', 'Drug Retail', 'Electric Utilities', 'Electrical Components & Equipment', 'Electronic Components', 'Electronic Equipment & Instruments', 'Electronic Manufacturing Services', 'Environmental & Facilities Services', 'Fertilizers & Agricultural Chemicals', 'Financial Exchanges & Data', 'Food Distributors', 'Food Retail', 'Gas Utilities', 'General Merchandise Stores', 'Gold', 'Health Care Distributors', 'Health Care Equipment', 'Health Care Facilities', 'Health Care REITs', 'Health Care Services', 'Health Care Supplies', 'Health Care Technology', 'Home Furnishings', 'Home Improvement Retail', 'Homebuilding', 'Hotel & Resort REITs', 'Hotels, Resorts & Cruise Lines', 'Household Appliances', 'Household Products', 'Housewares & Specialties', 'Human Resource & Employment Services', 'Hypermarkets & Super Centers', 'IT Consulting & Other Services', 'Independent Power Producers & Energy Traders', 'Industrial Conglomerates', 'Industrial Gases', 'Industrial Machinery', 'Industrial REITs', 'Insurance Brokers', 'Integrated Oil & Gas', 'Integrated Telecommunication Services', 'Interactive Home Entertainment', 'Interactive Media & Services', 'Internet & Direct Marketing Retail', 'Internet Services & Infrastructure', 'Investment Banking & Brokerage', 'Leisure Products', 'Life & Health Insurance', 'Life Sciences Tools & Services', 'Managed Health Care', 'Metal & Glass Containers', 'Motorcycle Manufacturers', 'Movies & Entertainment', 'Multi-Sector Holdings', 'Multi-Utilities', 'Multi-line Insurance', 'Office REITs', 'Oil & Gas Drilling', 'Oil & Gas Equipment & Services', 'Oil & Gas Exploration & Production', 'Oil & Gas Refining & Marketing', 'Oil & Gas Storage & Transportation', 'Packaged Foods & Meats', 'Paper Packaging', 'Personal Products', 'Pharmaceuticals', 'Property & Casualty Insurance', 'Publishing', 'Railroads', 'Real Estate Services', 'Regional Banks', 'Reinsurance', 'Research & Consulting Services', 'Residential REITs', 'Restaurants', 'Retail REITs', 'Semiconductor Equipment', 'Semiconductors', 'Soft Drinks', 'Specialized Consumer Services', 'Specialized REITs', 'Specialty Chemicals', 'Specialty Stores', 'Steel', 'Systems Software', 'Technology Distributors', 'Technology Hardware, Storage & Peripherals', 'Thrifts & Mortgage Finance', 'Tobacco', 'Trading Companies & Distributors', 'Trucking', 'Water Utilities', 'Wireless Telecommunication Services']

    Returns:
        list of str
            A list of symbols contained in the SP500 filtered by sub sector
    """

    if utils.refresh(sp500_symbols_table_path):
        download_sp500()
    df = pd.read_csv(sp500_full_table_path, usecols=[symbol_column_name, sub_sector_column_name])
    if type(sector) is str:
        df = df.loc[df[sub_sector_column_name] == sector]
    elif isinstance(sector, list):
        df = df.loc[df[sub_sector_column_name].isin(sector)]
    utils.debug(df[symbol_column_name])
    return df[symbol_column_name].tolist()


def get_all_sp500_sectors():
    """Returns a list of valid sectors in the SP500

    Returns:
        list of str
            A list of valid sectors in the SP500
    """

    if utils.refresh(sp500_symbols_table_path):
        download_sp500()
    df = pd.read_csv(sp500_full_table_path, usecols=[sector_column_name])
    return sorted(df[sector_column_name].unique())


def get_all_sp500_sub_sectors():
    """Returns a list of valid sub sectors in the SP500

    Returns:
        list of str
            A list of valid sub sectors in the SP500
    """

    if utils.refresh(sp500_symbols_table_path):
        download_sp500()
    df = pd.read_csv(sp500_full_table_path, usecols=[sub_sector_column_name])
    return sorted(df[sub_sector_column_name].unique())
