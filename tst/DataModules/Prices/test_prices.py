from pandas_datareader._utils import RemoteDataError
import prices as p
import utils
import config
import unittest
import warnings


class PricesTest(unittest.TestCase):

    def setUp(self):
        warnings.filterwarnings(action="ignore", message="unclosed", category=ResourceWarning)

    '''
    def test_download_data_from_fred_index(self):
        p.download_data_from_fred(config.index)

    def test_download_data_from_fred_vix(self):
        p.download_data_from_fred(config.vix)

    def test_download_data_from_fred_except(self):
        self.assertRaises(RemoteDataError, p.download_data_from_fred, config.test_symbol)
    '''

    def test_download_data_from_yahoo_index(self):
        p.download_data_from_yahoo(config.index)

    def test_download_data_from_yahoo_except(self):
        self.assertRaises(RemoteDataError, p.download_data_from_yahoo, "this is not a valid symbol")

    def test_get_average_price_index(self):
        p.get_average_price(config.index, refresh=config.refresh)

    def test_plot_prices_index(self):
        p.plot_prices(config.index, refresh=config.refresh)

    def test_plot_prices_mult_symbol(self):
        p.plot_prices([config.index, config.sp500_yahoo, config.vix_yahoo], refresh=config.refresh)

    def test_plot_percentage_gains_index(self):
        p.plot_percentage_gains(config.index, refresh=config.refresh)

    def test_plot_percentage_gains_mult_symbol(self):
        p.plot_percentage_gains([config.index, config.sp500_yahoo, config.vix_yahoo], refresh=config.refresh)

    '''
    def test_get_daily_return_index(self):
        if config.skip_test:
            self.skipTest("Too many files generated")
        p.get_daily_return(config.index, refresh=config.refresh)

    def test_get_after_hours_daily_return_index(self):
        if config.skip_test:
            self.skipTest("Too many files generated")
        p.get_after_hours_daily_return(config.index, refresh=config.refresh)

    def test_get_during_hours_daily_return_index(self):
        if config.skip_test:
            self.skipTest("Too many files generated")
        p.get_during_hours_daily_return(config.index, refresh=config.refresh)
    '''

    def test_get_daily_return_daily(self):
        if config.skip_test:
            self.skipTest("Too many files generated")
        p.get_daily_return_flex(config.index, func="daily", refresh=config.refresh)

    def test_get_daily_return_all(self):
        if config.skip_test:
            self.skipTest("Too many files generated")
        p.get_daily_return_flex(config.index, refresh=config.refresh)

    def test_after_during_hours_returns_index_period(self):
        if config.skip_test:
            self.skipTest("Too many files generated")
        p.after_during_hours_returns(config.index, period=90)

    def test_after_during_hours_returns_index_date_range(self):
        if config.skip_test:
            self.skipTest("Too many files generated")
        p.after_during_hours_returns(config.index, start_date="2006-1-1", end_date="2012-1-1")


if __name__ == '__main__':
    unittest.main()
    # I'd like tests to run in sequential order. Unnessecary for passing tests, but I would like the final data files even when refresh=True
    # unittest.TestLoader.sortTestMethodsUsing = None  # This supposed to work but it seems the api is broken

# This doesn't work
'''
class SequentialTestLoader(unittest.TestLoader):
    def getTestCaseNames(self, testCaseClass):
        test_names = super().getTestCaseNames(testCaseClass)
        testcase_methods = list(testCaseClass.__dict__.keys())
        test_names.sort(key=testcase_methods.index)
        return test_names
'''