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
    def test_download_data_from_fred_sp500(self):
        p.download_data_from_fred(config.sp500)

    def test_download_data_from_fred_vix(self):
        p.download_data_from_fred(config.vix)

    def test_download_data_from_fred_except(self):
        self.assertRaises(RemoteDataError, p.download_data_from_fred, config.test_symbol)
    '''

    def test_download_data_from_yahoo_sp500(self):
        p.download_data_from_yahoo(config.sp500)

    def test_download_data_from_yahoo_aapl(self):
        p.download_data_from_yahoo(config.test_symbol)

    def test_download_data_from_yahoo_except(self):
        self.assertRaises(RemoteDataError, p.download_data_from_yahoo, "----")

    def test_get_average_price_sp500(self):
        p.get_average_price(config.sp500, refresh=config.refresh)

    def test_get_average_price_aapl(self):
        p.get_average_price(config.test_symbol, refresh=config.refresh)

    def test_plot_prices_sp500(self):
        p.plot_prices(config.sp500, refresh=config.refresh)

    def test_plot_prices_aapl(self):
        p.plot_prices(config.test_symbol, refresh=config.refresh)

    def test_get_daily_return_sp500(self):
        if config.skip_test:
            self.skipTest("Too many files generated")
        p.get_daily_return(config.sp500, refresh=config.refresh)

    def test_get_daily_return_aapl(self):
        if config.skip_test:
            self.skipTest("Too many files generated")
        p.get_daily_return(config.test_symbol, refresh=config.refresh)

    def test_get_after_hours_daily_return_sp500(self):
        if config.skip_test:
            self.skipTest("Too many files generated")
        p.get_after_hours_daily_return(config.sp500, refresh=config.refresh)

    def test_get_after_hours_daily_return_aapl(self):
        if config.skip_test:
            self.skipTest("Too many files generated")
        p.get_after_hours_daily_return(config.test_symbol, refresh=config.refresh)

    def test_get_during_hours_daily_return_sp500(self):
        if config.skip_test:
            self.skipTest("Too many files generated")
        p.get_during_hours_daily_return(config.sp500, refresh=config.refresh)

    def test_get_during_hours_daily_return_aapl(self):
        if config.skip_test:
            self.skipTest("Too many files generated")
        p.get_during_hours_daily_return(config.test_symbol, refresh=config.refresh)

    def test_after_during_hours_returns_sp500(self):
        if config.skip_test:
            self.skipTest("Too many files generated")
        p.after_during_hours_returns(config.sp500, period=90)

    def test_after_during_hours_returns_aapl(self):
        if config.skip_test:
            self.skipTest("Too many files generated")
        p.after_during_hours_returns(config.test_symbol, start_date="2015-1-1")


class SequentialTestLoader(unittest.TestLoader):
    def getTestCaseNames(self, testCaseClass):
        test_names = super().getTestCaseNames(testCaseClass)
        testcase_methods = list(testCaseClass.__dict__.keys())
        test_names.sort(key=testcase_methods.index)
        return test_names


if __name__ == '__main__':
    # TODO: make tests run in order. Unnessecary for passing tests, but I would like the final data files
    unittest.main(testLoader=SequentialTestLoader())
    unittest.TestLoader.sortTestMethodsUsing = None
