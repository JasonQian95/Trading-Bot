import prices as p
import utils
import config
import unittest
from pandas_datareader._utils import RemoteDataError
import warnings


class PricesTest(unittest.TestCase):

    def setUp(self):
        warnings.filterwarnings(action="ignore", message="unclosed", category=ResourceWarning)

    def test_download_data_from_fred_sp500(self):
        p.download_data_from_fred(config.sp500)

    def test_download_data_from_fred_vix(self):
        p.download_data_from_fred(config.vix)

    def test_download_data_from_fred_except(self):
        self.assertRaises(RemoteDataError, p.download_data_from_fred, "AAPL")

    def test_download_data_from_yahoo_sp500(self):
        p.download_data_from_yahoo(config.index)

    def test_download_data_from_yahoo_aapl(self):
        p.download_data_from_yahoo("AAPL")

    def test_download_data_from_yahoo_except(self):
        self.assertRaises(RemoteDataError, p.download_data_from_yahoo, "----")

    def test_get_average_price_sp500(self):
        p.get_average_price(config.index, refresh=config.refresh)

    def test_get_average_price_aapl(self):
        p.get_average_price("AAPL", refresh=config.refresh)

    def test_plot_prices_sp500(self):
        p.plot_prices(config.index, refresh=config.refresh)

    def test_plot_prices_aapl(self):
        p.plot_prices("AAPL", refresh=config.refresh)

    def test_get_daily_return_sp500(self):
        p.get_daily_return(config.index, refresh=config.refresh)

    def test_get_daily_return_aapl(self):
        p.get_daily_return("AAPL", refresh=config.refresh)


if __name__ == '__main__':
    # TODO: make tests run in order. Unnessecary for passing tests, but I would like the final data files
    unittest.main(testLoader=utils.SequentialTestLoader())
    unittest.TestLoader.sortTestMethodsUsing = None

# Class that tries to run tests in order, isnt working
class SequentialTestLoader(unittest.TestLoader):
    def getTestCaseNames(self, testCaseClass):
        test_names = super().getTestCaseNames(testCaseClass)
        testcase_methods = list(testCaseClass.__dict__.keys())
        test_names.sort(key=testcase_methods.index)
        return test_names
