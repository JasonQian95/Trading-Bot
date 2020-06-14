import macd
import config
import unittest
import warnings


class MACDTest(unittest.TestCase):

    def setUp(self):
        warnings.filterwarnings(action="ignore", message="unclosed", category=ResourceWarning)

    def test_macd(self):
        macd.macd(config.index, refresh=config.refresh)

    def test_plot_macd(self):
        macd.plot_macd(config.index, refresh=config.refresh)

    def test_generate_signals(self):
        macd.generate_signals(config.index, refresh=config.refresh)

    def test_plot_signals(self):
        macd.plot_signals(config.index, refresh=config.refresh)


if __name__ == '__main__':
    unittest.main()
