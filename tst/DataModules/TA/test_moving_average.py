import moving_average as ma
import config
import unittest
import warnings


class MovingAverageTest(unittest.TestCase):

    def setUp(self):
        warnings.filterwarnings(action="ignore", message="unclosed", category=ResourceWarning)

    def test_sma_index_50(self):
        ma.sma(config.index, 50, refresh=config.refresh)

    def test_ema_index_50(self):
        ma.ema(config.index, 50, refresh=config.refresh)

    def test_plot_sma_index(self):
        ma.plot_sma(config.index, refresh=config.refresh)

    def test_plot_ema_index(self):
        ma.plot_ema(config.index, refresh=config.refresh)

    def test_generate_signals_sma_index(self):
        ma.generate_sma_signals(config.index, refresh=config.refresh)

    def test_generate_signals_ema_index(self):
        ma.generate_ema_signals(config.index, refresh=config.refresh)

    def test_plot_macd_index(self):
        ma.plot_macd(config.index, refresh=config.refresh)

    def test_generate_macd_signals_index(self):
        ma.generate_macd_signals(config.index, refresh=config.refresh)


if __name__ == '__main__':
    unittest.main()
