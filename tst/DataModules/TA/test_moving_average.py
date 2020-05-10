import moving_average as avg
import config
import unittest


class MovingAverageTest(unittest.TestCase):

    def test_sma_sp500_50(self):
        avg.sma(config.sp500, 50, refresh=config.refresh)

    def test_sma_aapl_200(self):
        avg.sma("AAPL", 200, refresh=config.refresh)

    def test_ema_sp500_50(self):
        avg.ema(config.sp500, 50, refresh=config.refresh)

    def test_ema_aapl_200(self):
        avg.ema("AAPL", 200, refresh=config.refresh)

    def test_plot_sma_sp500(self):
        avg.plot_sma(config.sp500, refresh=config.refresh)

    def test_plot_sma_aapl(self):
        avg.plot_sma("AAPL", refresh=config.refresh)

    def test_plot_ema_sp500(self):
        avg.plot_ema(config.sp500, refresh=config.refresh)

    def test_plot_ema_aapl(self):
        avg.plot_ema("AAPL", refresh=config.refresh)


if __name__ == '__main__':
    unittest.main()
