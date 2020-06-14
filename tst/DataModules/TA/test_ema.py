import ema
import config
import unittest
import warnings


class EMATest(unittest.TestCase):

    def setUp(self):
        warnings.filterwarnings(action="ignore", message="unclosed", category=ResourceWarning)

    def test_ema_50(self):
        ema.ema(config.index, 50, refresh=config.refresh)

    def test_plot_ema(self):
        ema.plot_ema(config.index, refresh=config.refresh)

    def test_generate_signals(self):
        ema.generate_signals(config.index, refresh=config.refresh)

    def test_plot_signals(self):
        ema.plot_signals(config.index, refresh=config.refresh)


if __name__ == '__main__':
    unittest.main()
