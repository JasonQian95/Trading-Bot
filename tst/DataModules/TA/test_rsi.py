import rsi
import config
import unittest
import warnings


class RSITest(unittest.TestCase):

    def setUp(self):
        warnings.filterwarnings(action="ignore", message="unclosed", category=ResourceWarning)

    def test_rsi(self):
        rsi.rsi(config.index, refresh=config.refresh)

    def test_plot_rsi(self):
        rsi.plot_rsi(config.index, refresh=config.refresh)

    def test_generate_signals(self):
        rsi.generate_signals(config.index, refresh=config.refresh)

    def test_plot_signals(self):
        rsi.plot_signals(config.index, refresh=config.refresh)


if __name__ == '__main__':
    unittest.main()
