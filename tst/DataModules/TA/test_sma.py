import sma
import config
import unittest
import warnings


class SMATest(unittest.TestCase):

    def setUp(self):
        warnings.filterwarnings(action="ignore", message="unclosed", category=ResourceWarning)

    def test_sma_50(self):
        sma.sma(config.index, 50, refresh=config.refresh)

    def test_plot_sma(self):
        sma.plot_sma(config.index, refresh=config.refresh)

    def test_generate_signals(self):
        sma.generate_signals(config.index, refresh=config.refresh)


if __name__ == '__main__':
    unittest.main()
