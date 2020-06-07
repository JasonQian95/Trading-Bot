import relative_strength_index as rsi
import config
import unittest
import warnings


class RelativeStrengthIndexTest(unittest.TestCase):

    def setUp(self):
        warnings.filterwarnings(action="ignore", message="unclosed", category=ResourceWarning)

    def test_rsi_index(self):
        rsi.rsi(config.index, refresh=config.refresh)

    def test_plot_rsi_index(self):
        rsi.plot_rsi(config.index, refresh=config.refresh)

    def test_generate_signals_rsi_index(self):
        rsi.generate_rsi_signals(config.index, refresh=config.refresh)


if __name__ == '__main__':
    unittest.main()
