import bb
import config
import unittest
import warnings


class BBTest(unittest.TestCase):

    def setUp(self):
        warnings.filterwarnings(action="ignore", message="unclosed", category=ResourceWarning)

    def test_bb(self):
        bb.bb(config.index, refresh=config.refresh)

    def test_plot_bb(self):
        bb.plot_bb(config.index, refresh=config.refresh)

    def test_generate_signals(self):
        bb.generate_signals(config.index, refresh=config.refresh)

    def test_plot_signals(self):
        bb.plot_signals(config.index, refresh=config.refresh)


if __name__ == '__main__':
    unittest.main()
