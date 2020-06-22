import random_symbols as rs
import config
import unittest


class SP500WikiScrapperTest(unittest.TestCase):

    def test_get_random_symbols_100(self):
        rs.get_random_symbols(num=100)


if __name__ == '__main__':
    unittest.main()
