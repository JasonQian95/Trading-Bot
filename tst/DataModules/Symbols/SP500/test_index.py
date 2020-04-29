import index
import config
import unittest


class IndexTest(unittest.TestCase):

    def test_get_sp500(self):
        index.get_index(refresh=config.refresh)


if __name__ == '__main__':
    unittest.main()
