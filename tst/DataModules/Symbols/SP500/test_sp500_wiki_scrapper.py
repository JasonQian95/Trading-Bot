import sp500_wiki_scrapper as sp500
import config
import unittest


class SP500WikiScrapperTest(unittest.TestCase):

    def test_get_sp500(self):
        sp500.get_sp500(refresh=config.refresh)

    def test_get_sp500_by_sector(self):
        sp500.get_sp500_by_sector('Communication Services')

    def test_get_sp500_by_subsector(self):
        sp500.get_sp500('Advertising')


if __name__ == '__main__':
    unittest.main()
