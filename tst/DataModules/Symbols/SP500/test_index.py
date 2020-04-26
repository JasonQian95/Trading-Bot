import unittest
import config
import index


class IndexTest(unittest.TestCase):

    def test_get_index(self):
        self.assertEqual(index.get_index(), config.index)


if __name__ == '__main__':
    unittest.main()
