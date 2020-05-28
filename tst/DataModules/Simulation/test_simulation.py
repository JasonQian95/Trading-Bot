import simulation as s
import config
import unittest
import warnings


class SimulationTest(unittest.TestCase):

    def setUp(self):
        warnings.filterwarnings(action="ignore", message="unclosed", category=ResourceWarning)

    def test(self):
        pass


if __name__ == '__main__':
    unittest.main()
