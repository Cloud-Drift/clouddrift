from clouddrift import datasets
import unittest


if __name__ == "__main__":
    unittest.main()


class datasets_tests(unittest.TestCase):
    def test_gdp1h_opens(self):
        ds = datasets.gdp1h()
        self.assertTrue(ds)

    def test_gdp6h_opens(self):
        ds = datasets.gdp6h()
        self.assertTrue(ds)
