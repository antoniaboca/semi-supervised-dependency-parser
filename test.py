import unittest
from tests.data_load import DatasetTestCase, PadderTestCase

def suite():
    suite = unittest.TestSuite()
    suite.addTests(unittest.makeSuite(DatasetTestCase))
    suite.addTests(unittest.makeSuite(PadderTestCase))
    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())