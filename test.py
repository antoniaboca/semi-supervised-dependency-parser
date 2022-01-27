import unittest
from tests.data_load import DatasetTestCase, PadderTestCase
from tests.nn_modules import BiaffineTestCase

def suite():
    suite = unittest.TestSuite()
    suite.addTests(unittest.makeSuite(DatasetTestCase))
    suite.addTests(unittest.makeSuite(PadderTestCase))
    suite.addTests(unittest.makeSuite(BiaffineTestCase))
    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())