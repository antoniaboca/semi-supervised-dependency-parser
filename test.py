import unittest
from tests.data_load import DatasetTestCase, PadderTestCase
from tests.nn_modules import BiaffineTestCase
from tests.edge_features import PriorTestCase, Top20TestCase, FeatureVectorTestCase

def suite():
    suite = unittest.TestSuite()
    suite.addTests(unittest.makeSuite(DatasetTestCase))
    suite.addTests(unittest.makeSuite(PadderTestCase))
    suite.addTests(unittest.makeSuite(BiaffineTestCase))
    suite.addTests(unittest.makeSuite(PriorTestCase))
    suite.addTests(unittest.makeSuite(Top20TestCase))
    suite.addTests(unittest.makeSuite(FeatureVectorTestCase))
    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())