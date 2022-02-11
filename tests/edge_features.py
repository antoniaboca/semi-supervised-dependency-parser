import unittest
import numpy as np
from dependency_parsers.lit_data_module import edge_count, top20, feature_vector

class PriorTestCase(unittest.TestCase):
    def test_edge_count_unique(self):
        idxs = [1,1,1,1]
        tags = ['tag'] * 4
        xpos = ['root', 'noun', 'verb', 'adj']
        pars = [0, 2, 0, 1]

        sentence = (idxs, tags, xpos, pars)
        tree = {
            ('root','verb'): 1,
            ('verb', 'noun'): 1,
            ('noun', 'adj'): 1
        }

        graph = {
            ('root', 'noun'): 1,
            ('root', 'verb'): 1,
            ('root', 'adj'): 1,
            ('noun', 'root'):1,
            ('noun', 'verb'):1,
            ('noun', 'adj'):1,
            ('verb', 'root'):1,
            ('verb', 'noun'):1,
            ('verb', 'adj'):1,
            ('adj', 'root'):1,
            ('adj', 'noun'):1,
            ('adj', 'verb'):1,
        }

        tree_edges, graph_edges = edge_count([sentence])
        self.assertDictEqual(tree, tree_edges)
        self.assertDictEqual(graph, graph_edges)
    
    def test_edge_count_same(self):
        idxs = [1,1,1]
        tags = ['tag'] * 3
        xpos = ['root', 'noun', 'noun']
        pars = [0, 0, 1]

        sentence = (idxs, tags, xpos, pars)
        tree = {
            ('root', 'noun'):1,
            ('noun', 'noun'):1,
        }

        graph = {
            ('root','noun'): 2,
            ('noun', 'noun'): 2,
            ('noun', 'root'): 2,
        }

        tree_edges, graph_edges = edge_count([sentence])
        self.assertDictEqual(tree, tree_edges)
        self.assertDictEqual(graph, graph_edges)
    
    def test_edge_count_multiple(self):
        idxs = np.random.randint(5, size=4)
        tags = ['tag'] * 4
        xpos = ['root', 'noun', 'verb', 'adj']
        pars = [0, 2, 0, 1]

        sentence1 = (idxs, tags, xpos, pars)

        idxs = np.random.randint(5, size=3)
        tags = ['tag'] * 3
        xpos = ['root', 'noun', 'noun']
        pars = [0, 0, 1]

        sentence2 = (idxs, tags, xpos, pars)

        tree = {
            ('root', 'noun'):1,
            ('noun', 'noun'):1,
            ('root','verb'): 1,
            ('verb', 'noun'): 1,
            ('noun', 'adj'): 1
        }

        graph = {
            ('root', 'noun'): 3,
            ('root', 'verb'): 1,
            ('root', 'adj'): 1,
            ('noun', 'noun'): 2,
            ('noun', 'root'): 3,
            ('noun', 'verb'):1,
            ('noun', 'adj'):1,
            ('verb', 'root'):1,
            ('verb', 'noun'):1,
            ('verb', 'adj'):1,
            ('adj', 'root'):1,
            ('adj', 'noun'):1,
            ('adj', 'verb'):1,
        }

        tree_edges, graph_edges = edge_count([sentence1, sentence2])
        self.assertDictEqual(tree, tree_edges)
        self.assertDictEqual(graph, graph_edges)

class Top20TestCase(unittest.TestCase):
    def test_top20(self):
        tree = {
            (0, 1): 100,
            (1, 2): 101,
            (2, 3): 102,
            (3, 4): 103,
            (4, 5): 104,
            (5, 6): 105,
            (6, 7): 106,
            (7, 8): 107,
            (8, 9): 108,
            (9, 10): 109,
            (10, 11): 110,
            (11, 12): 111,
            (12, 13): 112,
            (13, 14): 113,
            (14, 15): 114,
            (15, 16): 115,
            (16, 17): 116,
            (17, 18): 117,
            (18, 19): 118,
            (19, 20): 119,
            (20, 21): 120,
        }

        graph = {
            (0, 1): 200,
            (1, 2): 200,
            (2, 3): 200,
            (3, 4): 200,
            (4, 5): 200,
            (5, 6): 200,
            (6, 7): 200,
            (7, 8): 200,
            (8, 9): 200,
            (9, 10): 200,
            (10, 11): 200,
            (11, 12): 200,
            (12, 13): 200,
            (13, 14): 200,
            (14, 15): 200,
            (15, 16): 200,
            (16, 17): 200,
            (17, 18): 200,
            (18, 19): 200,
            (19, 20): 200,
            (20, 21): 200
        }

        top20_dict = {
            (1, 2): 0.505,
            (2, 3): 0.51,
            (3, 4): 0.515,
            (4, 5): 0.52,
            (5, 6): 0.525,
            (6, 7): 0.53,
            (7, 8): 0.535,
            (8, 9): 0.54,
            (9, 10): 0.545,
            (10, 11): 0.55,
            (11, 12): 0.555,
            (12, 13): 0.56,
            (13, 14): 0.565,
            (14, 15): 0.57,
            (15, 16): 0.575,
            (16, 17): 0.58,
            (17, 18): 0.585,
            (18, 19): 0.59,
            (19, 20): 0.595,
            (20, 21): 0.6
        }
        
        order = {
            (1, 2): 19,
            (2, 3): 18,
            (3, 4): 17,
            (4, 5): 16,
            (5, 6): 15,
            (6, 7): 14,
            (7, 8): 13,
            (8, 9): 12,
            (9, 10): 11,
            (10, 11): 10,
            (11, 12): 9,
            (12, 13): 8,
            (13, 14): 7,
            (14, 15): 6,
            (15, 16): 5,
            (16, 17): 4,
            (17, 18): 3,
            (18, 19): 2,
            (19, 20): 1,
            (20, 21): 0
        }

        top20_r, order_r = top20(tree, graph)
        self.assertDictEqual(top20_r, top20_dict)
        self.assertDictEqual(order_r, order)

class FeatureVectorTestCase(unittest.TestCase):
    
    @unittest.expectedFailure
    def test_feature_vector_set_size(self):
        my_set = [([1], [2])]
        list = feature_vector(my_set)
        self.assertEqual(list, [1])

    def test_feature_vector_one_sentence(self):
        idxs = [1,1]
        tags = [0,1]
        xpos = [3,4]
        part = [0,0]
        labs = [1,1]
        wxps = ['a', 'b']

        set = [(idxs,tags,xpos,part,labs,wxps)]
        order = {
            (3,4): 0,
            (4,3): 1,
        }

        list1 = np.zeros((2,2,2), dtype=np.int32)
        list1[0][0] = np.array([0,0])
        list1[0][1] = np.array([1,0])
        list1[1][0] = np.array([0,1])
        list1[1][1] = np.array([0,0])

        list = feature_vector(order, set)
        self.assertEqual(list[0].shape, (2,2,2))
        self.assertTrue(np.array_equal(list[0], list1))

    def test_feature_vector_multiple_sentences(self):

        idxs = [1,1]
        tags = [0,1]
        xpos = [0,1]
        part = [0,0]
        labs = [1,1]
        wxps = ['a', 'b']

        sentence1 = (idxs,tags,xpos,part,labs,wxps)
        
        idxs = [1,1,1]
        tags = [0,1,2]
        xpos = [0,1,2]
        part = [0,0,0]
        labs = [1,1,1]
        wxps = ['a', 'b', 'c']

        sentence2 = (idxs, tags, xpos, part, labs, wxps)

        order = {
            (0,1): 0,
            (1,0): 1,
            (0,2): 2,
        }

        list1 = np.zeros((2,2,3),dtype=np.int32)
        list1[0][0] = np.array([0,0, 0])
        list1[0][1] = np.array([1,0, 0])
        list1[1][0] = np.array([0,1, 0])
        list1[1][1] = np.array([0,0, 0])

        list = feature_vector(order, [sentence1, sentence2])
        self.assertEqual(list[0].shape, (2,2,3))
        self.assertTrue(np.array_equal(list[0], list1))
        
        list2 = np.zeros((3,3,3), dtype=np.int32)
        list2[0][0] = np.array([0,0,0])
        list2[0][1] = np.array([1,0,0])
        list2[0][2] = np.array([0,0,1])
                
        list2[1][0] = np.array([0,1,0])
        list2[1][1] = np.array([0,0,0])
        list2[1][2] = np.array([0,0,0])

        list2[2][0] = np.array([0,0,0])
        list2[2][1] = np.array([0,0,0])
        list2[2][2] = np.array([0,0,0])

        self.assertEqual(list[1].shape, (3,3,3))
        self.assertTrue(np.array_equal(list[1], list2))

