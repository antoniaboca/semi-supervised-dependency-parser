import unittest
import torch
import torch.nn as nn

from dependency_parsers.parsers.supervised_parser import Biaffine

class BiaffineTestCase(unittest.TestCase):
    def test_batch_size_1_set_weights(self):
        head = torch.tensor([1,2,3], dtype=float).transpose(-1, 0)
        dep = torch.tensor([13, 14, 15], dtype=float).transpose(-1, 0)
        
        head = torch.stack([head])
        dep = torch.stack([dep])

        with torch.no_grad():
            U = [[4,5,6], [7,8,9], [10,11,12]]
            biaffine = Biaffine(3,1, torch.tensor(U, dtype=float).unsqueeze(-3))
            self.assertEqual(biaffine(head, dep), torch.tensor([[2280]], dtype=float))
    
    def test_batch_size_2_set_weights(self):
        head1 = torch.tensor([1,2,3,4], dtype=float).transpose(-1, 0)
        head2 = torch.tensor([10, 11,12,13], dtype=float).transpose(-1, 0)

        dep1 = torch.tensor([7,8,9,10], dtype=float).transpose(-1, 0)
        dep2 = torch.tensor([14,15,16,17], dtype=float).transpose(-1, 0)

        batch_h = torch.stack([head1, head2])
        batch_d = torch.stack([dep1, dep2])

        with torch.no_grad():
            U = [[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1]]
            biaffine = Biaffine(4,1,torch.tensor(U, dtype=float).unsqueeze(-3))
            self.assertTrue(torch.equal(biaffine(batch_h, batch_d), torch.tensor([[340], [2852]], dtype=float)))