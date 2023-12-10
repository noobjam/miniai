import unittest
import torch
from backprob import *

class TestRelu(unittest.TestCase):
    def setUp(self):
        self.relu = Relu()
        self.input = torch.tensor([-1.0, 0.0, 1.0])
        self.expected_output = torch.tensor([0.0, 0.0, 1.0])

    def test_call(self):
        self.assertTrue(torch.allclose(self.relu(self.input), self.expected_output))

    def test_backward(self):
        self.relu(self.input)
        self.relu.out.g = torch.ones_like(self.input)
        self.relu.backward()
        self.assertTrue(torch.allclose(self.relu.input.g, torch.tensor([0.0, 1.0, 1.0])))

class TestLin(unittest.TestCase):
    def setUp(self):
        self.w = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        self.b = torch.tensor([1.0, 1.0])
        self.lin = Lin(self.w, self.b)
        self.input = torch.tensor([1.0, 1.0])

    def test_call(self):
        self.assertTrue(torch.allclose(self.lin(self.input), torch.tensor([4.0, 9.0])))

    def test_backward(self):
        self.lin(self.input)
        self.lin.out.g = torch.ones_like(self.lin.out)
        self.lin.backward()
        self.assertTrue(torch.allclose(self.lin.input.g, torch.tensor([4.0, 6.0])))

class TestMse(unittest.TestCase):
    def setUp(self):
        self.mse = Mse()
        self.input = torch.tensor([1.0, 2.0, 3.0])
        self.target = torch.tensor([2.0, 2.0, 2.0])

    def test_call(self):
        self.assertEqual(self.mse(self.input, self.target), 1.0)

    def test_backward(self):
        self.mse(self.input, self.target)
        self.mse.backward()
        self.assertTrue(torch.allclose(self.mse.inp.g, torch.tensor([1.0, 0.0, 1.0])))

class TestModel(unittest.TestCase):
    def setUp(self):
        self.w1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        self.b1 = torch.tensor([1.0, 1.0])
        self.w2 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        self.b2 = torch.tensor([1.0, 1.0])
        self.model = Model(self.w1, self.b1, self.w2, self.b2)
        self.input = torch.tensor([1.0, 1.0])
        self.target = torch.tensor([2.0, 2.0])

    def test_call(self):
        self.assertTrue(torch.allclose(self.model(self.input, self.target), torch.tensor(1.0)))

    def test_backward(self):
        self.model(self.input, self.target)
        self.model.backward()
        self.assertTrue(torch.allclose(self.model.layers[0].input.g, torch.tensor([4.0, 6.0])))

if __name__ == '__main__':
    unittest.main()