from light._C import Tensor
import unittest

class TestLight(unittest.TestCase):
	def test_basic(self):
		a = Tensor([2, 3], 0)
		b = Tensor([2, 3], 0)
		c = Tensor([2, 3], 0)

		a.initWithScalar(2.0)
		b.initWithScalar(3.0)
		c.initWithScalar(5.0)

		res = a + b
		print(res)
		self.assertEqual(c, res)
