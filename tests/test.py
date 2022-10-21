import os

if os.environ.get("USE_PYTORCH", None) == "1":
	from torch import Tensor
else:
	from light._C import Tensor

import unittest

class TestLight(unittest.TestCase):
	def test_basic(self):
		a = Tensor(2, 3)
		b = Tensor(2, 3)
		c = Tensor(2, 3)

		a.fill_(2.0)
		b.fill_(3.0)
		c.fill_(5.0)

		res = a + b
		print(res)
		self.assertEqual(list(a.size()), [2, 3])
		self.assertTrue(c.equal(res))
