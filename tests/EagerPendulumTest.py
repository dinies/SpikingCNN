import unittest
from context import models
from models.EagerPendulum import EagerPendulum

class EagerPendulumTest( unittest.TestCase):
    def setUp(self):
        self.p = EagerPendulum()
        self.epsilon = 0.0001

    def test_dummy(self):

        self.assertTrue( True)

if __name__ == '__main__':
    unittest.main()
