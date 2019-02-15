import unittest
from context import src
from src.MyMath import MyMath

class EagerPendulumTest( unittest.TestCase):
    def setUp(self):
        self.epsilon = 0.0001

    def test_boxPlus(self):
        alfa = 1.5
        beta = 0.0
        
        result= MyMath.boxplus( alfa, beta )
        print (result)
        self.assertTrue(
                result <=  1.5 +self.epsilon and
                result >= 1.5 -self.epsilon)


if __name__ == '__main__':
    unittest.main()
