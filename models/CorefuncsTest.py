import unittest
import numpy as np
import tensorflow as tf
#  

tf.enable_eager_execution()
class CorefuncsTest( unittest.TestCase):

    def test_conv2d(self):
        imageNp = np.zeros( [1,6,6,1])
        imageNp[0][0][1][0] = 8.0
        imageNp[0][1][0][0] = 8.0
        imageNp[0][1][1][0] = 8.0
        img = tf.constant( imageNp)
        filterNp = np.ones( [3,3,1,1])
        filter_input = tf.contrib.eager.Variable( filterNp)
        print( imageNp)
        stride = [1,2,2,1]
        padding= "SAME"
        conv_res = tf.nn.conv2d( img, filter_input, stride, padding)
        print(conv_res)
       
        print( conv_res.shape )
        


if __name__ == '__main__':
    unittest.main()
