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
        #print( imageNp)
  
        stride = [1,2,2,1]
        padding= "SAME"
        conv_res = tf.nn.conv2d( img, filter_input, stride, padding)
        #print(conv_res)
       
        print( "1st test")
        print( conv_res.shape )

    def test_pooling(self):

        imageNp = np.zeros( [1,6,6,1])
        imageNp[0][0][1][0] = 8.0
        imageNp[0][1][0][0] = 8.0
        imageNp[0][1][1][0] = 8.0
        img = tf.constant( imageNp)
        window_shape = [4,4]
        pool_type = "MAX"
        padding= "SAME"
        stride = [3,3]
        pool_res = tf.nn.pool( img, window_shape, pool_type, padding , strides= stride)
        print( "2nd test")
        print( pool_res.shape )
  #     print( pool_res)
        
       



    def test_complex_structure(self):

        padding= "SAME"
        pool_type = "MAX"
        strides_conv = [1,1,1,1]

        # Input layer 
        imageNp = np.zeros( [1,160,250,1])
        img = tf.constant( imageNp)

        # 1 conv layer 
        filterNp = np.ones( [5,5,1,4])
        filter_input = tf.contrib.eager.Variable( filterNp)
        conv1_res = tf.nn.conv2d( img, filter_input, strides_conv, padding)

        # 1 pooling layer 
        window_shape = [7,7]
        stridePool = [6,6]
        pooling1_res = tf.nn.pool( conv1_res, window_shape, pool_type, padding, strides = stridePool)

        # 2 conv layer 
        filterNp = np.ones( [17,17,4,20])
        filter_input = tf.contrib.eager.Variable( filterNp)
        conv2_res = tf.nn.conv2d( pooling1_res, filter_input, strides_conv, padding )

        # 2 pooling layer 
        window_shape = [5,5]
        stridePool = [5,5]
        pooling2_res = tf.nn.pool( conv2_res, window_shape, pool_type, padding, strides = stridePool)

        # 3 conv layer 
        filterNp = np.ones( [5,5,20,20])
        filter_input = tf.contrib.eager.Variable( filterNp)
        conv3_res = tf.nn.conv2d( pooling2_res, filter_input, strides_conv, padding )

        print( "1 conv: " )
        print( conv1_res.shape)
        print( "\n1 pool: ")
        print( pooling1_res.shape)
        print( "\n2 conv: ")
        print( conv2_res.shape )
        print( "\n2 pool: ")
        print( pooling2_res.shape)
        print( "\n3 conv: ")
        print( conv3_res.shape )
       

if __name__ == '__main__':
    unittest.main()
