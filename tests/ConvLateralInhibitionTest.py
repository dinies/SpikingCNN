import unittest
from context import src
from src.ConvolutionalLayer import *

import pdb

class ConvLateralInhibitionTest( unittest.TestCase):
    def setUp(self):
        '''
        unifluential params
        '''
        strides_conv= [1,1,1,1]
        padding= "SAME"
        pooling_type= "MAX"
        encoding_t = 0
        stdp_flag = True
        conv_threshold = 0.0
        a_plus = .0
        a_minus = .0
        a_decay = .0
        stdp_threshold = 0.0

        self.layer = ConvolutionalLayer(
                padding, strides_conv,
                [1,1,1,1], conv_threshold,
                [1,1,1,1], [1,1,1,1],
                encoding_t, stdp_threshold,
                a_plus,a_minus, a_decay, stdp_flag)


        S = np.zeros([1,3,3,3])
        S[:,:,:,0] = np.matrix([[0,1,1],[0,0,0],[1,0,0]])
        S[:,:,:,1] = np.matrix([[0,1,1],[1,0,0],[0,1,0]])
        S[:,:,:,2] = np.matrix([[0,0,1],[1,0,0],[0,1,0]])
        self.S = S

        V = np.zeros([1,3,3,3])
        V[:,:,:,0] = np.matrix([[0,2,3],[0,0,0],[1.5,1,0]])
        V[:,:,:,1] = np.matrix([[0,2,2],[1.9,0,0],[0.5,2,0]])
        V[:,:,:,2] = np.matrix([[0,1,1],[2,0,0],[0.5,1.5,0]])
        self.V = V

        self.K_inh = np.matrix([[1,1,1],[0,1,1],[1,1,1]])

        S_out= np.zeros([1,3,3,3])
        S_out[:,:,:,0] = np.matrix([[0,1,1],[0,0,0],[1,0,0]])
        S_out[:,:,:,1] = np.matrix([[0,0,0],[0,0,0],[0,1,0]])
        S_out[:,:,:,2] = np.matrix([[0,0,0],[0,0,0],[0,0,0]])
        self.S_out = S_out

        self.K_out = np.matrix([[1,0,0],[0,1,1],[0,0,1]])





    def test_lateralInhibition(self):

        S_res, K_inh_res = self.layer.lateral_inh_CPU( self.S, self.V, self.K_inh )
        print( self.K_out)
        print('')
        print( K_inh_res)
        print('')
        print( self.S_out[:,:,:,0])
        print('')
        print( S_res[:,:,:,0])
        print('')
        print( self.S_out[:,:,:,1])
        print('')
        print( S_res[:,:,:,1])
        print('')
        print( self.S_out[:,:,:,2])
        print('')
        print( S_res[:,:,:,2])
   
if __name__ == '__main__':
    unittest.main()



