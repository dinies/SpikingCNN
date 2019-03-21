import unittest
from context import src
from src.ConvolutionalLayer import *
import time


class ConvolutionalLayerTest( unittest.TestCase):
    def setUp(self):
        strides_conv= [1,1,1,1]
        padding= "SAME"
        pooling_type= "MAX"
        encoding_t = 15
        stdp_flag = True
        threshold = 10.0

        a_plus = .5
        a_minus = 0.
        a_decay = 0.

        self.layer = ConvolutionalLayer(
                padding, strides_conv,
                [3,3,2,2], threshold,
                [1,2,2,2], [1,2,2,2],
                encoding_t,a_plus,
                a_minus, a_decay, stdp_flag )

        self.layer.curr_iteration = 1
        self.layer.weights = np.ones( [3,3,2,2]) * .5

        presynBefore = np.zeros( [1,2,2,2])
        presynBefore[0,0,1,0] = 1.
        presynBefore[0,1,1,1] = 1.
        presynAfter= np.zeros( [1,2,2,2])
        presynAfter[0,1,0,1] = 1.

        postsynBefore= np.zeros( [1,2,2,2])
        postsynBefore[0,0,0,0] = 1.
        postsynBefore[0,1,0,1] = 1.
        postsynAfter= np.zeros( [1,2,2,2])
        postsynAfter[0,0,1,0] = 1.
        postsynAfter[0,0,1,1] = 1.

        self.layer.spikes_presyn[:,:,:,:,0] = presynBefore
        self.layer.spikes_presyn[:,:,:,:,1] = presynAfter
        self.layer.spikes_postsyn[:,:,:,:,0] = postsynBefore
        self.layer.spikes_postsyn[:,:,:,:,1] = postsynAfter

    def test_computeDeconvolutionIndexesSamePaddingOddFilterDim(self):
        filt_dim = [3,3,2,2]
        input_dim = [1,2,2,2] 
        output_dim = [1,2,2,2]
                
        indexes = computeDeconvolutionIndexesSamePaddingOddFilterDim( 0, 0, filt_dim, input_dim, output_dim)
        truth_indexes = [[0,0,1,1],[0,1,1,2],[1,0,2,1],[1,1,2,2]]
        for t in truth_indexes:
            self.assertTrue( t in indexes)
          

    def test_STDP(self):
        t_0 = time.time()
        self.layer.STDP_learning()
        t_1 = time.time()
        print(str(t_1- t_0))
        weights = self.layer.weights


   # ch_in: 0 , ch_out: 0
        self.assertEqual( weights[0,0,0,0], .5)
        self.assertEqual( weights[0,1,0,0], .5)
        self.assertEqual( weights[0,2,0,0], .5)
        self.assertEqual( weights[1,0,0,0], .5) 
        self.assertEqual( weights[1,1,0,0], .625)
        self.assertEqual( weights[1,2,0,0], .5)
        self.assertEqual( weights[2,0,0,0], .5)
        self.assertEqual( weights[2,1,0,0], .5) 
        self.assertEqual( weights[2,2,0,0], .5)
   # ch_in: 1 , ch_out: 0
        self.assertEqual( weights[0,0,1,0], .5)
        self.assertEqual( weights[0,1,1,0], .5)
        self.assertEqual( weights[0,2,1,0], .5) 
        self.assertEqual( weights[1,0,1,0], .5) 
        self.assertEqual( weights[1,1,1,0], .5)
        self.assertEqual( weights[1,2,1,0], .5)
        self.assertEqual( weights[2,0,1,0], .625)
        self.assertEqual( weights[2,1,1,0], .625)
        self.assertEqual( weights[2,2,1,0], .5)
   # ch_in: 0 , ch_out: 1
        self.assertEqual( weights[0,0,0,1], .5)
        self.assertEqual( weights[0,1,0,1], .5)
        self.assertEqual( weights[0,2,0,1], .5)
        self.assertEqual( weights[1,0,0,1], .5) 
        self.assertEqual( weights[1,1,0,1], .625)
        self.assertEqual( weights[1,2,0,1], .5)
        self.assertEqual( weights[2,0,0,1], .5)
        self.assertEqual( weights[2,1,0,1], .5) 
        self.assertEqual( weights[2,2,0,1], .5)
   # ch_in: 1 , ch_out: 1
        self.assertEqual( weights[0,0,1,1], .5)
        self.assertEqual( weights[0,1,1,1], .5)
        self.assertEqual( weights[0,2,1,1], .5)  
        self.assertEqual( weights[1,0,1,1], .5) 
        self.assertEqual( weights[1,1,1,1], .5)
        self.assertEqual( weights[1,2,1,1], .5)
        self.assertEqual( weights[2,0,1,1], .625) 
        self.assertEqual( weights[2,1,1,1], .625)
        self.assertEqual( weights[2,2,1,1], .5)

if __name__ == '__main__':
    unittest.main()


'''
this section of the code was extracted since it has obscure implications on the weight changing behaviour
due to the high redundancy of the convolutions in stride 1,1,1,1 . It also introduces an high computational cost,
this cost could be reduced utilizing support data structures to memorize the indexes of the cells that have spiked
in this way the algorithm would not iterate in this 4 dimensional structures of pre and post synaptic spike structs.
For now we decided to use the effect of the global decay factor to simulate the weakened synapses in lieu of a
targeted update as shown in the following lines.
# weaken synapsis 
                    for t_output in range( self.curr_iteration):
                        if self.spikes_postsyn[0,row,column,channel_output,t_output] == 1:
                            for [in_row , in_col , w_row, w_col] in indexes:   
                                for channel_in in range(channels_in):
                                    presyn_neuron = self.spikes_presyn[0,in_row,in_col,channel_in,self.curr_iteration]
                                    if presyn_neuron == 1:
                                        self.counter_weakened +=1
                                        oldWeight = self.weights[w_row,w_col,channel_in,channel_output] 
                                        self.weights[w_row,w_col,channel_in,channel_output] += \
                                        self.a_minus * oldWeight * (1- oldWeight)
'''

