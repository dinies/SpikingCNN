from abc import ABC,abstractmethod
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np
from numba import jit

class Layer( ABC):

    def __init__(self, padding_type, stride, expected_output_dim ):
        self.padding_type = padding_type
        self.stride = stride
        self.expected_output_dim = expected_output_dim

    @abstractmethod
    def makeOperation( self, input_to_layer):
        pass


class ConvolutionalLayer(Layer):

    def __init__(self, padding_type, stride, filter_dim, threshold_potential, expected_input_dim, expected_output_dim, encoding_t):
        super().__init__( padding_type, stride, expected_output_dim)
        self.filter_dim = filter_dim
        self.threshold_potential = threshold_potential
        self.weights = np.ones( filter_dim)
        self.oldPotentials = tfe.Variable( np.zeros( expected_output_dim ))
        self.K_inh = np.ones(( expected_output_dim[1], expected_output_dim[2])).astype(np.uint8)
        self.encoding_t = encoding_t
        self.spikes_presyn = np.zeros(  expected_input_dim[1:4]+[self.encoding_t])
        self.spikes_postsyn = np.zeros(  expected_output_dim[1:4]+[self.encoding_t])
        self.curr_iteration = 0
        self.expected_input_dim = expected_input_dim


    def resetStoredData( self):
        self.curr_iteration = 0
        self.spikes_presyn = np.zeros(  self.expected_input_dim[1:4]+[self.encoding_t])
        self.spikes_postsyn = np.zeros(  self.expected_output_dim[1:4]+[self.encoding_t])
        
    def resetOldPotentials( self):
        self.oldPotentials = tfe.Variable( np.zeros( self.expected_output_dim ))

    def resetInhibition( self):
        self.K_inh = np.ones(( self.expected_output_dim[1], self.expected_output_dim[2])).astype(np.uint8)


    def makeOperation( self, input_to_layer):
        input_filter = tfe.Variable( self.weights )
        out_conv = tf.nn.conv2d(input_to_layer,input_filter,self.stride,self.padding_type)


        # Spiking Logic plus STDP 
        currSpikesNp = np.zeros( self.expected_output_dim)
        newPotentialsNp = tf.math.add( out_conv , self.oldPotentials).numpy()

        for row in range(newPotentialsNp.shape[1]):
            for column in range(newPotentialsNp.shape[2]):
                for channel in range(newPotentialsNp.shape[3]):
                    if newPotentialsNp[0,row,column,channel] >= self.threshold_potential:
                        currSpikesNp[0, row, column, channel ] = 1.0
                        newPotentialsNp[0, row, column, channel ] = 0.0

        
        S, K_inh = self.lateral_inh_CPU( currSpikesNp, newPotentialsNp, self.K_inh)
        self.K_inh = K_inh

        self.spikes_presyn[:,:,:,self.curr_iteration] = input_to_layer
        self.spikes_postsyn[:,:,:,self.curr_iteration] = S

        self.STDP_learning()
                    
        currSpikes = tfe.Variable( S )
        newPotentials = tfe.Variable( newPotentialsNp )

        self.oldPotentials.assign( newPotentials)

        return currSpikes

    def computeDeconvolutionIndexes( self, row, column):
        top_left = [ row, column]
        bottom_right = [ row + self.filter_dim[0]-1, column + self.filter_dim[1]-1]
        return top_left, bottom_right
        
    def STDP_learning( self):
        for row in range(self.spikes_postsyn.shape[0]):
            for column in range(self.spikes_postsyn.shape[1]):
                for channel_output in range(self.spikes_postsyn.shape[2]):
                    if self.spikes_postsyn[row,column,channel_output,self.curr_iteration] == 1:
                        top_left, bottom_right = self.computeDeconvolutionIndexes(row, column)
                        for m in range(top_left[0], bottom_right[0]+1):
                            for n in range(top_left[1], bottom_right[1]+1):
                                for channel_input in range(self.spikes_postsyn.shape[3]):
                                    # strenghten synapsis 

                                    for t_input in range( self.curr_iteration+1):
                                        presyn_neuron = self.spikes_presyn[ m, n,channel_input, t_input]
                                        if presyn_neuron ==1:
                                            self.weights[ m , n, channel_input] += self.a_plus* self.weights[ m , n, channel_input] *(1- self.weights[ m , n, channel_input])

                # weaken synapsis  this goes in a second cycle maybe starting from there
                for t_output in range( self.curr_iteration+1):
 



    # BEGIN Function borrowed from the paper autors
    @jit
    def lateral_inh_CPU(self, S, V, K_inh):
        S_inh = np.ones(S.shape, dtype=S.dtype)
        K = np.ones(K_inh.shape, dtype=K_inh.dtype)
        for i in range(V.shape[1]):
            for j in range(V.shape[2]):
                for k in range(V.shape[3]):
                    flag = False
                    if S[0,i, j, k] != 1:
                        continue
                    if K_inh[i, j] == 0:
                        S_inh[0,i, j, k] = 0
                        continue
                    for kz in range(V.shape[3]):
                        if S[0,i, j, kz] == 1 and V[0,i, j, k] < V[0,i, j, kz]:
                            S_inh[0,i, j, k] = 0
                            flag = True
                    if flag:
                        continue
                    else:
                        K[i, j] = 0
        S *= S_inh
        K_inh *= K
        return S, K_inh
    # END Function borrowed from the paper autors

class PoolingLayer(Layer):

    def __init__(self,padding_type,stride,window_shape,pooling_type,expected_output_dim ):
        super().__init__( padding_type, stride, expected_output_dim)
        self.window_shape = window_shape
        self.pooling_type = pooling_type


    def makeOperation( self, input_to_layer):
        out_pool= tf.nn.pool(input_to_layer,self.window_shape,self.pooling_type,self.padding_type,strides = self.stride)

        return out_pool
