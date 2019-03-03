from abc import ABC,abstractmethod
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np

class Layer( ABC):

    def __init__(self, padding_type, stride, expected_output_dim ):
        self.padding_type = padding_type
        self.stride = stride
        self.expected_output_dim = expected_output_dim

    @abstractmethod
    def makeOperation( self, input_to_layer):
        pass


class ConvolutionalLayer(Layer):

    def __init__(self, padding_type, stride, filter_dim, threshold_potential, expected_output_dim):
        super().__init__( padding_type, stride, expected_output_dim)
        self.filter_dim = filter_dim
        self.threshold_potential = threshold_potential
        self.weights = np.ones( filter_dim)
        self.oldPotentials = tfe.Variable( np.zeros( expected_output_dim ))

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

        currSpikes = tfe.Variable( currSpikesNp)
        newPotentials = tfe.Variable( newPotentialsNp )
        self.oldPotentials.assign( newPotentials)

        return currSpikes


class PoolingLayer(Layer):

    def __init__(self,padding_type,stride,window_shape,pooling_type,expected_output_dim ):
        super().__init__( padding_type, stride, expected_output_dim)
        self.window_shape = window_shape
        self.pooling_type = pooling_type


    def makeOperation( self, input_to_layer):
        out_pool= tf.nn.pool(input_to_layer,self.window_shape,self.pooling_type,self.padding_type,strides = self.stride)

        return out_pool
