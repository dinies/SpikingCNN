from src.Layer import *

class PoolingLayer(Layer):

    def __init__(self,padding_type,stride,window_shape,pooling_type,expected_output_dim ):
        super().__init__( padding_type, stride, expected_output_dim)
        self.window_shape = window_shape
        self.pooling_type = pooling_type


    def makeOperation( self, input_to_layer):
        out_pool= tf.nn.pool(input_to_layer,self.window_shape,self.pooling_type,self.padding_type,strides = self.stride)

        return out_pool

    def resetLayer(self):
        pass

    def createWeights(self):
        pass

    def loadWeights(self,path,layer_index):
        pass

    def saveWeights(self,path,layer_index):
        pass
 
    def getSynapseChangeInfo(self):
        return -1, -1

    def getWeightsStatistics( self):
        return np.zeros( [10,1])
