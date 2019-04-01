import context
from src.Layer import *

import matplotlib.pyplot as plt
class PoolingLayer(Layer):

    def __init__(self,padding_type,stride,window_shape,pooling_type,expected_output_dim ):
        super().__init__( padding_type, stride, expected_output_dim)
        self.window_shape = window_shape
        self.pooling_type = pooling_type


    def makeOperation( self, input_to_layer, flag_plots = False):

        out_pool= tf.nn.pool(input_to_layer,self.window_shape,self.pooling_type,self.padding_type,strides = self.stride)


        if flag_plots:
            input_to_layer_np = input_to_layer.numpy()
            output_from_layer_np = out_pool.numpy()
            '''
            1st pooling layer
            Input size [160,250,4]
            Output Size [27,42,4]
            TODO parametrize sizes for reshape
            '''

            input_plots = []
            output_plots = []
            for k in range( self.expected_output_dim[3]):
                in_k = input_to_layer_np[:,:,:,k].reshape([160,250]) 
                out_k = output_from_layer_np[:,:,:,k].reshape([27,42]) 
                input_plots.append( in_k)
                output_plots.append( out_k)


            fig1, axes1 = plt.subplots(2, 4, figsize=(25, 10), tight_layout=True)

            axes1[0][0].imshow( input_plots[0])
            axes1[0][0].set_title('Input Pooling')
            axes1[0][1].imshow( input_plots[1])
            axes1[0][2].imshow( input_plots[2])
            axes1[0][3].imshow( input_plots[3])

            axes1[1][0].imshow( output_plots[0])
            axes1[1][0].set_title('Output Pooling')
            axes1[1][1].imshow( output_plots[1])
            axes1[1][2].imshow( output_plots[2])
            axes1[1][3].imshow( output_plots[3])


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
