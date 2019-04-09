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
        
            input_plots = []
            output_plots = []
            for k in range( self.expected_output_dim[3]):
                in_k = input_to_layer_np[:,:,:,k].reshape([input_to_layer_np.shape[1],input_to_layer_np.shape[2]]) 
                out_k = output_from_layer_np[:,:,:,k].reshape([self.expected_output_dim[1],self.expected_output_dim[2]]) 
                input_plots.append( in_k)
                output_plots.append( out_k)
            '''
            2nd layer
            '''
       
            if  self.expected_output_dim == [1,160,250,4] :

                fig1, axes1 = plt.subplots(2, 4, figsize=(25, 10), tight_layout=True)

                axes1[0][0].imshow( input_plots[0])
                axes1[0][0].set_title('Input Pooling')
                axes1[0][0].axis('off')
                axes1[0][1].imshow( input_plots[1])
                axes1[0][1].axis('off')
                axes1[0][2].imshow( input_plots[2])
                axes1[0][2].axis('off')
                axes1[0][3].imshow( input_plots[3])
                axes1[0][3].axis('off')

                axes1[1][0].imshow( output_plots[0])
                axes1[1][0].set_title('Output Pooling')
                axes1[1][0].axis('off')
                axes1[1][1].imshow( output_plots[1])
                axes1[1][1].axis('off')
                axes1[1][2].imshow( output_plots[2])
                axes1[1][2].axis('off')
                axes1[1][3].imshow( output_plots[3])
                axes1[1][3].axis('off')
            '''
            2nd layer
            '''

            if  self.expected_output_dim == [1,27,42,20]:
                fig1, axes1 = plt.subplots(2, 4, figsize=(25, 10), tight_layout=True)

                axes1[0][0].imshow( input_plots[2])
                axes1[0][0].set_title('Input Pooling')
                axes1[0][0].axis('off')
                axes1[0][1].imshow( input_plots[7])
                axes1[0][1].axis('off')
                axes1[0][2].imshow( input_plots[12])
                axes1[0][2].axis('off')
                axes1[0][3].imshow( input_plots[18])
                axes1[0][3].axis('off')

                axes1[1][0].imshow( output_plots[2])
                axes1[1][0].set_title('Output Pooling')
                axes1[1][0].axis('off')
                axes1[1][1].imshow( output_plots[7])
                axes1[1][1].axis('off')
                axes1[1][2].imshow( output_plots[12])
                axes1[1][2].axis('off')
                axes1[1][3].imshow( output_plots[18])
                axes1[1][3].axis('off')


        return out_pool

    def resetLayer(self):
        pass

    def createWeights(self):
        pass

    def loadWeights(self,path,layer_index):
        pass

    def saveWeights(self,path,layer_index):
        pass
 
    def getIterationInfo(self):
        return -1, -1, -1, -1

    def getWeightsStatistics( self):
        return np.zeros( [10,1])
