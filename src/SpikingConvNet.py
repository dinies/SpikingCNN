import tensorflow as tf
import tensorflow.contrib.eager as tfe
import DoGwrapper 
import Layer
import matplotlib.pyplot as plt
import numpy as np
import numpy.random
from os.path import dirname, realpath
import pdb

tf.enable_eager_execution()

class SpikingConvNet(object):
    def __init__(self):
        path = dirname(dirname(realpath(__file__)))
        path_img1 = path + '/datasets/TrainingSet/Face/image_0297.jpg'
        path_img2 = path + '/datasets/TrainingSet/Face/image_0264.jpg'
        self.DoG = DoGwrapper.DoGwrapper(  [ path_img1 ] )


        strides_conv= [1,1,1,1]
        padding= "SAME"
        pooling_type= "MAX"
       
        self.layers = [
            Layer.ConvolutionalLayer( padding , strides_conv , [5,5,1,4], 1 , [1,160,250,4]),
            Layer.PoolingLayer( padding, [6,6], [7,7], pooling_type, [1,27,42,4]),
            Layer.ConvolutionalLayer( padding, strides_conv, [17,17,4,20], 10, [1,27,42,20]),
            Layer.PoolingLayer( padding, [5,5], [5,5], pooling_type, [1,6,9,20]),
            Layer.ConvolutionalLayer( padding, strides_conv, [5,5,20,20], 60, [1,6,9,20])
            ]

        self.weights = [
                np.ones( self.layers[0].filter_dim),
                np.ones( self.layers[2].filter_dim),
                np.ones( self.layers[4].filter_dim)
                ]


        self.oldPotentials = [
            np.zeros( self.layers[0].expected_output_dim ),
            np.zeros( self.layers[2].expected_output_dim ),
            np.zeros( self.layers[4].expected_output_dim )
            ]

        
    def resetOldPotentials( self):
        for potentials in self.oldPotentials:
            potentials = np.zeros( potentials.shape )


    def evolutionLoop( self):

        spikeTrains = self.DoG.getSpikeTrains()
        counter = 0

           
        for st in spikeTrains:
            oldPotentials_1layer = tfe.Variable( self.oldPotentials[0] )
               
            for i in range(st.shape[2]):
                dogSlice = st[:,:,i]
                reshapedDogSlice =  dogSlice.reshape( [1,dogSlice.shape[0],dogSlice.shape[1],1])
                input_img = tf.constant( reshapedDogSlice)
               

                # 1 conv layer 
                thresh_1 = 1
                input_filter = tfe.Variable( self.weights[0])
                conv1_res = tf.nn.conv2d( input_img , input_filter , self.layers[0].stride , self.layers[0].padding_type)


                # Spiking Logic plus STDP 
                currSpikesNp_1layer = np.zeros( self.layers[0].expected_output_dim)
                newPotentialsNp_1layer = tf.math.add( conv1_res, oldPotentials_1layer ).numpy()

                for row in range(newPotentialsNp_1layer.shape[1]):
                    for column in range(newPotentialsNp_1layer.shape[2]):
                        for channel in range(newPotentialsNp_1layer.shape[3]):
                            if newPotentialsNp_1layer[0, row, column, channel ] >= thresh_1 :
                                counter += 1
                                currSpikesNp_1layer[0, row, column, channel ] = 1.0
                                newPotentialsNp_1layer[0, row, column, channel ] = 0.0
                            else:
                                currSpikesNp_1layer[0, row, column, channel ] = 0.0


                currSpikes_1layer = tfe.Variable( currSpikesNp_1layer)
                newPotentials_1layer = tfe.Variable( newPotentialsNp_1layer)
                oldPotentials_1layer.assign( newPotentials_1layer)

               
                # 1 pooling layer 
                curr_layer= self.layers[1]
                pooling1_res = tf.nn.pool( currSpikes_1layer , curr_layer.window_shape, curr_layer.pooling_type, curr_layer.padding_type, strides = curr_layer.stride)
                print( pooling1_res.shape)
                print( counter )
              #   print( currSpikes_1layer[:,:,0])

 
                break

                


 


      

   


if __name__ == '__main__':

    scn= SpikingConvNet()
    scn.evolutionLoop()
 
