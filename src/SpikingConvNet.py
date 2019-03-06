import tensorflow as tf
import tensorflow.contrib.eager as tfe
import DoGwrapper 
import Layer
import matplotlib.pyplot as plt
import numpy as np
import numpy.random
import time
from os.path import dirname, realpath
import pdb

tf.enable_eager_execution()

class SpikingConvNet(object):
    def __init__(self):
        encoding_t = 15
        path = dirname(dirname(realpath(__file__)))
        path_img1 = path + '/datasets/TrainingSet/Face/image_0297.jpg'
        path_img2 = path + '/datasets/TrainingSet/Face/image_0264.jpg'
        self.DoG = DoGwrapper.DoGwrapper(  [ path_img1 ], encoding_t )


        strides_conv= [1,1,1,1]
        padding= "SAME"
        pooling_type= "MAX"
       
        self.layers = [
            Layer.ConvolutionalLayer(padding, strides_conv, [5,5,1,4],10., [1,160,250,1], [1,160,250,4],encoding_t),
            Layer.PoolingLayer(padding, [6,6], [7,7], pooling_type, [1,27,42,4]),
            Layer.ConvolutionalLayer(padding,strides_conv,[17,17,4,20], 50., [1,27,42,4], [1,27,42,20],encoding_t),
            Layer.PoolingLayer(padding, [5,5], [5,5], pooling_type, [1,6,9,20]),
            Layer.ConvolutionalLayer(padding, strides_conv, [5,5,20,20], math.inf , [1,6,9,20], [1,6,9,20],encoding_t)
            ]
     
    def evolutionLoop( self):

        spikeTrains = self.DoG.getSpikeTrains()
        counter = 0
           
        # for st in spikeTrains:
        st = spikeTrains[0]
               
        for i in range(st.shape[2]):
            dogSlice = st[:,:,i]
            reshapedDogSlice=dogSlice.reshape([1,dogSlice.shape[0],dogSlice.shape[1],1])
            curr_input = tf.constant( reshapedDogSlice)
               
            for layer in self.layers:
                # In and out from the layer class are passed tf variables
                start = time.time()
                curr_input= layer.makeOperation( curr_input)
                end = time.time()
                              
                print( curr_input.shape)
                print( end- start)

                # print( currSpikes_1layer[:,:,0])

 
        for layer in self.layers:
           layer.resetLayer()



if __name__ == '__main__':

    scn= SpikingConvNet()
    scn.evolutionLoop()
 
