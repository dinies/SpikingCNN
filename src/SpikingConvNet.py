import tensorflow as tf
import DoGwrapper 
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
        self.padding_type= "SAME"
        self.pooling_type= "MAX"
        self.strides_conv= [1,1,1,1]





    def evolutionLoop( self):

        spikeTrains = self.DoG.getSpikeTrains()
        counter = 0

           
        for st in spikeTrains:
            oldPotentialsNp = np.zeros( [1, st.shape[0],st.shape[1],4])
            oldPotentials = tf.contrib.eager.Variable( oldPotentialsNp)

            currSpikesNp = np.zeros( [1, st.shape[0],st.shape[1],4])
            currSpikes= tf.contrib.eager.Variable( currSpikesNp)
        
               
            for i in range(st.shape[2]):
                dogSlice = st[:,:,i]
                reshapedDogSlice =  dogSlice.reshape( [1,dogSlice.shape[0],dogSlice.shape[1],1])
                input_img = tf.constant( reshapedDogSlice)
               

                # 1 conv layer 
                thresh_1 = 10
                filterNp = np.ones( [5,5,1,4])
                filter_input = tf.contrib.eager.Variable( filterNp)
                conv1_res = tf.nn.conv2d( input_img , filter_input,self.strides_conv, self.padding_type)



                # Spiking Logic plus STDP 
                newPotentials = tf.math.add( conv1_res, oldPotentials ) 

                for row in range(newPotentials.shape[0]):
                    for column in range(newPotentials.shape[1]):
                        for channel in range(newPotentials.shape[2]):
                            if newPotentials[0, row, column, channel ] >= thresh_1 :
                                counter += 1
                                currSpikes[0, row, column, channel ]= 1.0
                                newPotentials[0, row, column, channel ] = 0.0
                            else:
                                currSpikes[0, row, column, channel ]= 0.0

               
                # 1 pooling layer 
                window_shape = [7,7]
                stridePool = [6,6]
                pooling1_res = tf.nn.pool( currSpikes , window_shape, self.pooling_type, self.padding_type, strides = stridePool)
                print( pooling1_res.shape)
                print( counter )
                print( currSpikes[:,:,0])

 
                break

                


 


      

   


if __name__ == '__main__':

    scn= SpikingConvNet()
    scn.evolutionLoop()
 
