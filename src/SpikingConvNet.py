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
import csv
import math


tf.enable_eager_execution()

class SpikingConvNet(object):
    def __init__(self, start_from_scratch= True, classifierDatasetToAppend = None):
        encoding_t = 15
        path = dirname(dirname(realpath(__file__)))
        path_img1 = path + '/datasets/TrainingSet/Face/image_0297.jpg'
        dictImg1 = {
                'path':path_img1,
                'label':'Face'
                }
        path_img2 = path + '/datasets/TrainingSet/Face/image_0264.jpg'
        self.DoG = DoGwrapper.DoGwrapper(  [ dictImg1  ], encoding_t )
        self.start_from_scratch = start_from_scratch
        self.pathWeights = path + '/weights/'
        if start_from_scratch:
            self.classifierDataset_path = path + '/datasets/ClassifierSet/Data'+\
                time.strftime("%d_%m_%Y_%H_%M_%S")+ '.csv'
        else:
            self.classifierDataset_path = path + '/datasets/ClassifierSet/'+\
                    classifierDatasetToAppend


        strides_conv= [1,1,1,1]
        padding= "SAME"
        pooling_type= "MAX"
       
        self.layers = [
            Layer.ConvolutionalLayer(padding, strides_conv,
                [5,5,1,4],10., [1,160,250,1], [1,160,250,4],encoding_t),
            Layer.PoolingLayer(padding, [6,6], [7,7], pooling_type, [1,27,42,4]),
            Layer.ConvolutionalLayer(padding,strides_conv,
                [17,17,4,20], 50., [1,27,42,4], [1,27,42,20],encoding_t),
            Layer.PoolingLayer(padding, [5,5], [5,5], pooling_type, [1,6,9,20]),
            Layer.ConvolutionalLayer(padding, strides_conv,
                [5,5,20,20], math.inf , [1,6,9,20], [1,6,9,20],encoding_t)
            ]

    def lastMaxPooling( self, V, tensor_dim):
        [_,rows,columns,channels] = tensor_dim
        out_pool= tf.nn.pool(V,[rows,columns],"MAX","SAME", strides= [rows,columns])
        features = out_pool.numpy().reshape(channels)

        return features

    def writeFeatureNamesinDataset(self):
        with open(self.classifierDataset_path, 'a') as csvfile:
            fw=csv.writer(csvfile,delimiter=',',quotechar='|',quoting=csv.QUOTE_MINIMAL)
            feature_names = ['f1','f2','f3','f4','f5','f6','f7','f8','f9','f10',
                    'f11','f12','f13','f14','f15','f16','f17','f18','f19','f20','label']
            fw.writerow(feature_names)

    def writeFeaturesIntoDataset( self, features, label):
        with open(self.classifierDataset_path, 'a') as csvfile:
            fw=csv.writer(csvfile,delimiter=',',quotechar='|',quoting=csv.QUOTE_MINIMAL)
            feature_list =[]
            for f in features:
                feature_list.append( str(f))
            feature_list.append( label)
            fw.writerow( feature_list)
     
    def evolutionLoop( self):
        if self.start_from_scratch:
            self.writeFeatureNamesinDataset()
            for layer in self.layers:
                layer.createWeights()
        else:
            index = 0
            for layer in self.layers:
                layer.loadWeights( self.pathWeights, index )
                index+=1

        spikeTrains, label = self.DoG.getSpikeTrains()
           
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
                print("\n")

                # print( currSpikes_1layer[:,:,0])

 
        features = self.lastMaxPooling( self.layers[-1].oldPotentials, self.layers[-1].expected_output_dim )

        self.writeFeaturesIntoDataset( features,label)
        index = 0
        for layer in self.layers:
            layer.saveWeights(self.pathWeights, index)
            layer.resetLayer()
            index +=1



if __name__ == '__main__':
    start_from_scratch_FLAG = False
    dataset_name = "Data08_03_2019_12_41_38.csv"

    scn= SpikingConvNet()
  #   scn= SpikingConvNet( start_from_scratch_FLAG, dataset_name)
    scn.evolutionLoop()
 
