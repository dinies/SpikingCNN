import tensorflow as tf
import tensorflow.contrib.eager as tfe
import DoGwrapper 
import Layer
import matplotlib.pyplot as plt
import numpy as np
import numpy.random
import random
import time
from os.path import dirname, realpath
import pdb
import csv
import math
import os

tf.enable_eager_execution()

class SpikingConvNet(object):
    def __init__(self,  phase, start_from_scratch= False ):
        encoding_t = 15
        self.encoding_t = encoding_t
        self.phase = phase
        path = dirname(dirname(realpath(__file__)))
        self.path_to_img_folder = path + '/datasets/'+ phase + 'Set/'
        classifier_dataset_path = path + '/datasets/ClassifierSet/'
        self.classifier_training_dataset_path = classifier_dataset_path +'TrainingData.csv'
        self.classifier_testing_dataset_path = classifier_dataset_path +'TestingData.csv'
        self.start_from_scratch = start_from_scratch
        self.path_to_log_file = path+ '/logs/log.csv'

        self.pathWeights = path + '/weights/'

        strides_conv= [1,1,1,1]
        padding= "SAME"
        pooling_type= "MAX"

        stdp_flag = phase == 'Learning'

        self.layers = [
            Layer.ConvolutionalLayer(padding, strides_conv,
                [5,5,1,4],10., [1,160,250,1], [1,160,250,4],
                encoding_t,.09,-.08,-.005, stdp_flag ),
            Layer.PoolingLayer(padding, [6,6], [7,7], pooling_type, [1,27,42,4]),
            Layer.ConvolutionalLayer(padding,strides_conv,
                [17,17,4,20], 50., [1,27,42,4], [1,27,42,20],
                encoding_t,.05,-.04,-.001, stdp_flag),
            Layer.PoolingLayer(padding, [5,5], [5,5], pooling_type, [1,6,9,20]),
            Layer.ConvolutionalLayer(padding, strides_conv,
                [5,5,20,20], math.inf , [1,6,9,20], [1,6,9,20],
                encoding_t,.05,-.04,-.001, stdp_flag)
            ]

        if start_from_scratch:
            self.writeFeatureNamesinDataset( classifier_dataset_path +'TrainingData.csv')
            self.writeFeatureNamesinDataset( classifier_dataset_path +'TestingData.csv')

    def lastMaxPooling( self, V, tensor_dim):
        [_,rows,columns,channels] = tensor_dim
        out_pool= tf.nn.pool(V,[rows,columns],"MAX","SAME", strides= [rows,columns])
        features = out_pool.numpy().reshape(channels)

        return features

    def writeFeatureNamesinDataset(self,path):
        with open(path, 'a') as csvfile:
            fw=csv.writer(csvfile,delimiter=',',quotechar='|',quoting=csv.QUOTE_MINIMAL)
            feature_names = ['f1','f2','f3','f4','f5','f6','f7','f8','f9','f10',
                    'f11','f12','f13','f14','f15','f16','f17','f18','f19','f20','label']
            fw.writerow(feature_names)

    def writeFeaturesIntoDataset( self, features, label):
        if self.phase == 'Training':
            dataset_path = self.classifier_training_dataset_path
        else:
            dataset_path = self.classifier_testing_dataset_path

        with open(dataset_path , 'a') as csvfile:
            fw=csv.writer(csvfile,delimiter=',',quotechar='|',quoting=csv.QUOTE_MINIMAL)
            feature_list =[]
            for f in features:
                feature_list.append( str(f))
            if label == 'Face':
                feature_list.append(0)
            else:
                feature_list.append(1)
            fw.writerow( feature_list)

    def getImgPaths( self, number_of_images ):
        img_dicts = [] 
        path_to_faces = self.path_to_img_folder+'FaceToDo/'
        path_to_motors = self.path_to_img_folder+'MotorToDo/'

        for i in range(number_of_images):
            img_chosen = False
            max_trials = 10
            curr_trial = 0
            while not img_chosen and curr_trial < max_trials :
                if numpy.random.randint(2) == 1 and os.listdir( path_to_faces):
                    img_name = random.choice(os.listdir( path_to_faces))
                    dictImg = {
                        'path': path_to_faces + img_name,
                        'name': img_name,
                        'label':'Face'
                    }                               

                elif os.listdir( path_to_motors):

                    img_name = random.choice(os.listdir( path_to_motors ))
                    dictImg = {
                        'path': path_to_motors + img_name ,
                        'name': img_name,
                        'label':'Motor'
                    }                               
                curr_trial +=1
                if dictImg not in img_dicts:
                    img_chosen = True
                    img_dicts.append( dictImg)

        return  img_dicts
     

    def moveImgInDoneFolder( self, img_dict):
        old_path = img_dict['path']
        if img_dict['label'] == 'Face':
            new_path = self.path_to_img_folder +'FaceDone/' + img_dict['name']
        else:
            new_path = self.path_to_img_folder +'MotorDone/' + img_dict['name']

        os.rename( old_path, new_path)

    def writeInLog( self, log_list):
        with open(self.path_to_log_file, 'a') as csvfile:
            fw=csv.writer(csvfile,delimiter=',',quotechar='|',quoting=csv.QUOTE_MINIMAL)
            fw.writerow( log_list )

    def getTotalWeightsStats( self):
        magnitude_vec = np.zeros( [10,1])
        index = 0
        for layer in self.layers:
            layer.loadWeights( self.pathWeights, index )
            magnitude_vec += layer.getWeightsStatistics()
            index+=1

        return magnitude_vec

    def evolutionLoop( self, target_number_of_images ):

        if self.start_from_scratch:
            for layer in self.layers:
                layer.createWeights()
        else:
            index = 0
            for layer in self.layers:
                layer.loadWeights( self.pathWeights, index )
                index+=1



        img_dicts = self.getImgPaths( target_number_of_images)
        if img_dicts:
            for img in img_dicts:
                log_list = [ self.phase, img['label'],img['name'] ]
                DoG = DoGwrapper.DoGwrapper( img ,self.encoding_t )
                st = DoG.getSpikeTrains()
           
               
                for i in range(st.shape[2]):
                    dogSlice = st[:,:,i]
                    reshapedDogSlice=dogSlice.reshape([1,dogSlice.shape[0],dogSlice.shape[1],1])
                    curr_input = tf.constant( reshapedDogSlice)
                    log_list.append( 'st'+str(i))
               
                    for layer in self.layers:
                        # In and out from the layer class are passed tf variables
                        start = time.time()
                        curr_input= layer.makeOperation( curr_input)
                        end = time.time()
                        log_list.append( str( round( end - start)))
                        strenghtned,weakened = layer.getSynapseChangeInfo()
                        if strenghtned >=0:
                            log_list.append( 's'+str(strenghtned))
                        if weakened>=0:
                            log_list.append( 'w'+str(weakened))

                if self.phase != 'Learning':
                    features = self.lastMaxPooling( self.layers[-1].oldPotentials,\
                            self.layers[-1].expected_output_dim )
                    self.writeFeaturesIntoDataset( features, img['label'])

                index = 0
                for layer in self.layers:
                    if self.phase == 'Learning':
                        layer.saveWeights(self.pathWeights, index)
                    layer.resetLayer()
                    index +=1

                self.moveImgInDoneFolder( img )

                self.writeInLog( log_list)
                print( img['name'])



if __name__ == '__main__':
    # start_from_scratch = True
    start_from_scratch = False
    number_of_images =  20
    phase = "Learning"
    # phase = "Training"
    # phase = "Testing"
    scn= SpikingConvNet( phase,start_from_scratch)
    scn.evolutionLoop( number_of_images)
 
