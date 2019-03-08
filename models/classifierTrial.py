import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from os.path import dirname, realpath

tf.enable_eager_execution()


sub_path = dirname(dirname(realpath(__file__)))
pathTrainDataset = sub_path + '/models/datasets/'+'dummyTrain.csv'
train = pd.read_csv( pathTrainDataset)
train_X, train_y = train, train.pop('labels')

pathTestDataset = sub_path + '/models/datasets/'+'dummyTest.csv'
test= pd.read_csv( pathTestDataset)
test_X, test_y = test, test.pop('labels')


def input_fun( features, labels, batch_size,repeat_count ):
    dataset = tf.data.Dataset.from_tensor_slices( (dict(features),labels))
    dataset = dataset.shuffle(10).repeat(repeat_count).batch( batch_size)
    return dataset
    


classifier = tf.estimator.BaselineClassifier(n_classes = 2)
classifier.train( input_fn = lambda : input_fun( train_X,train_y, 2, 2))
loss = classifier.evaluate( input_fn = lambda : input_fun( test_X,test_y, 2, 2))["loss"]
print(loss)


        
