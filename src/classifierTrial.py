import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from os.path import dirname, realpath

tf.enable_eager_execution()


sub_path = dirname(dirname(realpath(__file__)))
pathTrainDataset = sub_path + '/datasets/ClassifierSet/TrainingData.csv'
train = pd.read_csv( pathTrainDataset)
train_X, train_y = train, train.pop('label')

pathTestDataset = sub_path + '/datasets/ClassifierSet/TestingData.csv'
test= pd.read_csv( pathTestDataset)
test_X, test_y = test, test.pop('label')


def input_fun( features, labels, batch_size,repeat_count ):
    dataset = tf.data.Dataset.from_tensor_slices( (dict(features),labels))
    dataset = dataset.shuffle(10).repeat(repeat_count).batch( batch_size)
    return dataset
    
from sklearn import linear_model
lin = linear_model.LogisticRegression()
lin.fit(train_X, train_y)
print(lin.score(test_X, test_y))

      
from sklearn import svm
svm = svm.SVC()
svm.fit(train_X, train_y)
print(svm.score(test_X, test_y))