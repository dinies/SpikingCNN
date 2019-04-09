import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from os.path import dirname, realpath
from sklearn.svm import SVC, LinearSVC
from sklearn import linear_model, preprocessing
from sklearn.neighbors import KNeighborsClassifier

tf.enable_eager_execution()


sub_path = dirname(dirname(realpath(__file__)))
 # pathTrainDataset = sub_path + '/models/datasets/dummyTrain.csv'
pathTrainDataset = sub_path + '/datasets/ClassifierSet/TrainingData.csv'
train = pd.read_csv( pathTrainDataset)
train_X, train_y = train, train.pop('label')

#pathTestDataset = sub_path + '/models/datasets/dummyTest.csv'
pathTestDataset = sub_path + '/datasets/ClassifierSet/TestingData.csv'
test= pd.read_csv( pathTestDataset)
test_X, test_y = test, test.pop('label')

scaler = preprocessing.StandardScaler()
train_X_scaled = scaler.fit_transform( train_X)
test_X_scaled = scaler.fit_transform( test_X)


classifer = SVC()
classifer.fit( train_X_scaled, train_y)
print( "svm score: \n")
print( classifer.score( test_X_scaled, test_y))


lin = linear_model.LogisticRegression()
lin.fit(train_X_scaled, train_y)
print( "linear model score: \n")
print(lin.score(test_X_scaled, test_y))

linSVC = LinearSVC(C=8.0)
linSVC.fit(  train_X_scaled, train_y)
print( "LinearSVC score: \n")
print( linSVC.score( test_X_scaled, test_y))


def input_fun( features, labels, batch_size,repeat_count ):
    dataset = tf.data.Dataset.from_tensor_slices( (dict(features),labels))
    dataset = dataset.shuffle(10).repeat(repeat_count).batch( batch_size)
    return dataset
