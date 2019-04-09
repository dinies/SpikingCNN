from SpikingConvNet import *
import pandas as pd
from os.path import dirname, realpath
from sklearn.svm import SVC, LinearSVC
from sklearn import linear_model, preprocessing
import subprocess
import sys


if sys.argv[1]:
    execution_type= sys.argv[1]
else:
    execution_type= '1'


'''
Launching a script to clean all the folders
'''
sub_path = dirname(dirname(realpath(__file__)))
if execution_type == '1':
    subprocess.call([sub_path+'/scripts/reinitializeOnlyDataset.sh'])

if execution_type == '2':
    subprocess.call([sub_path+'/scripts/reinitializeOnlyWeights.sh'])

if execution_type == '3' :
    subprocess.call([sub_path+'/scripts/reinitialize.sh'])

print("\nScript output: ignore the 'No such file or directory' statements if present\n")

'''
Learning of the weights
'''

if execution_type =='3':
    start_from_scratch = True
    spike_net = SpikingConvNet( 'Learning', start_from_scratch )
    spike_net.evolutionLoop( 10 )


'''
Build training and testing datasets
'''

if execution_type != '1':
    spike_net = SpikingConvNet( 'Training', False)
    spike_net.evolutionLoop( 10 )

    spike_net = SpikingConvNet( 'Testing', False)
    spike_net.evolutionLoop( 10 )

'''
Classifier
'''

pathTrainDataset = sub_path + '/datasets/ClassifierSet/TrainingData.csv'
train = pd.read_csv( pathTrainDataset)
train_X, train_y = train, train.pop('label')

pathTestDataset = sub_path + '/datasets/ClassifierSet/TestingData.csv'
test= pd.read_csv( pathTestDataset)
test_X, test_y = test, test.pop('label')

scaler = preprocessing.StandardScaler()
train_X_scaled = scaler.fit_transform( train_X)
test_X_scaled = scaler.fit_transform( test_X)


classifer = SVC()
classifer.fit( train_X_scaled, train_y)

print( "\n\nCLASSIFIER SCORES:\n")
print( "svm score: \n")
print( classifer.score( test_X_scaled, test_y))


lin = linear_model.LogisticRegression()
lin.fit(train_X_scaled, train_y)
print( "linear model score: \n")
print(lin.score(test_X_scaled, test_y))

linSVC = LinearSVC(C=3.0)
linSVC.fit(  train_X_scaled, train_y)
print( "LinearSVC score: \n")
print( linSVC.score( test_X_scaled, test_y))


