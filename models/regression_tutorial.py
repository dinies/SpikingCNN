#
#
from __future__ import absolute_import, division, print_function

import pathlib

import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
dataset_path = keras.utils.get_file("auto-mpg.data", "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")


#print(dataset_path)

column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight','Acceleration', 'Model Year', 'Origin'] 
raw_dataset = pd.read_csv(dataset_path, names=column_names,
        na_values = "?", comment='\t',
        sep=" ", skipinitialspace=True)

dataset = raw_dataset.copy()
dataset.tail()

#clean the data 
dataset.isna().sum()
 
dataset= dataset.dropna()

origin = dataset.pop('Origin')
dataset['USA'] = (origin ==1)*1.0
dataset['Europe'] = (origin ==2)*1.0
dataset['Japan'] = (origin ==3)*1.0
dataset.tail()

#split in train and test
train_dataset = dataset.sample(frac= 0.8, random_state= 0)
test_dataset = dataset.drop(train_dataset.index)

#inspect the data
sns.pairplot(train_dataset[["MPG","Cylinders","Displacement","Weight"]],diag_kind="kde")

train_stats = train_dataset.describe()
train_stats.pop("MPG")
train_stats = train_stats.transpose()
print(train_stats)

#split features from labels
train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')

#normalize the data
def norm(x):
    return (x - train_stats['mean']) / train_stats['std']

normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)






