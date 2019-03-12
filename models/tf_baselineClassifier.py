import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from os.path import dirname, realpath

tf.enable_eager_execution()

# Warning, this class has now shown to be useful
# classifierTrial.py is the useful version
# the key is to use csv files and pandas to import them



def input_fun_train():
    dataset = DatasetCreator()
    X_train, y_train,_ ,_ =dataset.load_dataset()
    data = dataset.processNpinTfData( X_train, y_train)
    print(data)
    return data

def input_fun_test():
    dataset = DatasetCreator()
    _, _, X_test, y_test =dataset.load_dataset()
    data = dataset.processNpinTfData( X_test, y_test)
    return data


class MyClassifier( object):
    def __init__(self, name = "trial"):
        self.classifier = tf.estimator.BaselineClassifier(n_classes = 2)
        dataset = DatasetCreator()
        dataset.create()


    def __call__(self):
        self.train_classifier()
        return
        loss = self.evaluate_classifier()
        print("loss  " + str( loss))



    def train_classifier( self):
        self.classifier.train(input_fun_train)

    def evaluate_classifier( self):
        loss = self.classifier.evaluate(input_fun_test)["loss"]
        return loss
    







class DatasetCreator( object):
    def __init__(self,name = "dummy"):
        sub_path = dirname(dirname(realpath(__file__)))
        self.pathDataset = sub_path + '/models/'+ name
        self.num_training_samples = 300
        self.num_testing_samples = 100
        self.dim_samples = 10


    def create(self):
        X_train, y_train = self.create_random_dataset( self.num_training_samples)
        X_test, y_test = self.create_random_dataset( self.num_testing_samples)
        self.save_dataset( X_train, y_train, X_test, y_test)




    # def load(self):
        # X_train, y_train, X_test, y_test = self.load_dataset()
        # tf_Data_train = self.processNpinTfData( X_train, y_train)
        # tf_Data_test = self.processNpinTfData( X_test , y_test)
        # self.print_dataset_dimensions("Train", X_train, y_train)
        # self.print_dataset_dimensions("Test", X_test, y_test)

        
    def create_random_dataset( self, num_samples):
        X_t = None
        y_t = None

        for _ in range( num_samples):
            rand = np.random.randint(2)
            if rand:
                x_t= np.ones([1, self.dim_samples])
                y = np.zeros([1,1])
            else:
                x_t= np.zeros([1, self.dim_samples])
                y = np.ones([1,1])

            if X_t is not None and y_t is not None:
                X_t= np.append( X_t, x_t, axis= 0)
                y_t= np.append( y_t, y, axis= 0)
            else:
                X_t= x_t
                y_t= y

        return X_t, y_t

    def save_dataset( self, X_train, y_train, X_test, y_test):
        np.save( self.pathDataset + '_X_train.npy', X_train)
        np.save( self.pathDataset + '_y_train.npy', y_train)
        np.save( self.pathDataset + '_X_test.npy', X_test)
        np.save( self.pathDataset + '_y_test.npy', y_test)

    def load_dataset( self):
        X_train = np.load( self.pathDataset + '_X_train.npy')
        y_train = np.load( self.pathDataset + '_y_train.npy')
        X_test = np.load( self.pathDataset + '_X_test.npy')
        y_test = np.load( self.pathDataset + '_y_test.npy')

        return X_train, y_train, X_test, y_test

    def processNpinTfData(self, X,y):
        tf_Data = tf.data.Dataset.from_tensor_slices(( X, y))
        return tf_Data

    def print_dataset_dimensions(self, name, X, y):
        print( name + " : \n")
        print( "X dims:  "+  str( X.shape[0])+" "+str( X.shape[1])+"\n")
        print( "y dims:  "+  str( y.shape[0])+" "+str( y.shape[1])+"\n")
       

if __name__ == '__main__':
    # c= MyClassifier()
    # c()
    d = DatasetCreator()
    X, y, _, _=d.load_dataset()
    d.print_dataset_dimensions( "trial",X,y)
    data = d.processNpinTfData( X, y)
    print( data)

 
