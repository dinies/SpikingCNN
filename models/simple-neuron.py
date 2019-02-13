import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
tf.enable_eager_execution()

class Neuron( object):
    def __init__(
            self,
            W = 1.0,
            b = 1.0):

        self.W = tf.contrib.eager.Variable(W)
        self.b = tf.contrib.eager.Variable(b)

    def __call__(self, x):
        return self.W * x + self.b





class Trainer( object):
    def __init__(self):
        self.neuron = Neuron()
        self.num_examples = 1000
        self.W_truth = 2.0
        self.b_truth = 4.0
        self.learning_rate = 0.125
        self.inputs = tf.random_normal( shape=[ self.num_examples])
        self.noise = tf.random_normal( shape=[ self.num_examples])
        self.outputs = self.inputs * self.W_truth + self.b_truth + self.noise

    def loss(self, predicted_y, truth_y):
        return tf.reduce_mean( tf.square(predicted_y - truth_y))



    def train(self):
        with tf.GradientTape() as tape:
            curr_loss = self.loss( self.neuron( self.inputs), self.outputs)
        dW, db = tape.gradient( curr_loss, [self.neuron.W, self.neuron.b])
        self.neuron.W.assign_sub( self.learning_rate* dW)
        self.neuron.b.assign_sub( self.learning_rate* db)
        return curr_loss



    def __call__(self):
        Ws, bs = [], []
        epochs = range(10)

        fig,(ax1,ax2,ax3) = plt.subplots(1, 3, figsize=(17, 11.5), tight_layout=True)
        
        self.plot(ax1)
        for epoch in epochs:
            Ws.append( self.neuron.W.numpy())
            bs.append( self.neuron.b.numpy())
            curr_loss = self.train()
            print('Epoch %2d: W=%1.2f b=%1.2f, loss=%2.5f' 
                    %( epoch, Ws[-1], bs[-1],curr_loss))

        ax2.plot(epochs, Ws, 'r',
                epochs, bs, 'b')
        ax2.plot([self.W_truth]*len(epochs), 'r--',
                 [self.b_truth]*len(epochs), 'b--')
        ax2.legend( ['W','b','true W', 'true b'])


        self.plot(ax3)
        plt.show()



    def plot(self, fig_handle):
        fig_handle.scatter( self.inputs,self.outputs, c='b')
        fig_handle.scatter( self.inputs, self.neuron( self.inputs), c='r')


if __name__ == '__main__':
    t= Trainer()
    t()
 
        



