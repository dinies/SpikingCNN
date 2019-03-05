import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import numpy.random
import pdb

tf.enable_eager_execution()

class SpikingNeuron(object):
    def __init__(
            self,
            spike_train_list,
            connections_type = None,
            P_initial = - 3.0
            ):

        self.rest_potential = P_initial
        self.spike_train_list = spike_train_list
        self.input_channels = len( spike_train_list)
        self.Potential = tf.contrib.eager.Variable( P_initial)
        self.oldPotential = tf.contrib.eager.Variable( P_initial)
        self.Decay = tf.constant( 2.0)
        self.P_data=[]
        self.spikes_data= []
        self.threshold_P = 20.0
        self.W = self.initialize_weigths(len(spike_train_list), connections_type )


    def initialize_weigths( self, connections_num, connections_type):
        excitatory_synapse = 1.2
        inhibitory_synapse = -0.6

        w = []
        if connections_type:
            for i in range(connections_num):

                if connections_type[i] == 1:
                    w.append( excitatory_synapse )
                else:
                    w.append( inhibitory_synapse )
        else:
            for i in range(connections_num):
                if numpy.random.rand() >= 0.45:
                    w.append( excitatory_synapse )
                else:
                    w.append( inhibitory_synapse )

        return tf.contrib.eager.Variable( w )
    

    def forward_pass( self ):

        index = 0
        sum_term = tf.contrib.eager.Variable( 0.0 )
        shortened_train_list = []

        for spike_train in self.spike_train_list:
            inputSignal = spike_train[-1]
            new_train = spike_train[:-1].copy()
            shortened_train_list.append( new_train)

            tf.assign_add( sum_term, tf.multiply( inputSignal, self.W[index]))
            index += 1
            
        self.spike_train_list = shortened_train_list

        tf.assign_sub( sum_term, self.Decay)
        tf.assign_add( sum_term, self.oldPotential)
        tf.assign( self.Potential , sum_term)

        self.P_data.append( self.Potential.numpy())
        if self.Potential.numpy() > self.threshold_P:
            tf.assign( self.Potential, self.rest_potential)
            output = 1
        elif self.Potential.numpy() < self.rest_potential:
            tf.assign( self.Potential, self.rest_potential)
            output = 0
        else:
            output = 0

        tf.assign( self.oldPotential, self.Potential)
        return output


    def  evolution_loop(self):

        while self.spike_train_list and self.spike_train_list[0].shape[0] > 0 :
            output = self.forward_pass()
            self.spikes_data.append( output)

        self.plot_data()
            
    def plot_data(self):

        steps = range( len( self.P_data))

        fig,(ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.5), tight_layout=True)
        ax1.plot( steps,self.P_data)
        ax2.plot( steps,self.spikes_data)

        print( self.W.numpy())
        plt.show()
        

if __name__ == '__main__':

    num_trains = 10
    trains = []

    for i in range(num_trains) : 
        spike_train = numpy.random.poisson( size = 350 )
        trains.append( spike_train)


    n= SpikingNeuron( trains )
    n.evolution_loop()
 
