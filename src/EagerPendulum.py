import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pdb
import MyMath

tf.enable_eager_execution()

class EagerPendulum( object):
    def __init__(
            self,
            initial_state = [ 0.0,0.0]
            ):

        self.state= tf.contrib.eager.Variable( initial_state )
        self.m= 4.0
        self.l= 7.0
        self.g= 9.81
        self.friction = 0.1
        self.simTime = 100
        self.dt= 0.01
        self.numSteps = int(self.simTime/self.dt)
        self.steps = range(self.numSteps)
        self.thetaData =[]
        self.thetaDotData =[]


    def evolution( self, u ):

        dq = [
            self.state[1],
            - (self.g/ self.l) * tf.math.sin( self.state[0]) + u - self.state[1]*self.friction
        ]
        dt_vec = [ self.dt , self.dt ]
        increment_q = tf.multiply( dq, dt_vec)

        new_state = tf.contrib.eager.Variable(
                [
                    MyMath.MyMath.boxplus( self.state[0], increment_q[0].numpy()),
                    self.state[1] + increment_q[1]
                ])

        tf.assign( self.state, new_state)

    def execution_loop(self):

        inputTorque = np.zeros(self.numSteps)
        k= 0
        for t in self.steps:

            self.evolution( inputTorque[k])
            k+=1

            self.thetaData.append( self.state[0] )
            self.thetaDotData.append( self.state[1] )
        self.plot_data()



    def plot_data(self):

        fig,(ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.5), tight_layout=True)
        ax1.plot( self.steps,self.thetaData )
        ax2.plot( self.steps,self.thetaDotData)
        plt.show()
        

if __name__ == '__main__':
    p= EagerPendulum( [1.5,0.0])
    p.execution_loop()
 
