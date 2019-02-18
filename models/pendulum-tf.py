import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np



class Pendulum( object):
    def __init__(self):
        self.initial_state = [ 0.1 , 0.0 ]
        self.mass = 4.0
        self.length = 7.0
        self.gravity = 9.81
        self.friction = 0.01
        self.simTime = 10
        self.deltaT= 0.01
        self.numSteps = int(self.simTime/self.deltaT)
        self.steps = range(self.numSteps)
        self.thetaData =[]
        self.thetaDotData =[]
        self.graph = tf.Graph()

        self.define_graph()

    def define_graph( self ):

        with self.graph.as_default(): 

            state = tf.Variable( self.initial_state , name='q')
            self.u = tf.placeholder(tf.float32,name='u')
            self.dt = tf.placeholder(tf.float32,name='dt')
            gravity = tf.constant( self.gravity, name='g')
            mass = tf.constant( self.mass,name='m')
            length = tf.constant( self.length,name='l')
            damping = tf.constant(self.friction,name='damping')
            dq = [
                state[1],
                - (gravity / length) * tf.math.sin( state[0]) + self.u - state[1]*damping
            ]
            dt_vec = [ self.dt , self.dt ]
            increment_q = tf.multiply( dq, dt_vec, name='dqxdt')
            update = tf.add( state,increment_q , name='update')
            self.update_op = tf.assign( state, update, name='save')

            tf.summary.scalar('curr u',self.u)
            tf.summary.scalar('dq1',dq[0])
            tf.summary.scalar('dq2',dq[1])
            tf.summary.scalar('q1',state[0])
            tf.summary.scalar('q2',state[1])
            self.summaries = tf.summary.merge_all()

    def run(self):

        with tf.Session(graph= self.graph) as sess:

            writer  = tf.summary.FileWriter('./graphs', sess.graph)
            sess.run(tf.global_variables_initializer())
            inputTorque = np.zeros(self.numSteps)
            inputTorque[1] =0.0
            k= 0
            for t in self.steps:

                feed = { self.u: inputTorque[k], self.dt: self.deltaT} 
                k+=1
                newState,mergedSummaries= sess.run([self.update_op,self.summaries],feed_dict= feed)

                self.thetaData.append( newState[0] )
                self.thetaDotData.append( newState[1] )
                writer.add_summary( mergedSummaries,k)
        self.plot_data()



    def plot_data(self):

        fig,(ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.5), tight_layout=True)
        ax1.plot( self.steps,self.thetaData )
        ax2.plot( self.steps,self.thetaDotData)
        plt.show()
        

if __name__ == '__main__':
    p= Pendulum()
    p.run()
 
