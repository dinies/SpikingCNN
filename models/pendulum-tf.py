import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np



simTime = 10
deltaT= 0.01
numSteps = int(simTime/deltaT)
steps = range(numSteps)
thetaData =[]
thetaDotData =[]

graph = tf.Graph()
with graph.as_default(): 

    state = tf.Variable(  [0.01,0.0], name='q')
    u = tf.placeholder(tf.float32,name='u')
    dt = tf.placeholder(tf.float32,name='dt')
    gravity = tf.constant( 9.81, name='g')
    mass = tf.constant( 4.0,name='m')
    length = tf.constant( 7.0,name='l')
    damping = tf.constant(0.00001,name='damping')
    dq = [
            state[1],
            - (gravity / length) * tf.math.sin( state[0]) + u - state[1]*damping
            ]


    tf.summary.scalar('curr u',u)
    tf.summary.scalar('dq1',dq[0])
    tf.summary.scalar('dq2',dq[1])
    tf.summary.scalar('q1',state[0])
    tf.summary.scalar('q2',state[1])
    update = tf.assign( state, tf.add( state, dq, name='update'),name='save')

    summaries = tf.summary.merge_all()


with tf.Session(graph= graph) as sess:

    writer  = tf.summary.FileWriter('./graphs', sess.graph)
    sess.run(tf.global_variables_initializer())

    inputTorque = np.zeros(numSteps)
    inputTorque[1] =0.0

    k= 0
    for t in steps:
        

        feed = { u: inputTorque[k], dt: deltaT} 
        k+=1
        newState , mergedSummaries=  sess.run( [update,summaries] , feed_dict= feed)

        thetaData.append( newState[0] )
        thetaDotData.append( newState[1] )
        writer.add_summary( mergedSummaries,k)



fig,(ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.5), tight_layout=True)
ax1.plot( steps,thetaData )
ax2.plot( steps,thetaDotData)
plt.show()
        
 
