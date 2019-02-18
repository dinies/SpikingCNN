import tensorflow as tf
import matplotlib.pyplot as plt



tf.enable_eager_execution()
simTime = 10
deltaT= 0.1
steps = range(int(simTime/deltaT))
data =[]

graph = tf.Graph()
with graph.as_default(): 

    x = tf.Variable( tf.zeros(1), name='q')
    u = tf.placeholder(tf.float32,name='u')
    op =  tf.assign( x, tf.add( x , u))

    with tf.Session(graph= graph) as sess:


        sess.run(tf.global_variables_initializer())
        for t in steps:

            feed = { u: 0.1} 
            x =  sess.run( op, feed_dict= feed)
            data.append( x )


plt.plot( steps, data )
plt.show()


