
import tensorflow as tf
tf.reset_default_graph()

a = tf.constant(2)
b = tf.constant(3)
c = tf.add(a,b)

with tf.Session() as sess:

    writer  = tf.summary.FileWriter('./graphs', sess.graph)

    print(sess.run(c))


