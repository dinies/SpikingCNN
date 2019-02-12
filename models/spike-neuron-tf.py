import tensorflow as tf
tf.enable_eager_execution()

graph = tf.Graph()

x = tf.zeros( [10,1])
x += 1

print(x)

V_old = tf.Variable(0.0)
V_curr = tf.Variable(0.0)
V_tresh = tf.constant(2.0)
W = tf.Variable(0.8)



