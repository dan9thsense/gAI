# this network finds the best action for a particular state, but it only uses immediate rewards, no discounted ones

from responder import Responder
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

state_in_OH = [1,0,0]

outputVector = slim.fully_connected(state_in_OH, 3,\
    biases_initializer=None,activation_fn=None, \
    weights_initializer=tf.ones_initializer(), scope='layer1')
output = tf.reshape(outputVector,[-1])
chosen_action = tf.argmax(self.output,0) #index of the largest value in output

loss = -(tf.log(np.amax(output)*reward)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
update = optimizer.minimize(loss)

tf.reset_default_graph()
sess = tf.Session()
action = sess.run(chosen_action)
_ = sess.run(update)
print(action)
sess.close()
