# this network finds the best action for a particular state, but it only uses immediate rewards, no discounted ones

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

state_in= tf.placeholder(shape=[1],dtype=tf.int32, name='state_in')
state_in_OH = slim.one_hot_encoding(state_in, 3)

outputVector = slim.fully_connected(state_in_OH, 5,\
    biases_initializer=None,activation_fn=None, \
    weights_initializer=tf.ones_initializer(), scope='layer1')
#outputVector has shape (1,5), we want it to be (5,) i.e. go from [[1,2,3,4,5]] to [1,2,3,4,5]
#this also changes tf.shape from (2,) to (1,)
output = tf.reshape(outputVector,[-1])
chosen_action = tf.argmax(output,0) #index of the largest value in output
#use an action holder to get the shape and the format (int32) good for using with loss and optimizers
action_holder = tf.placeholder(shape=[1],dtype=tf.int32, name='action')
selectedOutput = tf.slice(output,action_holder,[1])
#selectedOutput1 = output[chosen_action]
#selectedOutput2 = np.array([output[chosen_action]])

reward = -1.
tvars = tf.trainable_variables()
loss = -tf.log(selectedOutput) * reward  # when we have a negative reward, we want this value to be positive
#gradients() outputs the partial derivatives of loss with respect to weights.
 #It returns a list of Tensor of length len(tvars) where each tensor is the sum(dloss/dvars) for loss in list of losses.
#gradients = tf.gradients(loss,tvars)
#optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
#update = optimizer.apply_gradients(zip(gradients,tvars))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
update = optimizer.minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
inputValue = 0
action = sess.run(chosen_action, feed_dict={state_in:[inputValue]})

lossValue, outputValue, outputVectorValue, tvarsValue, state_in_OHValue = sess.run([\
    loss, output, outputVector, tvars, state_in_OH],\
    feed_dict={state_in:[inputValue], action_holder:[action]})
#print(state_in_OHValue)
#print(state_in_OHValue[0].shape)
#print(tf.shape(state_in_OHValue))
Wx = np.matmul(np.transpose(tvarsValue[0]), state_in_OHValue[0])
print(Wx)
print(tf.shape(outputValue))
outputChosenValue = outputValue[action]
print(outputChosenValue)
print('loss = ', lossValue)
print('chosen action = ', action)

_ = sess.run(update, feed_dict={state_in:[inputValue], action_holder:[action]})

print('after updating')

action = sess.run(chosen_action, feed_dict={state_in:[inputValue]})

selOutput, lossValue, outputValue, outputVectorValue, tvarsValue, state_in_OHValue = sess.run([\
    selectedOutput, loss, output, outputVector, tvars, state_in_OH],\
    feed_dict={state_in:[inputValue], action_holder:[action]})
#print(state_in_OHValue)
#print(state_in_OHValue[0].shape)
#print(tf.shape(state_in_OHValue))
print(tvarsValue)
#Wx = np.matmul(np.transpose(tvarsValue[0]), state_in_OHValue[0])
xW = np.matmul(state_in_OHValue[0], tvarsValue[0])
print(xW)
print(outputValue)
outputChosenValue = outputValue[action]
print(outputChosenValue)
print('loss = ', lossValue)
print('chosen action = ', action)
print('selected output value = ', selOutput)
print('selected output value shape = ', selOutput)
oca = np.array([outputValue[action]])
print('output[chosen_action] = ', oca)
print('output[chosen_action].shape = ', oca.shape)

'''
print(tvarsValue)
b = np.array(tvarsValue)
print(b)
print(tf.shape(b))
print(tvarsValue[0])
print(tf.shape(tvarsValue[0]))
a = tvarsValue[0]
print(a)
print(a.shape)
'''
sess.close()
