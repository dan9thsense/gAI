# this network finds the best action for a particular state, but it only uses immediate rewards, no discounted ones

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

time_window = 4
numberOfStates = 3
numberOfActions = 5
state_in= tf.placeholder(shape=[1,time_window],dtype=tf.int32, name='state_in')
state_in_OH = slim.one_hot_encoding(state_in, numberOfStates)

output = slim.fully_connected(state_in_OH, numberOfActions,\
    biases_initializer=None,activation_fn=None, \
    weights_initializer=tf.ones_initializer(), scope='layer1')

#print(state_in)
#print(state_in_OH)
#print(output)
#outputVector has shape (1,5), we want it to be (5,) i.e. go from [[1,2,3,4,5]] to [1,2,3,4,5]
#this also changes tf.shape from (2,) to (1,)
#output = tf.reshape(outputVector,[-1])

#(tf.argmax,0)finds the max index for each column, (tf.argmax,1)finds the max index for each row
#self.selected_output = tf.slice(self.output,self.action_holder,[1])
#here we have a chosen action for each time_window slot (each input)
chosen_action = tf.argmax(output,1) #index of the largest value in output
#use an action holder to get the shape and the format (int32) good for using with loss and optimizers
action_holder = tf.placeholder(shape=[1,numberOfActions],dtype=tf.int32, name='action')
#tf.slice(matrix,[startIndexDim0, startIndexDim1...], [numEntriesDim0, numEntriesDim1...]
#here, if output is just a row vector, we pull out a single entry [1], at the index in action_holder
#but when output has rows corresponding to time, we need multiple selected outputs
#selOutputSlicer = tf.placeholder(shape=[time_window], dtype=tf.int32)
#selOutputSlicer[0] = outputVector[0][0][action_holder[0][0]]
#selOutputSlicer[i] = outputVector[0][i][action_holder[0][i]]
selectedOutput = tf.slice(output[0][0],[action_holder[0][0]],[1])
#selectedOutput1 = tf.argmax(tf.transpose(output),0)

selectedOutput1 = tf.reduce_mean(tf.transpose(output), 0)
selectedTimeSlot = tf.argmax(selectedOutput1,1)
selectedOutput1 = tf.reshape(selectedOutput1, [-1])
chosen_action1 = tf.argmax(selectedOutput1)
#selectedOutput = selectedOutput1[chosen_action]
#selectedOutput = tf.slice(output[0][selectedTimeSlot],[action_holder[0][chosen_action]],[1])
#selectedOutput = tf.slice
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

inputValue = [ [0,0,2,1] ]
#print(inputValue.shape)
#print(inputValue)
#inputValueT = np.transpose(inputArray)
#print(inputValueT.shape)
#print(inputValueT)
#inputArray = np.array([ 0, 1, 2, 3 ]) # works when state_in placeholder has [4,1] shape
#inputTensor = tf.reshape(inputArray,[4,1]) # works when state_in placeholder has [4,1] shape
#inputValue = sess.run(inputTensor) # works when state_in placeholder has [4,1] shape

numRuns = 3
for i in range(numRuns):
    action = sess.run(chosen_action, feed_dict={state_in:inputValue})
    print(action)

    selOutput, selectedOutput1Value, lossValue, outputValue, tvarsValue, state_in_OHValue = sess.run([\
          selectedOutput, selectedOutput1, loss, output, tvars, state_in_OH],\
        feed_dict={state_in:inputValue, action_holder:action})
    #print(state_in_OHValue)
    #print(state_in_OHValue[0].shape)
    #print(tf.shape(state_in_OHValue))
    #xW = np.matmul(state_in_OHValue[0], tvarsValue[0])
    #print(xW)
    print('tvars = ', tvarsValue[0])

    print('outputValue shape = ', tf.shape(outputValue))
    print('outputValue = ', outputValue)
    print('selOutput = ', selOutput)
    print('selectedOutput1 = ', selectedOutput1Value)
    #outputChosenValue = outputValue[action]
    #print('output chosen value = ', outputChosenValue)
    print('chosen action = ', action)
    print('loss = ', lossValue)
    print('finished run ', i, 'updating for next run')
    print()
    _ = sess.run(update, feed_dict={state_in:inputValue, action_holder:action})

'''
print('after updating')

action = sess.run(chosen_action, feed_dict={state_in:inputValue})

selOutput, lossValue, outputValue, outputVectorValue, tvarsValue, state_in_OHValue = sess.run([\
    selectedOutput, loss, output, outputVector, tvars, state_in_OH],\
    feed_dict={state_in:inputValue, action_holder:[action]})
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

_ = sess.run(update, feed_dict={state_in:inputValue, action_holder:[action]})

print('after 2nd updating')

action = sess.run(chosen_action, feed_dict={state_in:inputValue})

selOutput, lossValue, outputValue, outputVectorValue, tvarsValue, state_in_OHValue = sess.run([\
    selectedOutput, loss, output, outputVector, tvars, state_in_OH],\
    feed_dict={state_in:inputValue, action_holder:[action]})
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

_ = sess.run(update, feed_dict={state_in:inputValue, action_holder:[action]})

print('after 3nd updating')

action = sess.run(chosen_action, feed_dict={state_in:inputValue})

selOutput, lossValue, outputValue, outputVectorValue, tvarsValue, state_in_OHValue = sess.run([\
    selectedOutput, loss, output, outputVector, tvars, state_in_OH],\
    feed_dict={state_in:inputValue, action_holder:[action]})
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
