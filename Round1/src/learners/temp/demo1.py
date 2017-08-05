
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import math

state = 3
newState = 8

numberOfStates = 10
numHiddenLayerNeurons = 20
numberOfActions = 71

#These lines established the feed-forward part of the network. The agent takes a state and produces an action.
charIndex = tf.placeholder(dtype=tf.int32)
state_in_OH = tf.to_float(slim.one_hot_encoding(charIndex, numberOfStates))
#need to reshape from (numberOfStates,) to (1,numberOfStates) to feed into state_in (?, numberOfStates)
state_in_rs = tf.reshape(state_in_OH, [1, numberOfStates])

#None is an alias for NP.newaxis. It creates an axis with length 1
#So that our placesholder is a row vector with numberOfStates number of columns
#The None element of the shape corresponds to a variable-sized dimension.
#we need that to feed in a variable number of states, based on the nuber of steps it takes before we get a reward
state_in= tf.placeholder(shape=[None,numberOfStates],dtype=tf.float32, name="input_x")



#state_in_test = tf.placeholder( dtype=tf.int32)
#one_hot = slim.one_hot_encoding(state_in_test, numberOfStates)
#value1 = tf.reshape(one_hot,[1, numberOfStates])
#state_in_test1 = tf.placeholder(shape=[None, numberOfStates], dtype=tf.float32)



'''
`fully_connected` creates a variable called `weights`, representing a fully
connected weight matrix, which is multiplied by the `inputs` to produce a
`Tensor` of hidden units. If a `normalizer_fn` is provided (such as
`batch_norm`), it is then applied. Otherwise, if `normalizer_fn` is
None and a `biases_initializer` is provided then a `biases` variable would be
created and added the hidden units. Finally, if `activation_fn` is not `None`,
it is applied to the hidden units as well.
'''

hidden = slim.fully_connected(state_in,numHiddenLayerNeurons,biases_initializer=None,activation_fn=tf.nn.relu)
output = slim.fully_connected(hidden,numberOfActions,activation_fn=tf.nn.softmax,biases_initializer=None)
#chosenAction = tf.argmax(output,1)


#myAgent = Agent(numHiddenLayerNeurons=100, learningRate=learningRate, numberOfStates=numStates, numberOfActions=numActions)

#Variable to call next(..) from the training generator. Calling the generator directly causes it to run from the start
#learner = runNet()



'''
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
        #charIndex = tf.placeholder(dtype=tf.int32)
        #state_in_OH = tf.to_float(slim.one_hot_encoding(charIndex, numberOfStates))
        #state_in = tf.placeholder(shape=[None, numberOfStates], dtype=tf.float32, name="input_x")

    test1 = sess.run(state_in_rs, feed_dict={charIndex:1})
    print(test1)
    test2 = sess.run(state_in, feed_dict={state_in:test1})
    print(test2)
    a_dist = sess.run(output,feed_dict={state_in:test1})
    print("a_dist = ", a_dist)
    a = np.random.choice(a_dist[0],p=a_dist[0])
    print("a = ", a)
    a = np.argmax(a_dist == a)
    print("a = ", a)
    #print(one_hot.shape)
    #print(sess.run(one_hot, feed_dict={state_in:[newState]}))
    #tf.reshape(one_hot,[1, numberOfStates])
    #tf.reshape(one_hot,[numberOfStates, 1])
    #tf.reshape(one_hot,[numberOfStates, None])
    #value = sess.run(one_hot, feed_dict={charIndex:[newState]}) #this works
    #print(value)
    #value2 = sess.run(value1, feed_dict={charIndex:[newState]})
    #print(value2.shape)
    #print(one_hot.shape) #unknown
    #test3 = sess.run(charIndex1, feed_dict={charIndex1:value}) #this works
    #print(test3.shape)
    #print(sess.run(charIndex1, feed_dict={value1:value}))
    #print(sess.run(charIndex1, feed_dict={value1:value}))
    #print(state_in.shape)
    #action = sess.run(myAgent.chosenAction,feed_dict={myAgent.charIndex:[currentState]})
    #print(sess.run(output))
    #value1 = sess.run(testAgent.myAgent.state_in_OH, feed_dict={testAgent.myAgent.charIndex:[newState]})
    #print(value1)
    #action = sess.run(testAgent.myAgent.state_in,feed_dict={testAgent.myAgent.state_in_OH:value1})
    '''
