from responder import Responder
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import math

class Agent():
    def __init__(self, numHiddenLayerNeurons, learningRate, numberOfStates, numberOfActions):
        #These lines established the feed-forward part of the network. The agent takes a state and produces an action.

        #None is an alias for NP.newaxis. It creates an axis with length 1
        #So that our placesholder is a row vector with numberOfStates number of columns
        # Define input placeholder
        #self.state_in= tf.placeholder(shape=[dtype=tf.int32,shape=[None, 1], name="input_x")
        self.charIndex = tf.placeholder(dtype=tf.int32)
        self.state_in_OH = tf.to_float(slim.one_hot_encoding(self.charIndex, numberOfStates))
        self.state_in = tf.placeholder(shape=[None, numberOfStates], dtype=tf.float32, name="input_x")

        '''
        `fully_connected` creates a variable called `weights`, representing a fully
        connected weight matrix, which is multiplied by the `inputs` to produce a
        `Tensor` of hidden units. If a `normalizer_fn` is provided (such as
        `batch_norm`), it is then applied. Otherwise, if `normalizer_fn` is
        None and a `biases_initializer` is provided then a `biases` variable would be
        created and added the hidden units. Finally, if `activation_fn` is not `None`,
        it is applied to the hidden units as well.
        '''

        hidden = slim.fully_connected(self.state_in,numHiddenLayerNeurons,biases_initializer=None,activation_fn=tf.nn.relu)
        self.output = slim.fully_connected(hidden,numberOfActions,activation_fn=tf.nn.softmax,biases_initializer=None)
        self.chosenAction = tf.argmax(self.output,1)

        # We need to define the parts of the network needed for learning a policy
        #Tensor("Placeholder_1:0", shape=(?,), dtype=int32)
        self.reward_holder = tf.placeholder(shape=[None],dtype=tf.int32)
        self.action_holder = tf.placeholder(shape=[None],dtype=tf.int32)

        #Tensor("add:0", shape=(?,), dtype=int32)
        self.indexes = tf.range(0, tf.shape(self.output)[0]) * tf.shape(self.output)[1] + self.action_holder

        #Tensor("Gather:0", shape=(?,), dtype=float32)
        self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]), self.indexes)

        self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs) * tf.to_float(self.reward_holder))

        # specify the trainable variables for later updating
        tvars = tf.trainable_variables()
        self.gradient_holders = []
        for idx,var in enumerate(tvars):
            placeholder = tf.placeholder(tf.float32,name=str(idx)+'_holder')
            self.gradient_holders.append(placeholder)

        #compute the gradients
        self.gradients = tf.gradients(self.loss,tvars)

        optimizer = tf.train.AdamOptimizer(learning_rate=learningRate)
        self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders,tvars))

class mdpPolicyAgent(Responder):
    def __init__(self):
        Responder.__init__(self)
        self.batch_size = 50
        self.gamma=0.99
        update_frequency = 5
        self.createGraph()


    #Compute the discounted reward_signal
    #for i,val in enumerate(r) returns a list containing (counter, value) for each element of r
    #That list is used to compute a vector of values with length = length of r
    #Takes 1d float array of rewards and computes discounted reward
    #    e.g. f([1, 1, 1], 0.99) -> [1, 0.99, 0.9801]
    def discount_rewards(r, gamma=0.99):
        return np.array([val * (gamma ** i) for i, val in enumerate(r)])

    def createGraph(self):
        #Clear the default graph stack and reset the global default graph.
        #tf.reset_default_graph()

        # Placeholders for our observations, outputs and rewards
        #these are not tf placeholders
        self.xs = np.empty(0).reshape(0,1)
        self.ys = np.empty(0).reshape(0,1)
        self.rewards = np.empty(0).reshape(0,1)

        self.myAgent = Agent(numHiddenLayerNeurons=100, learningRate=self.learningRate, numberOfStates=self.numStates, numberOfActions=self.numActions)

        #Variable to call next(..) from the training generator. Calling the generator directly causes it to run from the start
        #self.learner = self.runNet()

    def getOutput(self):
      if self.resetCalled:
        try:
          next(self.learner)
        except StopIteration:
          print("completed a reset net in mdpPolicy.py")
        self.netWasReset = True
        self.resetCalled = False

      else:
        if self.netWasReset:
          self.netWasReset = False
          print("creating a new tf graph in mdpPolcy.py")
          self.createGraph()
          return next(self.learner)
        else:
          return next(self.learner)

#The None element of the shape corresponds to a variable-sized dimension.
numberOfStates = 10
state_in_test = tf.placeholder( dtype=tf.int32)
one_hot = slim.one_hot_encoding(state_in_test, numberOfStates)
value1 = tf.reshape(one_hot,[1, numberOfStates])
state_in_test1 = tf.placeholder(shape=[None, numberOfStates], dtype=tf.float32)
state = 3
newState = 8
#tf.reshape(newState,[None,1])


with tf.Session() as sess:
        #self.charIndex = tf.placeholder(dtype=tf.int32)
        #self.state_in_OH = tf.to_float(slim.one_hot_encoding(self.charIndex, numberOfStates))
        #self.state_in = tf.placeholder(shape=[None, numberOfStates], dtype=tf.float32, name="input_x")

    testAgent = mdpPolicyAgent()
    sess.run(tf.global_variables_initializer())

    #print(one_hot.shape)
    #print(sess.run(one_hot, feed_dict={state_in:[newState]}))
    #tf.reshape(one_hot,[1, numberOfStates])
    #tf.reshape(one_hot,[numberOfStates, 1])
    #tf.reshape(one_hot,[numberOfStates, None])
    value = sess.run(one_hot, feed_dict={state_in_test:[newState]}) #this works
    print(value)
    value2 = sess.run(value1, feed_dict={state_in_test:[newState]})
    print(value2.shape)
    #print(one_hot.shape) #unknown
    test3 = sess.run(state_in_test1, feed_dict={state_in_test1:value}) #this works
    print(test3.shape)
    #print(sess.run(state_in_test1, feed_dict={value1:value}))
    #print(sess.run(state_in_test1, feed_dict={value1:value}))
    #print(state_in.shape)
    #self.action = sess.run(myAgent.chosenAction,feed_dict={myAgent.charIndex:[self.currentState]})
    #print(sess.run(output))
    #value1 = sess.run(testAgent.myAgent.state_in_OH, feed_dict={testAgent.myAgent.charIndex:[newState]})
    #print(value1)
    #action = sess.run(testAgent.myAgent.state_in,feed_dict={testAgent.myAgent.state_in_OH:value1})
