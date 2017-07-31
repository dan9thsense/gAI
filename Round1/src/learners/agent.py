# this network finds the best action for a particular state, but it only uses immediate rewards, no discounted ones

from responder import Responder
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

class Agent():
    def __init__(self, learningRate, numberOfStates, numberOfActions):
        #These lines established the feed-forward part of the network. The agent takes a state and produces an action.
        self.state_in= tf.placeholder(shape=[1],dtype=tf.int32)
        state_in_OH = slim.one_hot_encoding(self.state_in, numberOfStates)
        output = slim.fully_connected(state_in_OH, numberOfActions,\
            biases_initializer=None,activation_fn=tf.nn.sigmoid, weights_initializer=tf.ones_initializer())
        output = tf.reshape(output,[-1])
        self.chosen_action = tf.argmax(output,0)

        #The next six lines establish the training proceedure. We feed the reward and chosen action into the network
        #to compute the loss, and use it to update the network.
        self.reward_holder = tf.placeholder(shape=[1],dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[1],dtype=tf.int32)
        self.responsible_weight = tf.slice(output,self.action_holder,[1])
        self.loss = -(tf.log(self.responsible_weight)*self.reward_holder)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learningRate)
        self.update = optimizer.minimize(self.loss)

class simpleAgent(Responder):
    def __init__(self):
      Responder.__init__(self)

    def createGraph(self):

      #Clear the default graph stack and reset the global default graph.
      tf.reset_default_graph()

      #Establish the training proceedure. We feed the reward and chosen action into the network
      #to compute the loss, and use it to update the network.

      self.myAgent = Agent(learningRate=self.learningRate,numberOfStates=self.numStates,numberOfActions=self.numActions)

      #The weights we will evaluate to look into the network., Returns all variables created with `trainable=True
      self.weights = tf.trainable_variables()[0]

      #Variable to call next(..) from the training generator. Calling the generator directly causes it to run from the start
      self.learner = self.runNet()

    def getOutput(self):
      if self.resetCalled:
        try:
          next(self.learner)
        except StopIteration:
          print("completed a reset net in agent.py")
        self.netWasReset = True
        self.resetCalled = False

      else:
        if self.netWasReset:
          self.netWasReset = False
          print("creating a new tf graph in agent.py")
          self.createGraph()
          return next(self.learner)
        else:
          return next(self.learner)

    def runNet(self):
      # Launch the tensorflow graph
      sess = tf.Session()
      sess.run(tf.global_variables_initializer())
      while True:
         # first check if the call came from reset
        if self.resetCalled:
          # reset the weights
          print("reset called in agent.py, exiting tf session")
          sess.close()
          return

        if self.resetVariables:
          # re-initialize values, but keep the tree structure
          sess.run(tf.global_variables_initializer())
          self.resetVariables = False

        # include a chance to pick a random action
        if np.random.rand(1) < self.e:
          self.action = np.random.randint(self.numActions)
        else:
          # for this state, pick the action with the highest weight
          self.action = sess.run(self.myAgent.chosen_action,feed_dict={self.myAgent.state_in:[self.currentState]})

        yield self.characters[self.action]

        # we freeze here
        # while frozen, the output is sent, a reward is received
        # and a new state received, which becomes the current state

        #we now have a new current state as well as the reward based on the action we took in the previous state
        #Update the network
        feed_dict={self.myAgent.reward_holder:[self.reward],self.myAgent.action_holder:[self.action],self.myAgent.state_in:[self.previousState]}
        _,ww = sess.run([self.myAgent.update, self.weights], feed_dict=feed_dict)
