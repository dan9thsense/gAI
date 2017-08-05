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
      self.variable_statistics(self.weights)

      #Variable to call next(..) from the training generator. Calling the generator directly causes it to run from the start
      self.learner = self.runNet()

    def getOutput(self):
      if self.done:
        try:
          next(self.learner)
        except StopIteration:
          print("completed a reset net in agent.py")
        self.netWasReset = True
        self.done = False

      else:
        if self.netWasReset:
          self.netWasReset = False
          print("creating a new tf graph in agent.py")
          self.createGraph()
          return next(self.learner)
        else:
          return next(self.learner)

    def variable_statistics(self, var):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope('summaries'):
          mean = tf.reduce_mean(var)
          tf.summary.scalar('mean', mean)
          with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
          tf.summary.scalar('stddev', stddev)
          tf.summary.scalar('max', tf.reduce_max(var))
          tf.summary.scalar('min', tf.reduce_min(var))
          tf.summary.histogram('histogram', var)

    def runNet(self):
      # Launch the tensorflow graph
      sess = tf.Session()
      sess.run(tf.global_variables_initializer())
      writer = tf.summary.FileWriter("output", sess.graph)
      actionTB = tf.summary.scalar("action", self.action)
      counter = 0
      while True:
        # first check if the call came from reset or done
        if self.done:
            print("done called in agent.py, exiting tf session")
            self.writer.flush()
            self.writer.close()
            self.sess.close()
            return

        if self.resetVariables:
            # re-initialize values, but keep the tree structure
            self.sess.run(tf.global_variables_initializer())
            self.resetVariables = False

        # include a chance to pick a random action
        #Reduce chance of random action as we train the model.
        if counter < 10000:
          counter += 1
          self.e = self.initialRandomActionProbability/((counter/50) + 10)

        if np.random.rand(1) < self.e:
          self.action = np.random.randint(self.numActions)
        else:
          # for this state, pick the action with the highest weight
          self.action, summaryActionTB = sess.run([self.myAgent.chosen_action, actionTB],feed_dict={self.myAgent.state_in:[self.currentState]})
          writer.add_summary(summaryActionTB)


        yield self.characters[self.action]

        # we freeze here
        # while frozen, the output is sent, a reward is received
        # and a new state received, which becomes the current state

        #we now have a new current state as well as the reward based on the action we took in the previous state
        #Update the network
        feed_dict={self.myAgent.reward_holder:[self.reward],self.myAgent.action_holder:[self.action],self.myAgent.state_in:[self.previousState]}
        _,ww = sess.run([self.myAgent.update, self.weights], feed_dict=feed_dict)
