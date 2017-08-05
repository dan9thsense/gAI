# this network finds the best action without using inputs, it just sends out actions and looks for rewards

from responder import Responder
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

class Agent():
  def __init__(self, learningRate, numberOfActions):
    #These lines established the feed-forward part of the network. The agent produces an action without regard to state
    self.actionWeights = tf.Variable(tf.ones([numberOfActions]), name='weights')
    self.chosen_action = tf.argmax(self.actionWeights,0, name='chosenAction')

    #The next six lines establish the training proceedure. We feed the reward and chosen action into the network
    #to compute the loss, and use it to update the network.
    self.reward_holder = tf.placeholder(shape=[1],dtype=tf.float32, name='reward_holder')
    self.action_holder = tf.placeholder(shape=[1],dtype=tf.int32, name='actionHolder')
    self.rewardTB = tf.placeholder(dtype=tf.int32)
    with tf.name_scope('respWeight'):
        self.responsible_weight = tf.slice(self.actionWeights,self.action_holder,[1])
    with tf.name_scope('loss'):
        self.loss = -(tf.log(self.responsible_weight)*self.reward_holder)
    with tf.name_scope('GradDescOptimizer'):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learningRate)
        self.update = optimizer.minimize(self.loss)

class NoStatePolicy(Responder):
  def __init__(self):
    Responder.__init__(self)

  def createGraph(self):
    #Clear the default graph stack and reset the global default graph.
    tf.reset_default_graph()

    #Establish the training proceedure. We feed the reward and chosen action into the network
    #to compute the loss, and use it to update the network.
    self.myAgent = Agent(learningRate=self.learningRate,numberOfActions=self.numActions)
    #self.variable_statistics(self.myAgent.loss)
    #The weights we will evaluate to look into the network., Returns all variables created with `trainable=True
    #self.weights = tf.trainable_variables()[0]
    tf.summary.histogram('weights_zeroed', 0.99 - self.myAgent.actionWeights)
    self.variable_statistics(self.myAgent.actionWeights)

    tf.summary.scalar("action_q_no_state", self.myAgent.chosen_action)
    tf.summary.scalar("loss_q_no_state", self.myAgent.loss[0])
    tf.summary.scalar("reward_q_no_state", self.myAgent.rewardTB)

    #Variable to call next(..) from the training generator. Calling the generator directly causes it to run from the start
    self.sess = tf.Session()
    self.sess.run(tf.global_variables_initializer())
    # Launch the tensorflow graph
    self.writer = tf.summary.FileWriter("output", self.sess.graph)
    self.merged = tf.summary.merge_all()

    self.learner = self.runNet()

  def variable_statistics(self, var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries_q_no_state'):
      mean = tf.reduce_mean(var)
      tf.summary.scalar('mean', mean)
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      tf.summary.scalar('stddev', stddev)
      tf.summary.scalar('max', tf.reduce_max(var))
      tf.summary.scalar('min', tf.reduce_min(var))
      tf.summary.histogram('histogram', var)

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

  def runNet(self):
    counter = 0
    while True:
      # first check if the call came from reset or done
      if self.done:
        print("done called in q_no_state_poicyWithTB.py, exiting tf session")
        self.writer.flush()
        self.writer.close()
        self.sess.close()
        return

      if self.resetVariables:
        # re-initialize values, but keep the tree structure
        self.sess.run(tf.global_variables_initializer())
        self.resetVariables = False

      # include a chance to pick a random action
      if np.random.rand(1) < self.e:
        self.action = np.random.randint(self.numActions)
        #placeholder  = tf.placeholder(dtype=tf.int32)
        #summaryActionTB = self.sess.run(placeholder, feed_dict={self.action})
      else:
        #pick the action with the highest weight
        self.action = self.sess.run(self.myAgent.chosen_action)

      yield self.characters[self.action]

      # we freeze here
      # while frozen, the output is sent, a reward is received

      #we now have a reward based on the action we took
      #Update the network
      update, respW, weights, loss = \
      self.sess.run([self.myAgent.update, self.myAgent.responsible_weight, self.myAgent.actionWeights, self.myAgent.loss],\
        feed_dict={self.myAgent.reward_holder:[self.reward],self.myAgent.action_holder:[self.action]})

      summaryFeeder = {self.myAgent.loss:loss, self.myAgent.chosen_action:self.action,\
        self.myAgent.actionWeights:weights, self.myAgent.rewardTB:self.reward}
      summaryMerged = self.sess.run(self.merged, feed_dict=summaryFeeder)
      self.writer.add_summary(summaryMerged, counter)
      if counter < 1000000:
        counter += 1
