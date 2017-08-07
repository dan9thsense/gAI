# this network uses a trained model to repeat inputs
from responder import Responder
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

class Agent():
    def __init__(self, learningRate, numberOfStates, numberOfActions):

        with tf.name_scope('input'):
            self.state_in= tf.placeholder(shape=[1],dtype=tf.int32, name='state_in')
            state_in_OH = slim.one_hot_encoding(self.state_in, numberOfStates)
            self.reward_holder = tf.placeholder(shape=[1],dtype=tf.float32, name='reward')

        outputVector = slim.fully_connected(state_in_OH, numberOfActions,\
            biases_initializer=None,activation_fn=tf.nn.sigmoid,\
                weights_initializer=tf.ones_initializer(), scope='layer1')

        with tf.name_scope('output'):
            self.action_holder = tf.placeholder(shape=[1],dtype=tf.int32, name='action')
            self.output = tf.reshape(outputVector,[-1])
            self.selected_output = tf.slice(self.output,self.action_holder,[1])
            self.chosen_action = tf.argmax(self.output,0) #index of the largest value in output

        with tf.name_scope('calculations'):
            self.loss = -(tf.log(self.selected_output)*self.reward_holder)
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learningRate)
            self.update = optimizer.minimize(self.loss)

class simpleAgent(Responder):
    def __init__(self):
        Responder.__init__(self)
        self.loadModels = True

    def createGraph(self):
        #Clear the default graph stack and reset the global default graph.
        tf.reset_default_graph()
        self.myAgent = Agent(learningRate=self.learningRate,numberOfStates=self.numStates,numberOfActions=self.numActions)
        self.model_saver = tf.train.Saver()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        # Restore latest checkpoint
        self.model_saver.restore(self.sess, tf.train.latest_checkpoint('saved_models/agent_repeater/.'))
        print('using saved_models/agent_repeater/. in agentMinimalTrainedRepeater')
        self.learner = self.runNet()

    def getOutput(self):
        if self.done:
            try:
                next(self.learner)
            except StopIteration:
                print("finished a session in agentMinimalTrainedRepeater.py")
            self.netWasReset = True
            self.done = False

        else:
            if self.netWasReset:
                self.netWasReset = False
                print("creating a new tf graph in agentMinimalTrainedRepeater.py")
                self.createGraph()
            return next(self.learner)

    def runNet(self):
        while True:
            if self.done:
                self.sess.close()
                return
            self.action = self.sess.run(self.myAgent.chosen_action, feed_dict={self.myAgent.state_in:[self.currentState]})
            yield self.characters[self.action]
