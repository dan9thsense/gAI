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
        outputVector = slim.fully_connected(state_in_OH, numberOfActions,\
            biases_initializer=None,activation_fn=tf.nn.sigmoid, weights_initializer=tf.ones_initializer())
        self.output = tf.reshape(outputVector,[-1])
        self.chosen_action = tf.argmax(self.output,0)

        #The next six lines establish the training proceedure. We feed the reward and chosen action into the network
        #to compute the loss, and use it to update the network.
        self.reward_holder = tf.placeholder(shape=[1],dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[1],dtype=tf.int32)
        self.selected_weight = tf.slice(self.output,self.action_holder,[1])
        self.loss = -(tf.log(self.selected_weight)*self.reward_holder)
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

        #The weights we will evaluate to look into the network
        #tf.trainable... returns all variables created with trainable=True
        self.weights = tf.trainable_variables()[0]
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        if self.useTensorBoard:
            self.createTensorboardSummaries()

        #Variable to call next(..) from the training generator. Calling the generator directly causes it to run from the start
        self.learner = self.runNet()

    #Launch the tensorflow graph
    def createTensorboardSummaries(self):
        def variable_statistics(var):
             #Attach a lot of summaries to a Tensor (for TensorBoard visualization).
             with tf.name_scope('statistics_agent'):
                 mean = tf.reduce_mean(var)
                 tf.summary.scalar('mean', mean)
                 with tf.name_scope('stddev'):
                     stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
                 tf.summary.scalar('stddev', stddev)
                 tf.summary.scalar('max', tf.reduce_max(var))
                 tf.summary.scalar('min', tf.reduce_min(var))
                 tf.summary.histogram('histogram', var)
        #end of variable statistics

        variable_statistics(self.weights)
        tf.summary.histogram('weights_agent', 0.99 - self.weights)
        tf.summary.scalar('action_agent', self.myAgent.action_holder[0])
        tf.summary.scalar('selectedWeight_agent', self.myAgent.selected_weight[0])
        tf.summary.histogram("output_agent", self.myAgent.output)
        tf.summary.scalar("loss_agent", self.myAgent.loss[0])
        tf.summary.scalar("reward_agent", self.myAgent.reward_holder[0])

        self.merged = tf.summary.merge_all()
        # Launch the tensorflow graph
        self.writer = tf.summary.FileWriter("output/agent", self.sess.graph)
    #end of createTensorboardSummaries

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

    def runNet(self):
        counter = 0
        while True:
            # first check if the call came from reset or done
            if self.done:
                print("done called in agent.py, exiting tf session")
                if self.useTensorBoard:
                    self.writer.flush()
                    self.writer.close()
                self.sess.close()
                return

            if self.resetVariables:
                # re-initialize values, but keep the tree structure
                self.sess.run(tf.global_variables_initializer())
                self.resetVariables = False

            #Include a chance to pick a random action
            #Reduce chance of random action as we train the model.
            counter += 1
            self.e = self.initialRandomActionProbability/((counter/50) + 10)

            if np.random.rand(1) < self.e:
                self.action = np.random.randint(self.numActions)
                print("random action selected in agent")
            else:
                # for this state, pick the action with the highest weight
                self.action = self.sess.run(self.myAgent.chosen_action,\
                    feed_dict={self.myAgent.state_in:[self.currentState]})

            yield self.characters[self.action]
            # we freeze here
            # while frozen, the output is sent, a reward is received
            # and a new state received, which becomes the current state

            #we now have a new current state as well as the reward based on the action we took in the previous state
            #Update the network
            networkFeeder = {self.myAgent.reward_holder:[self.reward],self.myAgent.action_holder:[self.action],\
                self.myAgent.state_in:[self.previousState]}
            _, weights, loss, output, selW = self.sess.run([self.myAgent.update, self.weights, \
                self.myAgent.loss, self.myAgent.output, self.myAgent.selected_weight], feed_dict=networkFeeder)

            if self.useTensorBoard:
                summaryFeeder = {self.myAgent.reward_holder:[self.reward],self.myAgent.action_holder:[self.action],\
                self.weights:weights, self.myAgent.loss:loss, self.myAgent.output:output, \
                self.myAgent.selected_weight:selW }
                summaryMerged = self.sess.run(self.merged, feed_dict=summaryFeeder)
                self.writer.add_summary(summaryMerged, counter)
        #end while True
    #end def runNet(self)
 #end class simpleAgent
