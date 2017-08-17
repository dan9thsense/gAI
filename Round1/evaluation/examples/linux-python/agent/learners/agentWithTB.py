# this network finds the best action for a particular state, but it only uses immediate rewards, no discounted ones

from responder import Responder
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

class Agent():
    def __init__(self, learningRate, numberOfStates, numberOfActions):
        #These lines established the feed-forward part of the network. The agent takes a state and produces an action.
        with tf.name_scope('input'):
            self.state_in= tf.placeholder(shape=[1],dtype=tf.int32, name='state_in')
            state_in_OH = slim.one_hot_encoding(self.state_in, numberOfStates)
            self.reward_holder = tf.placeholder(shape=[1],dtype=tf.float32, name='reward')

        #with tf.name_scope('net'):
        outputVector = slim.fully_connected(state_in_OH, numberOfActions,\
            biases_initializer=None,activation_fn=tf.nn.sigmoid, \
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

    def createGraph(self):
        #Clear the default graph stack and reset the global default graph.
        tf.reset_default_graph()

        #Establish the training proceedure. We feed the reward and chosen action into the network
        #to compute the loss, and use it to update the network.
        self.myAgent = Agent(learningRate=self.learningRate,numberOfStates=self.numStates,numberOfActions=self.numActions)

        #output is a float32 numActions length vector that has the output values for a particular input (state)
        #We calculate the output using the matrix product of the input vector and the weights
        #So the weights matrix number of rows must equal the length of the input vector (numStates)
        #Since we use one_hot encoding for the inputs (vector of length numStates,
        #with all zeros except for a 1 in the index of the input state), the matrix multiplication
        #values are just the weights from input to output, put through the activation function
        #we currently are using the sigmoid activation function: y = 1 / (1 + exp(-x))
        #for example, a weight of 1.02 -> output of 0.7353,  1.00 -> .73106
        #The output equals those values plus the bias for that state.
        #If we did not use one_hot encoding, then
        #print('output:', self.myAgent.output)

        #chosen_action is an int32 length 1 vector that has the index of the largest value in output
        #evaluating it with a sess.run provides our action output
        #print('chosen_action', self.myAgent.chosen_action)

        #tf.trainable... returns all variables created with trainable=True
        #In our case, there are two, weights and biases

        # numStates x numActions matrix where self.weights[i,j] = weight to go from state i to action j
        self.weights = tf.trainable_variables()[0]

        #self.biases = tf.trainable_variables()[1]

        #1 x numActions vector of outputs (actions), each containing the input value with the highest weight for that action
        #self.actions_to_states = tf.argmax(self.weights, 0)

        #1 x numStates vector of inputs (states), each containing the output value with the highest weight for that state
        self.states_to_actions = tf.argmax(tf.transpose(self.weights), 0)
        self.model_saver = tf.train.Saver()
        if self.loadModels:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
            # Restore latest checkpoint
            self.model_saver.restore(self.sess, tf.train.latest_checkpoint('saved_models/.'))
            #self.model_saver.restore(self.sess, tf.train.latest_checkpoint('saved_models/agent_repeater/.'))
            print('using a saved model in agent')
            #self.model_saver.restore(self.sess, "saved_models/agent.ckpt")
        else:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())

        if self.useTensorBoard:
            self.createTensorboardSummaries()

        #Variable to call next(..) from the training generator. Calling the generator directly causes it to run from the start
        self.learner = self.runNet()

    def createTensorboardSummaries(self):
        def variable_statistics(var, name):
             #Attach a lot of summaries to a Tensor (for TensorBoard visualization).
             with tf.name_scope('statistics_agent' + name):
                 mean = tf.reduce_mean(var)
                 tf.summary.scalar('mean', mean)
                 with tf.name_scope('stddev'):
                     stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
                 tf.summary.scalar('stddev', stddev)
                 tf.summary.scalar('max', tf.reduce_max(var))
                 tf.summary.scalar('min', tf.reduce_min(var))
                 tf.summary.histogram('histogram', var)
        #end of variable statistics

        #variable_statistics(self.weights, 'weights')
        #tf.summary.histogram('weights_agent', 0.99 - self.weights)
        tf.summary.scalar('action_agent', self.myAgent.action_holder[0])
        tf.summary.scalar("reward_agent", self.myAgent.reward_holder[0])
        tf.summary.scalar("loss_agent", self.myAgent.loss[0])
        tf.summary.scalar('selectedOutput_agent', self.myAgent.selected_output[0])
        tf.summary.histogram("output_agent", self.myAgent.output)
        tf.summary.histogram('states_to_actions', self.states_to_actions)
        #tf.summary.histogram('actions_to_states', self.actions_to_states)

        self.merged = tf.summary.merge_all()
        # Launch the tensorflow graph
        self.writer = tf.summary.FileWriter("output/agent", self.sess.graph)
    #end of createTensorboardSummaries

    def printResetReport(self):
        print('In addition to the character a, the characters seen as inputs and their highest weighted targets were')
        numSeen = 0
        statesUsed = []
        print('a', self.characters[self.statesToActions[0]], \
            self.weightValues[0, self.statesToActions[0]])
        for i in range(len(self.statesToActions)):
            if self.statesToActions[i] > 0:
                print(self.characters[i], self.characters[self.statesToActions[i]], \
                    self.weightValues[i, self.statesToActions[i]])
                numSeen += 1
        print('a total of ', numSeen, ' characters')

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
                self.printResetReport()
                if self.useTensorBoard:
                    self.writer.flush()
                    self.writer.close()
                    print('closed TensorBoard writer with done call at counter = ', counter)
                if self.saveModels:
                    self.model_saver.save(self.sess, "saved_models/agent.ckpt")
                self.sess.close()
                return

            if self.resetVariables:
                if self.useTensorBoard:
                    self.printResetReport()
                # re-initialize values, but keep the tree structure
                print("resetting variables in agent")
                self.sess.run(tf.global_variables_initializer())
                self.resetVariables = False

            #If we did not get a positive reward, include a chance to pick a random action
            #Reduce chance of random action as we train the model.
            if counter < 10000:
                self.e = self.initialRandomActionProbability/((counter/50) + 10)

            if self.reward < 1 and np.random.rand(1) < self.e:
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

            _, self.weightValues, loss, output, selOutput, self.statesToActions = self.sess.run([self.myAgent.update, self.weights,\
                self.myAgent.loss, self.myAgent.output, self.myAgent.selected_output, self.states_to_actions],\
                    feed_dict=networkFeeder)

            if self.useTensorBoard:
                summaryFeeder = {self.myAgent.reward_holder:[self.reward],self.myAgent.action_holder:[self.action],\
                self.weights:self.weightValues, self.myAgent.loss:loss, self.myAgent.output:output,\
                self.myAgent.selected_output:selOutput }
                summaryMerged = self.sess.run(self.merged, feed_dict=summaryFeeder)
                self.writer.add_summary(summaryMerged, counter)
            #print('weights: ', self.weightValues)
            #print('weight used:', self.weightValues[self.action])
            #print('weight for character "a", action 0 = ', self.weightValues[0,0])
            # selected_output and loss are arrays with just single values
            print('step = ', counter) #, ' action = ', self.action, 'selected output = ', selOutput[0]) #, 'loss =', loss[0])
            counter += 1
            #print('statesToActions = ', self.statesToActions)
            #print(' output = ', output)
        #end while True
    #end def runNet(self)
 #end class simpleAgent
