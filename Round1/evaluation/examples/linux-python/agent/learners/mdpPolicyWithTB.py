# policy based agent

from responder import Responder
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import math

class Agent():
    def __init__(self, numHiddenLayerNeurons, learningRate, numberOfStates, numberOfActions):
        #These lines established the feed-forward part of the network. The agent takes a state and produces an action.
        #takes in the current state, which is the character index of the input

        #Input placeholders
        with tf.name_scope('input'):
            self.charIndex = tf.placeholder(dtype=tf.int32, name="charIndex")
            #do one-hot encoding on the state
            self.state_in_OH = tf.to_float(slim.one_hot_encoding(self.charIndex, numberOfStates))
            #need to reshape from (numberOfStates,) to (1,numberOfStates) to feed into state_in (?, numberOfStates)
            self.state_in_rs = tf.reshape(self.state_in_OH, [1, numberOfStates])

            #The None element of the shape corresponds to a variable-sized dimension.
            #we need that to be able to feed in a variable number of states,
            #based on the number of steps it takes before we get a reward
            self.state_in = tf.placeholder(shape=[None, numberOfStates], dtype=tf.float32, name="state_in")

        '''
        `fully_connected` creates a variable called `weights`, representing a fully
        connected weight matrix, which is multiplied by the `inputs` to produce a
        `Tensor` of hidden units. If a `normalizer_fn` is provided (such as
        `batch_norm`), it is then applied. Otherwise, if `normalizer_fn` is
        None and a `biases_initializer` is provided then a `biases` variable would be
        created and added the hidden units. Finally, if `activation_fn` is not `None`,
        it is applied to the hidden units as well.
        #hidden = slim.fully_connected(self.state_in, numHiddenLayerNeurons, biases_initializer=None, activation_fn=tf.nn.relu)
        #self.output = slim.fully_connected(hidden, numberOfActions, activation_fn=tf.nn.softmax, biases_initializer=None)

        #but we want to plot the weights and see parts of the layers in tensorboard, so we will create
        #them with more long-winded code
        '''

        #hidden is a function that takes input as a vector of length numberOfStates
        #multiplies it by the weights matrix (numberofStates x numHiddenLayerNeurons),
        #adds the biases, applies the relu activation function
        #and returns a vector of length numHiddenLayerNeurons
        #with values that correspond to the results.
        #it is evaluated using sess.run and we will send those outputs to the next layer
        #hidden = self.nn_layer(self.state_in, numberOfStates, numHiddenLayerNeurons, 'hidden_layer', tf.nn.relu)

        #self.output is a function just like hidden, where we take the results from hidden
        #as inputs, evaluate it with sess.run, and produce a vector of length numberOfActions
        #We could use the action that corresponds to the maximum value of that output
        #but we get better results by using a probabalistic approach
        #We will use the output vector as a set of probabilities, then choose which entry
        #to use based on those probabilities.  If the results are narrowly distributed around a
        #particular output, as it will be late in training, then that output is highly likely to get chosen.
        #If the distribution
        #is broad, as it will be early in training, then we have a greater chance of picking some
        #other value.  Once we pick the value to use, we find the corresponding index in the
        #vector and that is our chosen action.
        #self.output = self.nn_layer(hidden, numHiddenLayerNeurons, numberOfActions, 'output_layer', tf.nn.softmax)
        self.output = self.nn_layer(self.state_in, numberOfStates, numberOfActions, 'output_layer', tf.nn.sigmoid)

        #Tensor("Placeholder_1:0", shape=(?,), dtype=int32)
        self.reward_holder = tf.placeholder(shape=[None], dtype=tf.int32, name="reward_holder")
        self.action_holder = tf.placeholder(shape=[None], dtype=tf.int32, name="action_holder")

        with tf.name_scope('indexes'):
            #Tensor("add:0", shape=(?,), dtype=int32)
            self.indexes = tf.range(0, tf.shape(self.output)[0]) * tf.shape(self.output)[1] + self.action_holder

        with tf.name_scope('selected_output'):
            #Tensor("Gather:0", shape=(?,), dtype=float32)
            self.selected_output = tf.clip_by_value(tf.gather(tf.reshape(self.output, [-1]), self.indexes), 1e-20, 1.0)
            #since self.selected_output is used in a log function,
            #we avoid NaN by clipping when value in self.selected_output becomes zero

        with tf.name_scope('loss'):
            self.loss = -tf.reduce_mean(tf.log(self.selected_output) * tf.to_float(self.reward_holder))

        with tf.name_scope('trainable_vars'):
            # specify the trainable variables for later updating.
            # index [0] is weights (numStates x numHiddenLayerNeurons), index [1] is biases (numHiddenLayerNeurons,)
            # for the hidden layer.
            # index[2] is weights (numHiddenLayerNeurons x numActions), index [3] is biases (numActions,)
            # for the output layer
            tvars = tf.trainable_variables()

        #we have a gradient holder for each trainable variable
        #data over time.  The gradients calculated as we do this are stored in self.gradient_holders
        #self.gradient.holders will be size: (tvars size) x numberOfActions x numHiddenNeurons
        self.gradient_holders = []

        #we create a placeholder for each of the trainable variables, in our case, four:
        #weight and bias for the hidden and output layers
        #and append them to gradient_holders
        for idx, var in enumerate(tvars):
            placeholder = tf.placeholder(tf.float32, name=str(idx)+'_holder')
            self.gradient_holders.append(placeholder)

        #compute the gradients
        self.gradients = tf.gradients(self.loss, tvars)

        optimizer = tf.train.AdamOptimizer(learning_rate=learningRate)
        self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders, tvars))

    # We can't initialize these variables to 0 - the network will get stuck.
    def weight_variable(self, shape):
        #Called from nn_layer to create a weight variable with appropriate initialization.
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        #Called from nn_layer to create a bias variable with appropriate initialization.
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def nn_layer(self, input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
        #make a simple neural net layer.
        #takes an input vector of length input_dim
        #It does a matrix multiply, bias add, and then uses an activation function (default is relu) to nonlinearize.
        #It also sets up name scoping so that the resultant graph is easy to read,

        # Adding a name scope ensures logical grouping of the layers in the graph.
        with tf.name_scope(layer_name):
            #weights for the layer
            with tf.name_scope('weights'):
                weights = self.weight_variable([input_dim, output_dim])
            with tf.name_scope('biases'):
                biases = self.bias_variable([output_dim])
            with tf.name_scope('Wx_plus_b'):
                #multiply the input tensor by the weights and add biases
                preactivate = tf.matmul(input_tensor, weights) + biases
            with tf.name_scope('activation_fcn'):
                #apply the activation function (default is relu)
                activations = act(preactivate, name='activation')

        #return an output vector function of length output_dim that will be evaluated with sess.run to provide
        #an output vector of length output_dim with the results of act(Wx+b) in each entry
        return activations

class mdpPolicyAgent(Responder):
    def __init__(self):
        Responder.__init__(self)
        self.batch_size = 1
        self.gamma = 0.99
        self.numRewardsForUpdate = 1
        self.learningRate = 0.03
        self.numHiddenNeurons = 1
        self.createGraph()

    def discount_rewards(self, r, gamma=0.99):
        #Compute the discounted reward_signal
        #for i,val in enumerate(r) returns a list containing (counter, value) for each element of r
        #That list is used to compute a vector of values with length = length of r
        #Takes 1d float array of rewards and computes discounted reward
        #    e.g. f([1, 1, 1], 0.99) -> [1, 0.99, 0.9801]
        #print("r =", r)
        dR = np.array([val * (gamma ** i) for i, val in enumerate(r)])
        dR -= dR.mean()
        if dR.std() != 0:
            dR /= dR.std()
        return dR


    def createGraph(self):
        #Clear the default graph stack and reset the global default graph.
        tf.reset_default_graph()

        #create the network
        self.myAgent = Agent(numHiddenLayerNeurons=self.numHiddenNeurons, learningRate=self.learningRate,\
            numberOfStates=self.numStates, numberOfActions=self.numActions)

        #The weights we will evaluate to look into the network, just using names that are clearer than tvar
        #tf.trainable... returns all variables created with trainable=True
        self.hiddenLayerWeights = tf.trainable_variables()[0] #size (numStates, numHiddenLayerNeurons)
        self.hiddenLayerBiases = tf.trainable_variables()[1]  #size (numHiddenLayerNeurons,)
        #self.outputLayerWeights = tf.trainable_variables()[2] #size (numHiddenLayerNeurons, numActions)
        #self.outputLayerBiases = tf.trainable_variables()[3]  #size (numActions,)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        if self.useTensorBoard:
            self.createTensorboardSummaries()

        #Variable to call next(..) from the training generator. Calling the generator directly causes
        #the whole runNet() function to run from the start
        self.learner = self.runNet()

    def createTensorboardSummaries(self):
        def variable_statistics(var, name):
             #Attach a lot of summaries to a Tensor (for TensorBoard visualization).
             with tf.name_scope('statistics_mdpPolicy_' + name):
                 mean = tf.reduce_mean(var)
                 tf.summary.scalar('mean', mean)
                 with tf.name_scope('stddev'):
                     stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
                     tf.summary.scalar('stddev', stddev)
                 tf.summary.scalar('max', tf.reduce_max(var))
                 tf.summary.scalar('min', tf.reduce_min(var))
                 tf.summary.histogram('histogram', var)
        #end of variable statistics

        with tf.name_scope('mdpPolicy'):
            #variable_statistics(self.hiddenLayerWeights, 'hidden_layer')
            tf.summary.histogram('zeroed_weights_mdpPolicy', 0.99 - self.hiddenLayerWeights[self.action])
            #tf.summary.histogram('biases_mdpPolicy', self.hiddenLayerBiases)
            #tf.summary.histogram('action_mdpPolicy', self.myAgent.action_holder)
            #tf.summary.histogram("reward_mdpPolicy", self.myAgent.reward_holder)
            #tf.summary.scalar('selectedOutput_mdpPolicy', self.myAgent.selected_output[0])
            #tf.summary.histogram("output_mdpPolicy", self.myAgent.output)
            #tf.summary.scalar("loss_mdpPolicy", self.myAgent.loss[0])
            self.currentReward = tf.placeholder(dtype=tf.int32, name="current_reward")
            self.currentAction = tf.placeholder(dtype=tf.int32, name="current_action")
            tf.summary.scalar('current_reward', self.currentReward)
            tf.summary.scalar('current_action', self.currentAction)

        #merge all the summaries
        self.merged = tf.summary.merge_all()
        # Launch the tensorflow graph
        self.writer = tf.summary.FileWriter("output/mdpPolicy", self.sess.graph)
    #end of createTensorboardSummaries

    def getOutput(self):
        if self.done:
            try:
                next(self.learner)
            except StopIteration:
                print("completed a reset net in mdpPolicy.py")
                self.netWasReset = True
                self.done = False

        else:
            if self.netWasReset:
                self.netWasReset = False
                print("creating a new tf graph in mdpPolcy.py")
                self.createGraph()
                return next(self.learner)
            else:
                return next(self.learner)

    def runNet(self):
        #calculate the values for weights and biases in both layers into gradBuffer, just to establish
        #the first dimension of gradBuffer as (trainable_variables (4 in our case))
        # and the other dimensions as specified earlier
        #in our case,
        #gradBuffer[0] holds the weights for the hidden layer, so is size (numStates, numHiddenLayerNeurons)
        #gradBuffer[1] holds the biases for the hidden layer, so is size (numHiddenLayerNeurons,)
        #gradBuffer[2] holds the weights for the output layer, so is size (numHiddenLayerNeurons, numActions)
        #gradBuffer[3] holds the biases for the hidden layer, so is size (numActions,)
        gradBuffer = self.sess.run(tf.trainable_variables())
        #zero out all the entries in gradBuffer
        for ix,grad in enumerate(gradBuffer):
            gradBuffer[ix] = grad * 0

        counter = 0
        summaryCounter = 0 #count the tensorflow summary writes
        ep_history = [] #we will store the past states, actions, and rewards with this list
        rewardsArray = np.empty(0).reshape(0, 1) #an extra list of rewards, used for calculating the discounted rewards
        positive_reward_sum = 0
        gradsCalculated = 0
        gradsUpdated = 0

        while True:
            print(counter, self.globalCounter)
            self.globalCounter += 1
            counter += 1
            # first check if the call came from reset or done
            if self.done:
                print("done called in mdpPolicyWithTB, exiting tf session")
                print('number of times grads calculated = ', gradsCalculated, 'grads updated: ', gradsUpdated)
                if self.useTensorBoard:
                    self.writer.flush()
                    self.writer.close()
                self.sess.close()
                return

            if self.resetVariables:
                # re-initialize values, but keep the tree structure
                self.sess.run(tf.global_variables_initializer())
                self.resetVariables = False

            #take the character that is input and process it to one_hot encoding ready to go into the hidden layer
            stateRS = self.sess.run(self.myAgent.state_in_rs, feed_dict={self.myAgent.charIndex:[self.currentState]})

            #Take the input, calculate act(Wx+b) for both layer and provide an output vector
            #Then use that vector (output_dist) to probabilistically pick an action.
            #output_dist is a vector of length numActions. We use each entry as the probability of picking that entry
            #then choose which entry to use based on those probabilities.
            #If the results are narrowly distributed around a particular entry, as they will be late in training,
            #then that output is highly likely to get chosen.  If the distribution
            #is broad, as it will be early in training, then we have a greater chance of picking some
            #other value.  Once we pick the value to use, we find the corresponding index in the
            #vector and that is our chosen action.

            #output_dist, hiddenWeights, hiddenBiases = self.sess.run([self.myAgent.output, self.hiddenLayerWeights, \
            #    self.hiddenLayerBiases], feed_dict={self.myAgent.state_in:stateRS})

            output_dist = self.sess.run(self.myAgent.output, feed_dict={self.myAgent.state_in:stateRS})

            #this uses output_dist as a set of probabilities.  Using those values as weights,
            # it selects an entry in ouput_dist. So the weight most likely to be selected is the highest weight
            #but overall things depend on the shape of the distribtion.
            #selectedWeight = np.random.choice(output_dist[0],p=output_dist[0])
            #find the index of the value chosen
            #this finds the argument in output_dist where the entry = selectedWeight
            #if there is no match, it returns 0 (character 'a')
            #self.action = np.argmax(output_dist == selectedWeight)

            #could make it not random, but you can get locked in where the initial values do not generate
            #a reward, so the net never updates and you just keep sending the same subset of characters
            self.action = np.argmax(output_dist)

            #oneHot = self.sess.run(self.myAgent.state_in_OH, feed_dict={self.myAgent.charIndex:[self.currentState]})
            #stateIn =  self.sess.run(self.myAgent.state_in, feed_dict={self.myAgent.charIndex:[self.currentState]})
            #self.action = self.sess.run(self.myAgent.chosenAction, feed_dict={self.myAgent.state_in:stateIn})
            #self.action = self.sess.run(self.myAgent.chosenAction, feed_dict={self.myAgent.charIndex:[self.currentState]})
            self.previousState = self.currentState

            yield self.characters[self.action]

            #we sent off the action and there is now a reward and a new currentState
            #we will save the state which we acted upon (stateRS), the action, the reward, and the new currentState
            ep_history.append([stateRS,self.action,self.reward,self.currentState])

            #and we also save the rewards in its own array
            rewardsArray = np.vstack([rewardsArray, self.reward])
            if self.reward == 1:
                positive_reward_sum += 1

            #put ep_history into an array for processing below
            ep_history_array = np.array(ep_history)

            if positive_reward_sum >= 0: #self.numRewardsForUpdate:
                #Update the network.
                #print("ep_history: ", ep_history[:,2])
                #ep_history[:,2] = self.discount_rewards(ep_history[:,2])

                #run the list of rewards through to discount older ones
                #discounted_rewards = np.array([val * (self.gamma ** i) for i, val in enumerate(rewardsArray)])
                discounted_rewards = self.discount_rewards(rewardsArray)
                #discounted_rewards -= discounted_rewards.mean()
                #discounted_rewards /= discounted_rewards.std()
                #print(discounted_rewards.shape)
                #print(ep_history_array[:,2].shape)
                #print(ep_history_array[:,2])
                #print(ep_history_array.shape)
                reshapeDR = tf.reshape(discounted_rewards, [len(ep_history_array[:,2])])
                #print(reshapeDR.shape)
                #print(reshapeDR)
                ep_history_array[:,2] = self.sess.run(reshapeDR)
                #print(ep_history_array[:,2])

                #create the feeder with the states we acted on, the actions, and the discounted rewards
                gradFeeder={self.myAgent.reward_holder:ep_history_array[:,2], \
                    self.myAgent.action_holder:ep_history_array[:,1], \
                    self.myAgent.state_in:np.vstack(ep_history_array[:,0])}

                #calculate the gradients and some others for use with tensorboard
                grads, loss, selOut = self.sess.run([self.myAgent.gradients, self.myAgent.loss, \
                    self.myAgent.selected_output], feed_dict=gradFeeder)

                #add these gradients to gradBuffer
                for idx,grad in enumerate(grads):
                    gradBuffer[idx] += grad

                rewardsArray = np.empty(0).reshape(0, 1)
                positive_reward_sum = 0
                ep_history = []
                gradsCalculated += 1
                print('mdpPolicy gradients calculated')

                '''
                indexes = self.sess.run(self.myAgent.indexes, \
                    feed_dict={self.myAgent.output:output_dist, self.myAgent.action_holder:ep_history_array[:,1] })
                selOut = self.sess.run(self.myAgent.selected_output, \
                    feed_dict={self.myAgent.output:output_dist, self.myAgent.indexes:indexes})
                loss = self.sess.run(self.myAgent.loss, \
                     feed_dict={self.myAgent.selected_output:selOut, self.myAgent.reward_holder:ep_history_array[:,2] })

                loss, selOut = self.sess.run([self.myAgent.loss, self.myAgent.selected_output], \
                    feed_dict={self.myAgent.output:output_dist, self.myAgent.reward_holder:ep_history_array[:,2], \
                    self.myAgent.action_holder:ep_history_array[:,1]})
                '''

                #if we have reached the batch_size, update the network with the saved gradients
                if counter % self.batch_size == 0 and counter != 0:
                    #zip converts a tuple of sequences to a sequence of tuples. dict makes it a dictionary
                    #keys = ['a', 'b', 'c'] values = [1, 2, 3]
                    #zip(keys,values) outputs [('a', 1), ('b', 2), ('c', 3)]
                    #dictionary = dict(zip(keys, values)) outputs  {'a': 1, 'b': 2, 'c': 3}
                    #create the feeder, pugging in gradBuffer
                    updateFeeder = dict(zip(self.myAgent.gradient_holders, gradBuffer))

                    #update the network weights and biases
                    _ = self.sess.run(self.myAgent.update_batch, feed_dict=updateFeeder)


                    #print('gradBuffer = ', len(gradBuffer), len(gradBuffer[0]), len(gradBuffer[1]))

                    #zero out the gradBuffer to get ready to take in the next batch of gradients
                    for ix,grad in enumerate(gradBuffer):
                        gradBuffer[ix] = grad * 0
                    print('mdpPolicy gradients updated')
                    gradsUpdated += 1
                 #end of if counter % self.batch.....
            #end of if reward >= self.numRewardsForUpdate

            if self.useTensorBoard:
                summaryFeeder = {self.myAgent.reward_holder:ep_history_array[:,2], \
                    self.myAgent.action_holder:ep_history_array[:,1], \
                    self.hiddenLayerWeights:hiddenWeights, self.hiddenLayerBiases:hiddenBiases, \
                    self.currentReward:self.reward, self.currentAction:self.action}
                    #self.myAgent.loss:loss, \
                    #self.myAgent.output:output_dist, self.myAgent.selected_output:selOut, \

                summaryMerged = self.sess.run(self.merged, feed_dict=summaryFeeder)
                self.writer.add_summary(summaryMerged, summaryCounter)
                summaryCounter += 1
            #end of if self.useTensorBoard
        #end of while True
        if self.useTensorBoard:
            self.writer.flush()
            self.writer.close()
        self.sess.close()
        return
