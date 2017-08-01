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

    hidden = self.nn_layer(self.state_in, numberOfStates, numHiddenLayerNeurons, 'hidden_layer', tf.nn.relu)
    self.output = self.nn_layer(hidden, numHiddenLayerNeurons, numberOfActions, 'output_layer', tf.nn.softmax)

    #self.chosenAction = tf.argmax(self.output,1, name="chosenAction")

    # We need to define the parts of the network needed for learning a policy
    #Tensor("Placeholder_1:0", shape=(?,), dtype=int32)
    self.reward_holder = tf.placeholder(shape=[None], dtype=tf.int32, name="reward_holder")
    self.action_holder = tf.placeholder(shape=[None], dtype=tf.int32, name="action_holder")

    with tf.name_scope('indexes'):
        #Tensor("add:0", shape=(?,), dtype=int32)
        self.indexes = tf.range(0, tf.shape(self.output)[0]) * tf.shape(self.output)[1] + self.action_holder

    with tf.name_scope('resp_outputs'):
        #Tensor("Gather:0", shape=(?,), dtype=float32)
        self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]), self.indexes)

    with tf.name_scope('loss'):
        self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs) * tf.to_float(self.reward_holder))
        #tf.summary.scalar('loss', self.loss)

    with tf.name_scope('ttrainable_vars'):
        # specify the trainable variables for later updating
        tvars = tf.trainable_variables()

    self.gradient_holders = []
    for idx, var in enumerate(tvars):
      placeholder = tf.placeholder(tf.float32, name=str(idx)+'_holder')
      self.gradient_holders.append(placeholder)

    #compute the gradients
    self.gradients = tf.gradients(self.loss, tvars)

    optimizer = tf.train.AdamOptimizer(learning_rate=learningRate)
    self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders, tvars))
    #self.merged = tf.summary.merge_all()

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

  # We can't initialize these variables to 0 - the network will get stuck.
  def weight_variable(self, shape):
    """Create a weight variable with appropriate initialization."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

  def bias_variable(self, shape):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

  def nn_layer(self, input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    """Reusable code for making a simple neural net layer.
    It does a matrix multiply, bias add, and then uses ReLU to nonlinearize.
    It also sets up name scoping so that the resultant graph is easy to read,
    and adds a number of summary ops.
    """
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
      # This Variable will hold the state of the weights for the layer
      with tf.name_scope('weights'):
        weights = self.weight_variable([input_dim, output_dim])
        #self.variable_statistics(weights)
      with tf.name_scope('biases'):
        biases = self.bias_variable([output_dim])
        #self.variable_statistics(biases)
      with tf.name_scope('Wx_plus_b'):
        preactivate = tf.matmul(input_tensor, weights) + biases
        #tf.summary.histogram('pre_activations', preactivate)
      activations = act(preactivate, name='activation')
      #tf.summary.histogram('activations', activations)
      return activations

class mdpPolicyAgent(Responder):
    def __init__(self):
        Responder.__init__(self)
        self.batch_size = 50
        self.gamma = 0.99
        self.update_frequency = 5
        self.learningRate = 0.1
        self.createGraph()


    #Compute the discounted reward_signal
    #for i,val in enumerate(r) returns a list containing (counter, value) for each element of r
    #That list is used to compute a vector of values with length = length of r
    #Takes 1d float array of rewards and computes discounted reward
    #    e.g. f([1, 1, 1], 0.99) -> [1, 0.99, 0.9801]
    def discount_rewards(self, r, gamma=0.99):
        #print("r =", r)
        return np.array([val * (gamma ** i) for i, val in enumerate(r)])

    def createGraph(self):
        #Clear the default graph stack and reset the global default graph.
        tf.reset_default_graph()

        # Placeholders for our observations, outputs and rewards
        #these are not tf placeholders
        self.xs = np.empty(0).reshape(0, 1)
        self.ys = np.empty(0).reshape(0, 1)
        self.rewards = np.empty(0).reshape(0, 1)

        self.myAgent = Agent(numHiddenLayerNeurons=100, learningRate=self.learningRate, numberOfStates=self.numStates, numberOfActions=self.numActions)

        #Variable to call next(..) from the training generator. Calling the generator directly causes it to run from the start
        self.learner = self.runNet()

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

    def runNet(self):
        # Launch the tensorflow graph
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        # use tensorBoard, writing to the directory "output"
        writer = tf.summary.FileWriter("output", sess.graph)
        learningRateTB = tf.summary.scalar("learning_rate", self.learningRate)
        #lossTB = tf.summary.scalar("loss", self.myAgent.loss)
        merged = tf.summary.merge_all()

        gradBuffer = sess.run(tf.trainable_variables())
        for ix,grad in enumerate(gradBuffer):
            gradBuffer[ix] = grad * 0
        counter = 0
        ep_history = []
        while True:
            # first check if the call came from reset
            if self.resetCalled:
                # reset the weights
                print("reset called in agent.py, exiting tf session")
                #with tf.name_scope('summaries'):
                #merged_summary = tf.summary.merge_all()
                #summary = sess.run(merged_summary)
                #writer.add_summary(summary)
                writer.flush()
                writer.close()
                sess.close()
                return

            if self.resetVariables:
               # re-initialize values, but keep the tree structure
               sess.run(tf.global_variables_initializer())
               self.resetVariables = False

            #Probabilistically pick an action given our network outputs.
            #generate a vector of length numActions, where each entry is the probability of picking that entry
            stateRS, summary = sess.run([self.myAgent.state_in_rs, merged], feed_dict={self.myAgent.charIndex:[self.currentState]})
            a_dist = sess.run(self.myAgent.output,feed_dict={self.myAgent.state_in:stateRS})
            #pick an entry and return its value
            self.action = np.random.choice(a_dist[0],p=a_dist[0])
            #find the index of the value chosen
            self.action = np.argmax(a_dist == self.action)

            #oneHot = sess.run(self.myAgent.state_in_OH, feed_dict={self.myAgent.charIndex:[self.currentState]})
            #stateIn =  sess.run(self.myAgent.state_in, feed_dict={self.myAgent.charIndex:[self.currentState]})
            #self.action = sess.run(self.myAgent.chosenAction, feed_dict={self.myAgent.state_in:stateIn})
            #self.action = sess.run(self.myAgent.chosenAction, feed_dict={self.myAgent.charIndex:[self.currentState]})
            self.previousState = self.currentState
            #print(self.action)
            #summary = sess.run([self.myAgent.merged], feed_dict={self.myAgent.charIndex:[self.currentState], self.myAgent.state_in:stateRS})
            writer.add_summary(summary, counter)

            yield self.characters[self.action]
            ep_history.append([stateRS,self.action,self.reward,self.currentState])
            self.rewards = np.vstack([self.rewards, self.reward])
            #print("lengths = ", len(self.rewards), len(ep_history))
            #print("values = ", self.rewards, ep_history)
            #self.rewards.append(self.reward)

            if self.reward == 1:
                #Update the network.
                print("Got rewarded in mdpPolicy!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                ep_history_array = np.array(ep_history)
                #print("ep_history: ", ep_history[:,2])
                #ep_history[:,2] = self.discount_rewards(ep_history[:,2])
                # Determine standardized rewards
                #print("rewards = ", self.rewards)
                #discounted_rewards = np.array([val * (self.gamma ** i) for i, val in enumerate(self.rewards)])
                discounted_rewards = np.array([val * (self.gamma ** i) for i, val in enumerate(self.rewards)])
                #discounted_rewards = self.discount_rewards(self.rewards)
                discounted_rewards -= discounted_rewards.mean()
                discounted_rewards /= discounted_rewards.std()
                #print(discounted_rewards.shape)
                #print(ep_history_array[:,2].shape)
                #print(ep_history_array[:,2])
                #print(ep_history_array.shape)
                reshapeDR = tf.reshape(discounted_rewards, [len(ep_history_array[:,2])])
                #print(reshapeDR.shape)
                #print(reshapeDR)
                ep_history_array[:,2] = sess.run(reshapeDR)
                #print(ep_history_array[:,2])
                feed_dict={self.myAgent.reward_holder:ep_history_array[:,2], self.myAgent.action_holder:ep_history_array[:,1], self.myAgent.state_in:np.vstack(ep_history_array[:,0])}
                grads = sess.run(self.myAgent.gradients, feed_dict=feed_dict)

                for idx,grad in enumerate(grads):
                    gradBuffer[idx] += grad

                if counter % self.update_frequency == 0 and counter != 0:
                    feed_dict= dictionary = dict(zip(self.myAgent.gradient_holders, gradBuffer))
                    _ = sess.run(self.myAgent.update_batch, feed_dict=feed_dict)
                    for ix,grad in enumerate(gradBuffer):
                        gradBuffer[ix] = grad * 0
                    counter = 0

            counter += 1
        #with tf.name_scope('summaries'):
        #merged_summary = tf.summary.merge_all()
        #summary = sess.run([merged_summary])
        #writer.add_summary(summary)
        writer.flush()
        writer.close()
        sess.close()
        return
