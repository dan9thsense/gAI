# this network finds the best action without using inputs, it just sends out actions and looks for rewards

from responder import Responder
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

class Agent55():
    def __init__(self, learningRate, numberOfStates, numberOfActions):
        #These lines established the feed-forward part of the network. The agent takes a state and produces an action.
        self.state_in= tf.placeholder(shape=[1],dtype=tf.int32)
        state_in_OH = slim.one_hot_encoding(self.state_in, numberOfStates)
        output = slim.fully_connected(state_in_OH, numberOfActions,\
            biases_initializer=None,activation_fn=tf.nn.sigmoid, weights_initializer=tf.ones_initializer())
        self.output = tf.reshape(output,[-1])
        self.chosen_action = tf.argmax(self.output,0)

        #The next six lines establish the training proceedure. We feed the reward and chosen action into the network
        #to compute the loss, and use it to update the network.
        self.reward_holder = tf.placeholder(shape=[1],dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[1],dtype=tf.int32)
        self.responsible_weight = tf.slice(self.output,self.action_holder,[1])
        self.loss = -(tf.log(self.responsible_weight)*self.reward_holder)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learningRate)
        self.update = optimizer.minimize(self.loss)

class simpleAgent55(Responder):
    def __init__(self):
      Responder.__init__(self)
      self.NetWasReset = True
      
    def createGraph(self):
      
      #Clear the default graph stack and reset the global default graph.
      tf.reset_default_graph()

      #Establish the training proceedure. We feed the reward and chosen action into the network
      #to compute the loss, and use it to update the network.
      
      self.myAgent55 = Agent55(learningRate=self.learningRate,numberOfStates=self.numStates,numberOfActions=self.numActions)
      
      #The weights we will evaluate to look into the network., Returns all variables created with `trainable=True
      self.weights55 = tf.trainable_variables()[0] 
      
      #Variable to call next(..) from the training generator. Calling the generator directly causes it to run from the start
      self.learner55 = self.runNet()   
    
    def getOutput(self):
      if self.resetCalled:
        print("running a reset net in agent.py")
        try:
          next(self.learner55)
        except StopIteration:
          print("completed the reset net in agent.py")
        self.netWasReset = True        
        self.resetCalled = False
        
      else:
        if self.netWasReset:
          self.netWasReset = False
          self.createGraph()
          return next(self.learner55)
        else:
          return next(self.learner55)
        
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
          print("session closed in agent.py")
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
          self.action = sess.run(self.myAgent55.chosen_action,feed_dict={self.myAgent55.state_in:[self.currentState]})
               
        yield self.characters[self.action]

        # we freeze here
        # while frozen, the output is sent, a reward is received
        # and a new state received, which becomes the current state (but we don't use it)
        # or a reset command is received
        
           
        
        #we now have a new current state as well as the reward based on the action we took in the previous state
        #Update the network
        feed_dict={self.myAgent55.reward_holder:[self.reward],self.myAgent55.action_holder:[self.action],self.myAgent55.state_in:[self.previousState]}
        _,ww = sess.run([self.myAgent55.update, self.weights55], feed_dict=feed_dict)

      

