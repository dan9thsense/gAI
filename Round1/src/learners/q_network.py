import numpy as np
import random
import tensorflow as tf

from responder import Responder

class myQNetwork(Responder):
    def __init__(self):
      Responder.__init__(self) 
      self.numStates = 71
      self.numActions = 71 
      tf.reset_default_graph()
      
      # Set learning parameters
      learningRate = 0.1
      self.action = 0
      self.currentState = 0
      self.y = .99
      self.e = 0.1
      self.numResets = 0
      
      self.resetCalled = False
      self.learn = self.runNet()

      #These lines establish the feed-forward part of the network used to choose actions
      self.inputs1 = tf.placeholder(shape=[1,self.numStates],dtype=tf.float32)
      self.W = tf.Variable(tf.random_uniform([self.numStates,self.numActions],0,0.01))
      self.Qout = tf.matmul(self.inputs1, self.W)
      self.predict = tf.argmax(self.Qout,1)

      #Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
      self.nextQ = tf.placeholder(shape=[1,self.numActions],dtype=tf.float32)
      self.loss = tf.reduce_sum(tf.square(self.nextQ - self.Qout))
      self.trainer = tf.train.GradientDescentOptimizer(learning_rate = learningRate)
      self.updateModel = self.trainer.minimize(self.loss)      
      self.init = tf.global_variables_initializer() #tf.initialize_all_variables()

    def rewardIn(self, reward):
      self.reward = reward
      self.totalReward += reward
      self.rewardList.append(self.totalReward)
           
    def charOut(self, charIn):
      self.inputCharacter = charIn
      # find the index of the character
      for i in range(len(self.characters)):
        if self.characters[i] == charIn:
          self.currentState = i
          break
      self.outputCharacter = next(self.learn)
      return self.outputCharacter
            
    def reset(self):
      self.resetCalled = True
      # break out of generator
      i = next(self.learner())
      self.resetCalled = False
      #Initialize table with all zeros
      self.Q = np.zeros([self.numStates, self.numActions])
      # Set learning parameters
      self.action = 0
      self.currentState = 0
      self.totalReward = 0
      self.rewardList = []
      #Reduce chance of random action as we train the model.
      if self.numResets < 10000:
        self.numResets += 1
        self.e = 1./((self.numResets/50) + 10)
      
    def runNet(self):
      with tf.Session() as sess:
          sess.run(self.init)
          while True:
            #Choose an action by greedily (with e chance of random action) from the Q-network
            # newAction returns as a length 1 array, with the index of the character to return
            # allQ returns the new Q vector, length = number of states
            newAction,allQ = sess.run([self.predict, self.Qout], feed_dict={self.inputs1:np.identity(self.numStates)[self.currentState:self.currentState+1]})
   
            # chance e of random action
            if np.random.rand(1) < self.e:
                newAction[0] = np.random.randint(1,self.numActions)
            self.action = newAction[0]    
            previousState = self.currentState
            #print("row ", self.currentState, " of the Q table = ", self.Q[self.currentState,:])
            yield self.characters[self.action]

            # we freeze here
            # while frozen, the output is sent, a reward is received
            # and a new state received, which becomes the current state
            # or a reset command is received
            
            # first check if the call came from reset
            if self.resetCalled:
              break
              
            #Obtain the Q' values by feeding the new state through our network
            Q1 = sess.run(self.Qout,feed_dict={self.inputs1:np.identity(self.numStates)[self.currentState:self.currentState+1]})
            #Obtain maxQ' and set our target value for chosen action.
            maxQ1 = np.max(Q1)
            targetQ = allQ
            targetQ[0,self.action] = self.reward + self.y * maxQ1
            #Train our network using target and predicted Q values
            _,W1 = sess.run([self.updateModel,self.W], feed_dict={self.inputs1:np.identity(self.numStates)[previousState:previousState+1],self.nextQ:targetQ})


