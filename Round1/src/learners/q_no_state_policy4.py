# top class for learners
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from responder import Responder

class QNoStatePolicy(Responder):
  def __init__(self):
    Responder.__init__(self)

    #Clear the default graph stack and reset the global default graph.
    tf.reset_default_graph()
    self.init = tf.global_variables_initializer()
    self.weights = tf.Variable(tf.ones([self.numActions]))
    self.chosen_action = tf.argmax(self.weights,0)  
    
    #self.createGraph()
             
    #def createGraph(self):
  
    
    #Establish the training proceedure. We feed the reward and chosen action into the network
    #to compute the loss, and use it to update the network.
    
    #placeholders for rewards and actions
    self.reward_holder = tf.placeholder(shape=[1],dtype=tf.float32)
    self.action_holder = tf.placeholder(shape=[1],dtype=tf.int32)
    
    #pick out the weights that correspond to the action     
    self.responsible_weight = tf.slice(self.weights, self.action_holder,[1])
    
    #loss function, we'll minimize this
    self.loss = -(tf.log(self.responsible_weight)*self.reward_holder)
    
    #optimize with gradient descent
    self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learningRate)
    
    self.update = self.optimizer.minimize(self.loss)
    self.weights = tf.ones([self.numActions]) 
    
    #Variable to call next(..) from the training generator. Calling the generator directly causes it to run from the start
    
    # setup to use the generator
    self.learn = self.runNet()
    #print next(self.learn) this causes a bad error that just says uninitialed when running sess

    
        
    #send in a input
    testChar = self.charOut("L")
    self.rewardIn(1)
    #testChar = self.charOut("L")
    
                     
  def charOut(self, charIn):
    self.inputCharacter = charIn
    print("character in was = ", charIn)
    # find the index of the character
    for i in range(len(self.characters)):
      if self.characters[i] == charIn:
       self.currentState = i
       break
    self.outputCharacter = next(self.learn)
    print("character out was = ", self.outputCharacter)
    return self.outputCharacter   

  def rewardIn(self, reward):
    self.reward = reward
    print("reward received = ", self.reward)
    self.totalReward += reward      
    if self.recordRewards:
      self.rewardList.append(self.totalReward)
              
  def runNet(self):
    # Launch the tensorflow graph
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        while True:
          print("chosen_action = ", self.chosen_action)            
          # include a chance to pick a random action
          if np.random.rand(1) < self.e:
           self.action = np.random.randint(self.numActions)              
          else:
            #pick the action with the highest weight
            self.action = sess.run(self.chosen_action)         
          print("output index = ", self.action)
          print("output character = ", self.characters[self.action])
          yield self.characters[self.action]

          # we freeze here
          # while frozen, the output is sent, a reward is received
          # and a new state received, which becomes the current state (but we don't use it)
          # or a reset command is received
          
          # first check if the call came from reset
          if self.resetCalled:
            break              
          
          #Update the network.
          print("self.action = ", self.action)
          print("self.action_holder = ", self.action_holder)
          print("self.reward = ", self.reward)
          print("self.reward_holder = ", self.reward_holder)
          sess.run([self.update, self.responsible_weight, self.weights], feed_dict={self.reward_holder:[self.reward],self.action_holder:[self.action]})

