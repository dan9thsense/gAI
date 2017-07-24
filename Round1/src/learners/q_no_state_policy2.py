# top class for learners
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt

class QNoStatePolicy():
  def __init__(self):
    #Clear the default graph stack and reset the global default graph.
    tf.reset_default_graph()
    self.init = tf.global_variables_initializer()
    
    
    self.characters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', \
        'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', \
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', \
        'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', \
        '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', \
        ',', '.', '!', ';', '?', '-', ':', '"', ' ' ]
    self.specialCharacters = [ ',', '.', '!', ';', '?', '-', ':', '"', ' ' ]
    self.anCharacters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', \
        'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', \
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', \
        'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', \
        '1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
    self.numberCharacters = [ '1', '2', '3', '4', '5', '6', '7', '8', '9', '0' ]
    self.inputCharacter = '-'
    self.previousCharacter = '+'
    self.outputCharacter = '.'
    self.quietCharacter = ' ' # this is the correct response when we get an input that corrects our previous response
    self.resetCalled = False
    self.numResets = 0
    self.numActions = len(self.characters)
    self.initialRandomActionProbability = 0.02
    self.e = self.initialRandomActionProbability
    self.recordRewards = True
    self.plotResults = True
    self.learningRate = 0.001
    self.currentState = 0
    self.reward = 0
    self.totalReward = 0
    self.rewardList = []
    self.action = 0
    self.weights = tf.Variable(tf.ones([self.numActions]))
    self.chosen_action = tf.argmax(self.weights,0)
    #self.weights = tf.ones([self.numActions])
  

    
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
    
    #Variable to call next(..) from the training generator. Calling the generator directly causes it to run from the start
    
    # setup to use the generator
    #self.learn = self.runNet()
    #print next(self.learn) this causes a bad error that just says uninitialed when running sess
    
    #send in a input
    self.rewardIn(1)
    testChar = self.charOut("L")
    self.rewardIn(1)
    testChar = self.charOut("L")
        
  def rewardIn(self, reward):
    self.reward = reward
    self.totalReward += reward      
    if self.recordRewards:
      self.rewardList.append(self.totalReward)
           
  def charOut(self, charIn):
    self.inputCharacter = charIn
    print("character in was = ", charIn)
    # find the index of the character
    for i in range(len(self.characters)):
      if self.characters[i] == charIn:
       self.currentState = i
       break
    self.outputCharacter = next(self.runNet())
    print("character out was = ", self.outputCharacter)
    return self.outputCharacter
            
  def reset(self):
    self.resetCalled = True
    # break out of generator
    next(self.learn) 
    self.resetCalled = False
    self.initializeValues()
    
    #Reduce chance of random action as we train the model.
    if self.numResets < 10000:
      self.numResets += 1
      self.e = self.initialRandomActionProbability/((self.numResets/50) + 10)
    
  def plotReward(self, learnerName):
    if len(self.rewardList) > 2 and self.recordRewards and self.plotResults:
      print("plotting rewardList.  Its length = ", len(self.rewardList))
      plt.plot(self.rewardList)
      plt.title(learnerName)
      plt.show()
    else:
      print("not plotting rewardList")
    self.rewardList = []
    self.totalReward = 0
    

        
  def runNet(self):
    # Launch the tensorflow graph
    with tf.Session() as sess:        
        sess.run(tf.global_variables_initializer())
        #test = sess.run(self.weights)
        #print(test)
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

