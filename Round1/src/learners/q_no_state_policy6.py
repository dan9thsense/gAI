# this network finds the best action without using inputs, it just sends out actions and looks for rewards

from responder import Responder
import numpy as np
import tensorflow as tf

class QNoStatePolicy(Responder):
    def __init__(self):
      Responder.__init__(self)
      #Clear the default graph stack and reset the global default graph.
      tf.reset_default_graph()
      
      #placeholders for rewards and actions
      self.reward_holder = tf.placeholder(shape=[1],dtype=tf.float32)
      self.action_holder = tf.placeholder(shape=[1],dtype=tf.int32)
            
      self.weights = tf.Variable(tf.ones([self.numActions]))
        

      
      #Establish the training proceedure. We feed the reward and chosen action into the network
      #to compute the loss, and use it to update the network.
      
      #pick out the weights that correspond to the action     
      self.responsible_weight = tf.slice(self.weights,self.action_holder,[1])
      self.chosen_action = tf.argmax(self.weights, 0)
           
      #loss function, we'll minimize this
      self.loss = -(tf.log(self.responsible_weight)*self.reward_holder)
      
      #optimize with gradient descent
      self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learningRate)
      
      self.update = self.optimizer.minimize(self.loss)     
      
      #Variable to call next(..) from the training generator. Calling the generator directly causes it to run from the start
      self.learn = self.runNet()
      
    def getOutput(self):
      return next(self.learn)
        
    def runNet(self):
      # Launch the tensorflow graph
      with tf.Session() as sess:
          sess.run(tf.global_variables_initializer())
          while True:           
            self.action = np.random.randint(self.numActions) 
            # include a chance to pick a random action
            if np.random.rand(1) < self.e:
              self.action = np.random.randint(self.numActions)              
            else:
              #pick the action with the highest weight
              self.action = sess.run(self.chosen_action)
              #self.vals = sess.run(self.weights)
              #print(self.vals)       
            yield self.characters[self.action]

            # we freeze here
            # while frozen, the output is sent, a reward is received
            # and a new state received, which becomes the current state (but we don't use it)
            # or a reset command is received
            
            # first check if the call came from reset
            if self.resetCalled:
              # reset the weights
              sess.run(tf.global_variables_initializer())
              continue              
            
            #Update the network.
            _,resp,ww = sess.run([self.update, self.responsible_weight, self.weights], feed_dict={self.reward_holder:[self.reward],self.action_holder:[self.action]})

