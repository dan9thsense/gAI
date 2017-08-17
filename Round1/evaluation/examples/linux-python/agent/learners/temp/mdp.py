# use tensorflow to solve an mdp

from responder import Responder
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
  
class agent():
    def __init__(self, lr, s_size,a_size,h_size):
        #These lines established the feed-forward part of the network. The agent takes a state and produces an action.
        #self.state_in= tf.placeholder(shape=[None,s_size],dtype=tf.float32)
        self.state_in= tf.placeholder(shape=[1],dtype=tf.int32)
        hidden = slim.fully_connected(self.state_in,h_size,biases_initializer=None,activation_fn=tf.nn.relu)
        self.output = slim.fully_connected(hidden,a_size,activation_fn=tf.nn.softmax,biases_initializer=None)
        self.chosen_action = tf.argmax(self.output,1)

        #The next six lines establish the training proceedure. We feed the reward and chosen action into the network
        #to compute the loss, and use it to update the network.
        self.reward_holder = tf.placeholder(shape=[None],dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[None],dtype=tf.int32)
        
        self.indexes = tf.range(0, tf.shape(self.output)[0]) * tf.shape(self.output)[1] + self.action_holder
        self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]), self.indexes)

        self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs)*self.reward_holder)
        
        tvars = tf.trainable_variables()
        self.gradient_holders = []
        for idx,var in enumerate(tvars):
            placeholder = tf.placeholder(tf.float32,name=str(idx)+'_holder')
            self.gradient_holders.append(placeholder)
        
        self.gradients = tf.gradients(self.loss,tvars)
        
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders,tvars))


class mdpAgent(Responder):
    def __init__(self):
      Responder.__init__(self)
      self.gamma = 0.99
      
    def createGraph(self):
      tf.reset_default_graph() #Clear the Tensorflow graph.

      self.myAgent = agent(lr=1e-2,s_size=71,a_size=71,h_size=8) #Load the agent.

      self.max_ep = 999
      self.update_frequency = 5
 
      self.learner = self.runNet()   
    
    def getOutput(self):
      if self.resetCalled:
        try:
          next(self.learner)
        except StopIteration:
          print("completed a reset net in agent.py")
        self.netWasReset = True        
        self.resetCalled = False
        
      else:
        if self.netWasReset:
          self.netWasReset = False
          print("creating a new tf graph in agent.py")
          self.createGraph()
          return next(self.learner)
        else:
          return next(self.learner)
                
    def discount_rewards(self, r):
        """ take 1D float array of rewards and compute discounted reward """
        discounted_r = np.zeros_like(r)
        running_add = 0
        for t in reversed(xrange(0, r.size)):
            running_add = running_add * self.gamma + r[t]
            discounted_r[t] = running_add
        return discounted_r
        

    def runNet(self):
    # Launch the tensorflow graph
      # Launch the tensorflow graph
      sess = tf.Session()
      sess.run(tf.global_variables_initializer())
      total_reward = []
      total_length = []
          
      gradBuffer = sess.run(tf.trainable_variables())
      for ix,grad in enumerate(gradBuffer):
          gradBuffer[ix] = grad * 0
            
      while True:
        # first check if the call came from reset
        if self.resetCalled:
          # reset the weights
          print("reset called in agent.py, exiting tf session")
          sess.close()
          return
          
        if self.resetVariables:
          # re-initialize values, but keep the tree structure
          sess.run(tf.global_variables_initializer())
          self.resetVariables = False
                  
        ep_history = []
        for j in range(self.max_ep):
            #Probabilistically pick an action given our network outputs.
            currentVector[self.currentState:]
            a_dist = sess.run(self.myAgent.output,feed_dict={self.myAgent.state_in:[]})
            self.action = np.random.choice(a_dist[0],p=a_dist[0])
            self.action = np.argmax(a_dist == self.action)
            
            yield self.action
            
            # we freeze here
            # while frozen, the output is sent, a reward is received
            # and a new state received, which becomes the current state
            
            ep_history.append([self.previousState,self.action,self.reward,self.currentState])
            
            #Update the network.
            ep_history = np.array(ep_history)
            ep_history[:,2] = self.discount_rewards(ep_history[:,2])
            feed_dict={self.myAgent.reward_holder:ep_history[:,2],
                    self.myAgent.action_holder:ep_history[:,1],self.myAgent.state_in:np.vstack(ep_history[:,0])}
            grads = sess.run(self.myAgent.gradients, feed_dict=feed_dict)
            for idx,grad in enumerate(grads):
                gradBuffer[idx] += grad

            if i % self.update_frequency == 0 and i != 0:
                feed_dict= dictionary = dict(zip(self.myAgent.gradient_holders, gradBuffer))
                _ = sess.run(self.myAgent.update_batch, feed_dict=feed_dict)
                for ix,grad in enumerate(gradBuffer):
                    gradBuffer[ix] = grad * 0
                    
