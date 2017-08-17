import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim


class agent():
    def __init__(self, lr, s_size,a_size,h_size):
        #These lines established the feed-forward part of the network. The agent takes a state and produces an action.
        #self.state_in= tf.placeholder(shape=[None,s_size],dtype=tf.float32)

        self.state_in= tf.placeholder(shape=[1],dtype=tf.int32, name='state_in')
        state_in_OH = slim.one_hot_encoding(self.state_in, s_size)


        hidden = slim.fully_connected(state_in_OH,h_size,biases_initializer=tf.constant_initializer(0.1),activation_fn=tf.nn.sigmoid)
        self.output = slim.fully_connected(hidden,a_size,activation_fn=tf.nn.softmax,biases_initializer=tf.constant_initializer(0.1))
        self.chosen_action = tf.argmax(self.output,1)

        #The next six lines establish the training proceedure. We feed the reward and chosen action into the network
        #to compute the loss, and use it to update the network.
        self.reward_holder = tf.placeholder(shape=[None],dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[None],dtype=tf.int32)

        self.indexes = tf.range(0, tf.shape(self.output)[0]) * tf.shape(self.output)[1] + self.action_holder
        self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]), self.indexes)

        self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs)*self.reward_holder)

        tvars = tf.trainable_variables()
        print(tvars)
        self.gradient_holders = []
        for idx,var in enumerate(tvars):
            placeholder = tf.placeholder(tf.float32,name=str(idx)+'_holder')
            self.gradient_holders.append(placeholder)

        self.gradients = tf.gradients(self.loss,tvars)

        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders,tvars))

tf.reset_default_graph() #Clear the Tensorflow graph.

myAgent = agent(lr=1e-2,s_size=4,a_size=2,h_size=8) #Load the agent.


sess = tf.Session()
sess.run(tf.global_variables_initializer())
writer = tf.summary.FileWriter("output/tl", sess.graph)
gradBuffer = sess.run(tf.trainable_variables())
for ix,grad in enumerate(gradBuffer):
    gradBuffer[ix] = grad * 0
a_dist = sess.run(myAgent.output,feed_dict={myAgent.state_in:[0]})
writer.flush()
writer.close()
sess.close()
