# policy based agent
import numpy as np
from matplotlib import animation
from IPython.display import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import math


# Constants defining our neural network
hidden_layer_neurons = 10
batch_size = 50
learning_rate = 1e-2
gamma = .99
dimen = 4

tf.reset_default_graph()
#None is an alias for NP.newaxis. It creates an axis with length 1
#So that our placesholder is a row vector with dimen number of columns
# Define input placeholder
observations = tf.placeholder(tf.float32, [None, dimen], name="input_x")

# First layer of weights
#tf.get_variable is the new way to share or create a tf variable
#the xavier initializer is just a standard way
#This initializer is designed to keep the scale of the gradients roughly the same in all layers.
#For a uniform distribution, this ends up being the range: x = sqrt(6. / (in + out)); [-x, x]
#and for a normal distribution with a standard deviation of sqrt(3. / (in + out)) is used.
#In this case, it has no arguments specified, so it uses the defaults of
#uniform=True,seed=None,dtype=tf.float32
#It comes from the paper, Understanding the diffiuclty of training deep feedforward neural networks
#by Xavier Glorot, here: http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
W1 = tf.get_variable("W1", shape=[dimen, hidden_layer_neurons],
                    initializer=tf.contrib.layers.xavier_initializer())
#we will use a very simple non-linear (activation) function
#relu stands for Rectified Linear Unit and its definition is
#max(0,x), so if the input value is negative, relu returns 0,
#if the input is positive, then relu returns the value of the input.
#note that the derivative exists except when x=0 and is 0 for x<0 and 1 for x>0
#here we are multiplying the observations (a row vector with dimen columns)
#by the weights (a matrix of size dimen x hidden_layer_neurons)
#and then applying the relu
#the result is a row vector with hidden_layer_neurons number of columns
#which is a 1 x hidden_layer_neurons matrix
layer1 = tf.nn.relu(tf.matmul(observations,W1))

# Second layer of weights,
# size is a column vector with hidden_layer_neurons rows, again uniform initialization
W2 = tf.get_variable("W2", shape=[hidden_layer_neurons, 1],
                    initializer=tf.contrib.layers.xavier_initializer())

#multiply layer1, hidden_layer_neurons columns, by W2, hidden_layer_neurons rows
#to get a scalar output.  We then use a sigmoid non-linear "activation" function
output = tf.nn.sigmoid(tf.matmul(layer1,W2))

# We need to define the parts of the network needed for learning a policy
# specify the trainable variables for later updating
trainable_vars = [W1, W2]

#create a placeholder for the inputs, this will be a 1 x 1 matrix
#with the entry as a column
input_y = tf.placeholder(tf.float32, [None,1], name="input_y")
#create a placeholder for the reward, just a scalar
advantages = tf.placeholder(tf.float32, name="reward_signal")

#define the loss function to be minimized
# Loss function
log_lik = tf.log(input_y * (input_y - output) +
                  (1 - input_y) * (input_y + output))
loss = -tf.reduce_mean(log_lik * advantages)

# Gradients
#tf.gradients(a,b) constructs symbolic partial derivatives of a with respect to each element of b
#In our case, those are the partial derivatives of loss with respect to W1 and to W2
new_grads = tf.gradients(loss, trainable_vars)
#create scalar placeholders for values of the gradients
W1_grad = tf.placeholder(tf.float32, name="batch_grad1")
W2_grad = tf.placeholder(tf.float32, name="batch_grad2")

# Learning
#create a list with our gradients
batch_grad = [W1_grad, W2_grad]
#specify our optimization method as adam and specify the learning rate
adam = tf.train.AdamOptimizer(learning_rate=learning_rate)

#this will apply the optimizer to the gradients
#Apply gradients to variables.
#This is the second part of minimize().
#It takes in a list of (gradient, variable) pairs as returned by compute_gradients()
#It returns an Operation that applies gradients.
#The zip function returns a list of tuples, where the i-th tuple contains the i-th element
# from each of the argument sequences or iterables.
#The returned list is truncated in length to the length of the shortest argument sequence.
# In our case, it returns [ (W1_grad, W1), (W2_grad, W2) ] which are our (gradiant, variable) pairs
update_grads = adam.apply_gradients(zip(batch_grad, [W1, W2]))

#Compute the discounted reward_signal
#for i,val in enumerate(r) returns a list containing (counter, value) for each element of r
#That list is used to compute a vector of values with length = length of r
#Takes 1d float array of rewards and computes discounted reward
#    e.g. f([1, 1, 1], 0.99) -> [1, 0.99, 0.9801]
def discount_rewards(r, gamma=0.99):
    return np.array([val * (gamma ** i) for i, val in enumerate(r)])

#keep track of the total reward
reward_sum = 0

#this is used to initialize all th tf variables
init = tf.global_variables_initializer()

# Placeholders for our observations, outputs and rewards
#these are not tf placeholders
xs = np.empty(0).reshape(0,dimen)
ys = np.empty(0).reshape(0,1)
rewards = np.empty(0).reshape(0,1)

# Setting up our environment
sess = tf.Session()
#don't display an image from the gym environment
rendering = False
sess.run(init)

#get an initial obervation (4 values)
#this is a bad choice for a name because we have the tf placeholder "observations"
observation = env.reset()

# Placeholder for our gradients
#not a tf placeholder
gradients = np.array([np.zeros(var.get_shape()) for var in trainable_vars])

num_episodes = 10000
num_episode = 0

while num_episode < num_episodes:
    # Append the observations to our batch
    #reshapes oservation to be a column vector with one row
    #instead of a row vector with one column
    x = np.reshape(observation, [1, dimen])

    # Run the neural net to determine output
    #see that we are feeding the tf placeholder "observations"
    #with x, which is the latest observation reshaped
    tf_prob = sess.run(output, feed_dict={observations: x})

    # Determine the output as 0 or 1, based on our net, allowing for some randomness
    #this limits the possible actions to just 2 (0 or 1)
    #tf_prob is the probability that we want to take action 0
    #we could just choose y based on whether it is > 0.5
    #but we generate some randomness by comparing it to a random value instead
    y = 0 if tf_prob > np.random.uniform() else 1

    # Append the observations for learning
    #using vstack: xs top rows will be xs, next row will be x
    xs = np.vstack([xs, x])
    #append the outputs
    ys = np.vstack([ys, y])

    # Determine the oucome of our action
    observation, reward, done, _ = env.step(y)
    reward_sum += reward
    rewards = np.vstack([rewards, reward])

    if done:
        # Determine standardized rewards
        discounted_rewards = discount_rewards(rewards, gamma)
        discounted_rewards -= discounted_rewards.mean()
        discounted_rewards /= discounted_rewards.std()

        # Append gradients for this case to the overall running gradients
        #calculate the new gradients using the entire history of observations,
        #outputs (actions), and rewards
        #and append that to the non-tf matrix gradients
        gradients += np.array(sess.run(new_grads, feed_dict={observations: xs,
                                               input_y: ys,
                                               advantages: discounted_rewards}))

        # Clear out game variables
        xs = np.empty(0).reshape(0,dimen)
        ys = np.empty(0).reshape(0,1)
        rewards = np.empty(0).reshape(0,1)

        # Once batch full
        if num_episode % batch_size == 0:
            # Updated gradients
            #now that we have calculated the gradients, we can apply the adam optimizer
            sess.run(update_grads, feed_dict={W1_grad: gradients[0],
                                             W2_grad: gradients[1]})
            # Clear out gradients
            gradients *= 0

            # Print status
            print("Average reward for episode {}: {}".format(num_episode, reward_sum/batch_size))

            if reward_sum / batch_size > 200:
                print("Solved in {} episodes!".format(num_episode))
                break
            reward_sum = 0
        num_episode += 1
        observation = env.reset()

# See our trained bot in action

observation = env.reset()
observation
reward_sum = 0

while True:
    env.render()
    x = np.reshape(observation, [1, dimen])
    y = sess.run(output, feed_dict={observations: x})
    y = 0 if y > 0.5 else 1
    observation, reward, done, _ = env.step(y)
    reward_sum += reward
    if done:
        print("Total score: {}".format(reward_sum))
        break
