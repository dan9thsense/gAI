from __future__ import print_function, division
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

num_epochs = 100
total_series_length = 50000
truncated_backprop_length = 15
state_size = 4
num_classes = 2
echo_step = 3
batch_size = 5
num_batches = total_series_length//batch_size//truncated_backprop_length

def generateData():
    x = np.array(np.random.choice(2, total_series_length, p=[0.5, 0.5]))
    y = np.roll(x, echo_step)
    y[0:echo_step] = 0

    x = x.reshape((batch_size, -1))  # The first index changing slowest, subseries as rows
    y = y.reshape((batch_size, -1))

    return (x, y)

#input will be floats, size (batch_size, truncated_backprop_length) e.g., 5 x 15
batchX_placeholder = tf.placeholder(tf.float32, [batch_size, truncated_backprop_length])
#labels will be ints, size (batch_size, truncated_backprop_length) e.g., 5 x 15
batchY_placeholder = tf.placeholder(tf.int32, [batch_size, truncated_backprop_length])

#an individual state is size (batch_size, state_size), e.g. 5 x 4
init_state = tf.placeholder(tf.float32, [batch_size, state_size])

#populate W with random uniformly distributed entries
#W has rows = state_size+1 because the input is concatenated with the current state, so it has state_size+1 columns
W = tf.Variable(np.random.rand(state_size+1, state_size), dtype=tf.float32)
b = tf.Variable(np.zeros((1,state_size)), dtype=tf.float32)

W2 = tf.Variable(np.random.rand(state_size, num_classes),dtype=tf.float32)
b2 = tf.Variable(np.zeros((1,num_classes)), dtype=tf.float32)

# unstack columns, now we will have 15 vectors, with 5 entries each
inputs_series = tf.unstack(batchX_placeholder, axis=1) #
labels_series = tf.unstack(batchY_placeholder, axis=1)

# Forward pass
current_state = init_state #an individual state is size (batch_size, state_size), e.g. 5 x 4
states_series = []
for current_input in inputs_series: #15 sets with 5 points each
    #send in a size (5,) get out size (5,1)
    #remember, np.array([1,2,3]).shape = (3,) while np.array([[1],[2],[3]]).shape = (3,1)
    current_input = tf.reshape(current_input, [batch_size, 1])
    #send in size (5,1) and (5,4) get out size (5,5) This is state_size+1 number of columns
    input_and_state_concatenated = tf.concat([current_input, current_state], 1)  # Increasing number of columns

    #calculate activationFunction(Wx +b) for the hidden layer, this is next_state for the hidden layer
    next_state = tf.tanh(tf.matmul(input_and_state_concatenated, W) + b)  # Broadcasted addition
    states_series.append(next_state)
    current_state = next_state

#calculate W2x + b for every state we have produced
logits_series = [tf.matmul(state, W2) + b2 for state in states_series] #Broadcasted addition
#run the logits through the softmax activation, so we get outputs that are probabilities
predictions_series = [tf.nn.softmax(logits) for logits in logits_series]

#use logits and labels to calculate losses for each state we have produced
losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels) for logits, labels in zip(logits_series,labels_series)]
#get the mean of the losses over all the states we produced
total_loss = tf.reduce_mean(losses)

#minimize that loss to train the net
train_step = tf.train.AdagradOptimizer(0.3).minimize(total_loss)

def plot(loss_list, predictions_series, batchX, batchY):
    plt.subplot(2, 3, 1)
    plt.cla()
    plt.plot(loss_list)

    for batch_series_idx in range(5):
        one_hot_output_series = np.array(predictions_series)[:, batch_series_idx, :]
        single_output_series = np.array([(1 if out[0] < 0.5 else 0) for out in one_hot_output_series])

        plt.subplot(2, 3, batch_series_idx + 2)
        plt.cla()
        plt.axis([0, truncated_backprop_length, 0, 2])
        left_offset = range(truncated_backprop_length)
        plt.bar(left_offset, batchX[batch_series_idx, :], width=1, color="blue")
        plt.bar(left_offset, batchY[batch_series_idx, :] * 0.5, width=1, color="red")
        plt.bar(left_offset, single_output_series * 0.3, width=1, color="green")

    plt.draw()
    plt.pause(0.0001)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer)
    plt.ion()
    plt.figure()
    plt.show()
    loss_list = []

    for epoch_idx in range(num_epochs): #0 to 100
        x,y = generateData()
        _current_state = np.zeros((batch_size, state_size)) #_current_state starts with all 0's

        print("New data, epoch", epoch_idx)

        for batch_idx in range(num_batches): #0 to 666
            start_idx = batch_idx * truncated_backprop_length #for idx=100, start_idx = 1500
            end_idx = start_idx + truncated_backprop_length     #for idx=100, end_idx=1515

            #take all 5 rows along with columns 1500 to 1515
            batchX = x[:,start_idx:end_idx] #inputs
            batchY = y[:,start_idx:end_idx] #labels

            _total_loss, _train_step, _current_state, _predictions_series = sess.run(
                [total_loss, train_step, current_state, predictions_series],
                feed_dict={
                    batchX_placeholder:batchX,
                    batchY_placeholder:batchY,
                    init_state:_current_state
                })

            loss_list.append(_total_loss)

            if batch_idx%100 == 0:
                print("Step",batch_idx, "Loss", _total_loss)
                plot(loss_list, _predictions_series, batchX, batchY)

plt.ioff()
plt.show()
