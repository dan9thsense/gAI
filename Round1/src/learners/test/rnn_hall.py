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
# // in python 3 means do integer division (floor the result)
# for total series length 50000, batch size 5, and truncated_backprop_length 15, num_batches = int(10000/15) = 666
num_batches = total_series_length//batch_size//truncated_backprop_length

def generateData():
    #outputs an array with 0's and 1's
    x = np.array(np.random.choice(2, total_series_length, p=[0.5, 0.5]))

    #y is x shifted by echo_step, this provides the answers, so we can use it for training
    y = np.roll(x, echo_step)
    #set the first echo_step values in y to zero
    y[0:echo_step] = 0

    #reshape into size (batch_size, total_series_length/batch_size)) e.g., 5 x 10000
    x = x.reshape((batch_size, -1))  # The first index changing slowest, subseries as rows
    y = y.reshape((batch_size, -1))

    return (x, y)

batchX_placeholder = tf.placeholder(tf.float32, [batch_size, truncated_backprop_length])
batchY_placeholder = tf.placeholder(tf.int32, [batch_size, truncated_backprop_length])

init_state = tf.placeholder(tf.float32, [batch_size, state_size])

W2 = tf.Variable(np.random.rand(state_size, num_classes),dtype=tf.float32)
b2 = tf.Variable(np.zeros((1,num_classes)), dtype=tf.float32)

# unstack columns
#inputs_series = tf.split(1, truncated_backprop_length, batchX_placeholder)
inputs_series = tf.split(batchX_placeholder, truncated_backprop_length, 1 )
labels_series = tf.unstack(batchY_placeholder, axis=1)

# Forward passes
#_, encoder_state = encoder_cell(encoder_inputs, encoder_state)
cell = tf.nn.rnn_cell.BasicRNNCell(state_size)
#states_series, current_state = tf.nn.dynamic_rnn(cell=cell, inputs=inputs_series, sequence_length=init_state)
states_series, current_state = tf.contrib.rnn.static_rnn(cell, inputs_series, init_state)
logits_series = [tf.matmul(state, W2) + b2 for state in states_series] #Broadcasted addition
predictions_series = [tf.nn.softmax(logits) for logits in logits_series]

losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels) for logits, labels in zip(logits_series,labels_series)]
total_loss = tf.reduce_mean(losses)

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

    #0 to 100
    for epoch_idx in range(num_epochs):
        x,y = generateData()
        _current_state = np.zeros((batch_size, state_size))

        print("New data, epoch", epoch_idx)

        #0 to 666: 666 batches, each uses 15 (truncated_backprop_length) columns = 10000 and 5 (batch_size) rows = 50000
        for batch_idx in range(num_batches):
            #for batch 100, this will be start_idx = 1500, end_idx = 1515
            start_idx = batch_idx * truncated_backprop_length
            end_idx = start_idx + truncated_backprop_length

            #take all 5 rows along with columns 1500 to 1515
            batchX = x[:,start_idx:end_idx] #inputs
            batchY = y[:,start_idx:end_idx] #answers

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
