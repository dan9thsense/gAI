from __future__ import print_function, division
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

num_epochs = 10
total_series_length = 5000
truncated_backprop_length = 15 #how far to look back in time
state_size = 4  #number of batches to analyze simultaneously
num_classes = 2 # 0 or 1
echo_step = 3   # number of offset steps between input and answer
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
# placeholder sizes (5,15)
batchX_placeholder = tf.placeholder(tf.float32, [batch_size, truncated_backprop_length])
batchY_placeholder = tf.placeholder(tf.int32, [batch_size, truncated_backprop_length])

#init state size (5,4)
init_state = tf.placeholder(tf.float32, [batch_size, state_size])

#start with random weights, size (4,2), zero biases, size (1,2)
W2 = tf.Variable(np.random.rand(state_size, num_classes),dtype=tf.float32)
b2 = tf.Variable(np.zeros((1,num_classes)), dtype=tf.float32)

# unstack columns
#inputs_series = tf.split(1, truncated_backprop_length, batchX_placeholder)
#split up the input (batchX) values size (5,15) to 15 column vectors, each length 5, taking from
#in the first column vector, the first element is batchX_placeholder [0][0], next [1][]0], ... [4][0]
#the second column vector has batchX_placeholder =  [0][1], next [1][]1], ... [4][1]
#the last column vector has batchX_placeholder = [0][14], next [1][14]...[4][14]
#so if the values of batchX_placeholder were [[1,2,3,4,5], [6,7,8,9,10],....[56,57,58,59,60], ... [71,72,73,74,75 ] ]
#instead of just 0's and 1's,
#then inputs_series would have 15 column vectors = [ [ [1],[16],[31],[46],[61] ], [ [2], [17],[32],[47],[62] ],...
#[ [15],[30],[45],[60],[75] ] ]
#inputs_series has size (15,5,1)
#batchX_placeholder is size (5,15)
inputs_series = tf.split(batchX_placeholder, truncated_backprop_length, 1 )

#batchY_placeholder is size (5,15).  label series is (15,5) where the sets of 5 are taken from the columns
#of batchY_plaaceholder, just as they were taken for input_series from batchX_placeholder
labels_series = tf.unstack(batchY_placeholder, axis=1)

# Forward passes
#_, encoder_state = encoder_cell(encoder_inputs, encoder_state)
cell = tf.contrib.rnn.BasicRNNCell(state_size)
#states_series, current_state = tf.nn.dynamic_rnn(cell=cell, inputs=inputs_series, sequence_length=init_state)

#takes the input series and runs the graph, multiplying a batch of input
#with the current state (weights), then applying the tanh activation fcn
#to get an output, which it then uses as the new current state.
#repeat for all the batches of inputs
#states_series is all of these states appended together
states_series, current_state = tf.contrib.rnn.static_rnn(cell, inputs_series, init_state)

#send those rnn states into an output layer
#multiply the inputs (states) by the output layer weights, add the biases
logits_series = [tf.matmul(state, W2) + b2 for state in states_series] #Broadcasted addition
#apply the softmax activation function to get the network prediction
#predictions_series has shape (5,2)
predictions_series = [tf.nn.softmax(logits) for logits in logits_series]
#tf.slice(matrix,[startIndexDim0, startIndexDim1...], [numEntriesDim0, numEntriesDim1...]
actions = tf.slice(predictions_series,[0,0,0],[15,5,2])
actions = tf.reshape(actions, [75,2])
actions = tf.argmax(actions, 1)
actions = tf.cast(actions, tf.int32)
actions = tf.reshape(actions, [1,75])
reward_pl = tf.placeholder(tf.float32,shape=[1,75])
#losses1 = tf.multiply(actions, reward_pl)
#losses = predictions_series * answer_var
logits_loss = tf.slice(logits_series,[0,0,0],[15,5,2])
logits_loss = tf.reshape(logits_loss, [75,2])
#selected_logits = tf.slice(logits_loss,[0,1],[75,1])
selected_logits = []
for i in range(75):
    selected_logits.append(logits_loss[i][actions[0][i]])
#selected_logits = tf.reshape(selected_logits [75, -1])
rewards_loss = tf.reshape(reward_pl,[75,1])
selected_logits = tf.reshape(selected_logits,[75,1])
losses1 = tf.multiply(selected_logits, rewards_loss)
total_loss1 = tf.reduce_mean(losses1)

#reward += np.dot(single_output_series,answer)
"""
predict_series = tf.placeholder(tf.float32, [batch_size, truncated_backprop_length])
reward_series = tf.placeholder(tf.float32, [batch_size, truncated_backprop_length])

_series = []
for i in range(len(answers_series)):
    if answers_series[i] == predictArray[i]:
        reward = 1.0
    else:
        reward = 0.0
    reward_series.append(reward)
#reward series has shape (75)
reward_series = tf.cast(np.array(reward_series), dtype=tf.float32)
tf.shape(reward_series,[15,5])
print(reward_series)
print(predictions_series)
losses = [tf.matmul(reward_series, predict) for predict in predictions_series]
"""

losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels) for logits, labels in zip(logits_series,labels_series)]
total_loss = tf.reduce_mean(losses)

#train to minimize losses
train_step = tf.train.AdagradOptimizer(0.3).minimize(total_loss1)

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
        plt.bar(left_offset, 0.3 + batchX[batch_series_idx, :], width=1, color="blue")
        plt.bar(left_offset, 0.2 + batchY[batch_series_idx, :] * 0.7, width=1, color="red")
        plt.bar(left_offset, 0.1 + single_output_series * 0.4, width=1, color="green")

    plt.draw()
    plt.pause(0.0001)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
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

            #take all 5 rows along with columns 1500 to 1515, so batchX size is [75]
            batchX = x[:,start_idx:end_idx] #inputs
            batchY = y[:,start_idx:end_idx] #answers

            _predictions_series, _logits_series = sess.run([predictions_series, logits_series],
                 feed_dict={
                     batchX_placeholder:batchX,
                     batchY_placeholder:batchY,
                     init_state:_current_state})
            reward_series = []
            for batch_series_idx in range(5):
                one_hot_output_series = np.array(_predictions_series)[:, batch_series_idx, :]
                single_output_series = np.array([(1 if out[0] < 0.5 else 0) for out in one_hot_output_series])
                answer = batchY[batch_series_idx, :]
                #print('single_output_series = ', single_output_series)
                #print('answer = ', answer)
                for i in range(len(single_output_series)):
                 if single_output_series[i] == answer[i]:
                     reward = -1
                 else:
                     reward = 1
                 #print('single_output_series[i] = ', single_output_series[i])
                 #print('answer[i] = ', answer[i])
                 #print('i, reward = ', i, reward)
                 reward_series.append(reward)
            #print('reward series = ', reward_series)
            #reward_series = tf.shape(reward_series,[75,1])
            #print('logits = ', _logits_series)

            _selected_logits, _losses1, _total_loss1, _total_loss, _current_state, _predictions_series, _logits_series, _actions = sess.run(
                [selected_logits, losses1, total_loss1, total_loss, current_state, predictions_series, logits_series, actions],
                feed_dict={
                    batchX_placeholder:batchX,
                    batchY_placeholder:batchY,
                    init_state:_current_state,
                    reward_pl:[reward_series]
                })
            """
            #print('prediction_series = ', _predictions_series[0][0][0], _predictions_series[0][0][1] )
            #print('actions = ', _actions)
            print('selected_logits = ', _selected_logits)
            print('selected_logits shape = ', np.array(_selected_logits).shape)
            print('rewards_series shape = ', np.array(reward_series).shape)
            print('losses1 = ', _losses1)
            print('total_loss = ', _total_loss1)
            exit()
            """
            _train_step = sess.run(train_step,
                feed_dict={
                    batchX_placeholder:batchX,
                    batchY_placeholder:batchY,
                    init_state:_current_state,
                    reward_pl:[reward_series]
            })

            #tf.reshape(predictArray, [-1])
            #tf.reshape(answers_series, [-1])
            loss_list.append(_total_loss)

            if batch_idx%100 == 0:
                print("Step",batch_idx, "Loss", _total_loss)
                plot(loss_list, _predictions_series, batchX, batchY)

plt.ioff()
plt.show()
