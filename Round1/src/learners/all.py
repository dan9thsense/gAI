from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import random
from core.serializer import StandardSerializer, IdentitySerializer
from learners.base import BaseLearner

import tensorflow as tf
from tensorflow.contrib import rnn

hm_epochs = 3
n_classes = 68
batch_size = 128
chunk_size = 68
n_chunks = 1
rnn_size = 128


x = tf.placeholder('float', [None, n_chunks,chunk_size])
y = tf.placeholder('float')

class RandomCharacterLearner(BaseLearner):
    def __init__(self):
        self.inputCharacter = '*'
        self.lastCharacter = '+'
        self.outputCharacter = '='
        self.memory = ''
        self.foundCharacter = False
        self.teacher_stopped_talking = False
        self.numRewards = 0
        self.numTries = 0
        self.characterIndex = -1
        # the learner has the serialization hardcoded to
        # detect spaces
        self.serializer = StandardSerializer()
        self.silence_code = self.serializer.to_binary(' ')
        self.silence_i = 0
        self.characters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', \
          'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', \
          'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', \
          'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', \
          '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', \
          ',', '.', '!', ';', '?', '-', ' ']
        self.inputArray = [0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, \
          0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0]
        self.outputArray = [0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, \
          0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0]

    def recurrent_neural_network(x):
        layer = {'weights':tf.Variable(tf.random_normal([rnn_size,n_classes])),
                 'biases':tf.Variable(tf.random_normal([n_classes]))}

        x = tf.transpose(x, [1,0,2])
        x = tf.reshape(x, [-1, chunk_size])
        x = tf.split(x, n_chunks, 0)

        #lstm_cell = rnn_cell.BasicLSTMCell(rnn_size,state_is_tuple=True)
        #outputs, states = rnn.dynamic_rnn(lstm_cell, x, dtype=tf.float32)
        
        lstm_cell = rnn.BasicLSTMCell(rnn_size) 
        outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

        output = tf.matmul(outputs[-1],layer['weights']) + layer['biases']

        return output

    def train_neural_network(x):
        prediction = recurrent_neural_network(x)
        cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
        optimizer = tf.train.AdamOptimizer().minimize(cost)
        
        for i in n_classes:
          if self.characters[i] == self.inputCharacter:
            inputIndex = i
            break
            
        self.inputArray[inputIndex] = 1
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for epoch in range(hm_epochs):
                epoch_loss = 0
                for _ in range(int(mnist.train.num_examples/batch_size)):
                    #epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                    #epoch_x = epoch_x.reshape((batch_size,n_chunks,chunk_size))

                    _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                    epoch_loss += c

                print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            #print('Accuracy:',accuracy.eval({x:mnist.test.images.reshape((-1, n_chunks, chunk_size)), y:mnist.test.labels}))
        self.inputArray[inputIndex] = 0

    def reward(self, reward):
        # YEAH! Reward!!! Whatever...
        # Now this robotic teacher is going to mumble things again
        self.teacher_stopped_talking = False
        self.silence_i = 0
        self.memory = ''
        self.numTries += 1
        print('when input was: ', self.inputCharacter, 'we got ')
        if reward >= 0:
          if reward == 1:
            self.foundCharacter = True
            self.outputCharacter = self.lastCharacter
            print('rewarded for character: ', self.outputCharacter)
            self.numRewards += 1
            if self.numRewards > 100:
              print('exiting with ', self.numTries, ' tries', self.numRewards, 'rewards')  
              exit()
          else:
            print('no reward for character: ', self.lastCharacter)
            if self.numTries > 100:
              print('exiting with ', self.numTries, ' tries', self.numRewards, 'rewards') 
              exit()
        else:
          print('negative reward for character: ', self.lastCharacter)
          if self.numTries > 100:
            print('exiting with ', self.numTries, ' tries', self.numRewards, 'rewards') 
            exit()
          self.foundCharacter = False

    def next(self, input):
        self.inputCharacter = input
        #train_neural_network(x)
        #r = random.randint(0,25)
        if not self.foundCharacter:
          self.characterIndex += 1
        if self.characterIndex > 68:
          self.characterIndex = 0

        # If we have received a silence byte
        text_input = self.serializer.to_text(self.memory)
        if text_input and text_input[-2:] == '  ':
            self.teacher_stopped_talking = True
        if self.teacher_stopped_talking:
            # send the memorized sequence
            output, self.memory = self.memory[0], self.memory[1:]
        else:
            output = self.characters[self.characterIndex]
            self.lastCharacter = output
            self.silence_i = (self.silence_i + 1) % len(self.silence_code)
        # memorize what the teacher said
        self.memory += input
        return output
        
class SampleRepeatingLearner(BaseLearner):
    def reward(self, reward):
        # YEAH! Reward!!! Whatever...
        pass

    def next(self, input):
        # do super fancy computations
        # return our guess
        return input


class SampleSilentLearner(BaseLearner):
    def __init__(self):
        self.serializer = StandardSerializer()
        self.silence_code = self.serializer.to_binary(' ')
        self.silence_i = 0

    def reward(self, reward):
        # YEAH! Reward!!! Whatever...
        self.silence_i = 0

    def next(self, input):
        output = self.silence_code[self.silence_i]
        self.silence_i = (self.silence_i + 1) % len(self.silence_code)
        return output


class SampleMemorizingLearner(BaseLearner):
    def __init__(self):
        self.memory = ''
        self.teacher_stopped_talking = False
        # the learner has the serialization hardcoded to
        # detect spaces
        self.serializer = StandardSerializer()
        self.silence_code = self.serializer.to_binary(' ')
        self.silence_i = 0

    def reward(self, reward):
        # YEAH! Reward!!! Whatever...
        # Now this robotic teacher is going to mumble things again
        self.teacher_stopped_talking = False
        self.silence_i = 0
        self.memory = ''

    def next(self, input):
        # If we have received a silence byte
        text_input = self.serializer.to_text(self.memory)
        if text_input and text_input[-2:] == '  ':
            self.teacher_stopped_talking = True

        if self.teacher_stopped_talking:
            # send the memorized sequence
            output, self.memory = self.memory[0], self.memory[1:]
        else:
            output = self.silence_code[self.silence_i]
            self.silence_i = (self.silence_i + 1) % len(self.silence_code)
        # memorize what the teacher said
        self.memory += input
        return output

