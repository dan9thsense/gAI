# Copyright (c) 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import random
from core.serializer import StandardSerializer, IdentitySerializer
from learners.base import BaseLearner

class RandomCharacterLearner(BaseLearner):
    def __init__(self):
        self.inputCharacter = '-'
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
        #r = random.randint(0,25)
        if not self.foundCharacter:
          self.characterIndex += 1
        if self.characterIndex > 68:
          self.characterIndex = 0
        characters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', \
          'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', \
          'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', \
          'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', \
          '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', \
          ',', '.', '!', ';', '?', '-', ' ']
        # If we have received a silence byte
        text_input = self.serializer.to_text(self.memory)
        if text_input and text_input[-2:] == '  ':
            self.teacher_stopped_talking = True
        if self.teacher_stopped_talking:
            # send the memorized sequence
            output, self.memory = self.memory[0], self.memory[1:]
        else:
            output = characters[self.characterIndex]
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
