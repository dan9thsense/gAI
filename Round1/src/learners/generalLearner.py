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
#import random
#from core.serializer import StandardSerializer, IdentitySerializer
from learners.base import BaseLearner
from learners import my_learners

class GeneralLearner(BaseLearner):
  def __init__(self):
    self.quietCharacter = ' ' # this is the correct response when we get an input that corrects our previous response
    self.inputCharacter = '-'
    self.lastCharacter = '+'
    self.outputCharacter = '='
    self.memory = ''
    self.foundCharacter = False
    self.teacher_stopped_talking = False
    self.numRewards = 0
    self.numConsecutiveRewards = 0
    self.numConsecutiveFailures = 0
    self.numTries = 0
    self.learner = my_learners
    self.repeater = self.learner.Repeater()
    self.randomChar = self.learner.RandomCharacter()
    self.inputOutputFeedback = self.learner.InputOutputFeedback()
    
    # list of the learners with the max number of allowed failures for each
    # and slots for the number of tasks solved and the number of tasks failed for each
    self.learnerList = [ [self.repeater, 3, 0, 0], [self.randomChar, 68, 0, 0],  [self.inputOutputFeedback, 5, 0, 0]]
    self.learnerIndex = 0
    self.individualTaskCompleted = False
    self.numIndividualTasksCompleted = 0
    #print("max number of consecutive failures for repeater = ", self.learnerList[self.learnerIndex][1])
    
      
  # we get an input character and send an output character in response
  def next(self, input):
    self.inputCharacter = input
    self.outputCharacter = self.learnerList[self.learnerIndex][0].charOut(self.inputCharacter)
    return self.outputCharacter
 
  # as a response to our output character, we get a reward value of -1, 0, or 1 
  def reward(self, myReward):
    self.numTries += 1
    if self.numTries > 100:
      print('exiting with ', self.numTries, ' tries', self.numRewards, 'rewards')
      for i in range(len(self.learnerList)):
        print("learner ", i, " completed ", self.learnerList[i][2], " and failed ", self.learnerList[i][3], " tasks")
      exit()
     
    print('when input was: ', self.inputCharacter, 'we got ')
    printChar = self.outputCharacter
    if self.outputCharacter == self.quietCharacter:
      printChar = "quietCharacter (space)"
    if myReward == -1:
      if self.numConsecutiveRewards >= 10:
        self.numConsecutiveFailures = 0
        self.numConsecutiveRewards = 0
        self.numIndividualTasksCompleted += 1
        self.learnerList[self.learnerIndex][2] += 1
        print("learner number ", self.learnerIndex, " completed this task") 
        print("The number of tasks it has completed = ", self.learnerList[self.learnerIndex][2])
        self.learnerIndex += 1
        if self.learnerIndex >= len(self.learnerList):
          self.learnerIndex = 0
        print("failure after many consecutive rewards for this learner, moving to learner number ", self.learnerIndex)
        return
          
      self.numConsecutiveRewards = 0
      self.numConsecutiveFailures += 1     
      if self.numConsecutiveFailures > self.learnerList[self.learnerIndex][1]:
        self.learnerList[self.learnerIndex][3] += 1
        print("that learner number ", self.learnerIndex, " failed this task with ", self.numConsecutiveFailures, "failures") 
        print("The number of tasks it has failed = ", self.learnerList[self.learnerIndex][3])
        
        print("moving on to learner number ", self.learnerIndex)
        self.learnerIndex += 1
        if self.learnerIndex >= len(self.learnerList):
          self.learnerIndex = 0
        self.numConsecutiveFailures = 0
        return
           
      print('negative reward for character: ', printChar)        

    elif myReward == 0:
        print('no reward for character: ', printChar)
        
    elif myReward == 1:
        print('rewarded for character: ', printChar)
        self.numConsecutiveFailures = 0
        self.numRewards += 1
        self.numConsecutiveRewards += 1
      
    self.learnerList[self.learnerIndex][0].rewardIn(myReward)
    
