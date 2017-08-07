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
#from learners import q_learner
#from learners import q_network
from learners import q_no_state_policy
from learners import agent
from learners import mdpPolicyWithTB


# when testing stand-alone
#from base import BaseLearner
#import learners.my_learners
#import rnn_learner
#import learners.q_learner

class GeneralLearner(BaseLearner):
  def __init__(self):
    self.enablePlotting = False
    self.quietCharacter = ' ' # this is the correct response when we get an input that corrects our previous response
    self.inputCharacter = '-'
    self.lastCharacter = '+'
    self.outputCharacter = '='
    self.memory = ''
    self.foundCharacter = False
    self.teacher_stopped_talking = False
    self.numRewards = 0
    self.rewardsNeededToFinishTask = 10
    self.numConsecutiveRewards = 0
    self.numConsecutiveFailures = 0
    self.numFailures = 0
    self.maxNumFailures = 3000
    self.numTries = 0
    self.maxTries = 3000
    self.learner = my_learners
    self.repeater = self.learner.Repeater()
    self.randomChar = self.learner.RandomCharacter()
    self.alphaNumeric = self.learner.alphaNumeric()
    self.inputOutputFeedback = self.learner.InputOutputFeedback()
    #self.rnn_learner = rnn_learner.myRNN()
    #self.qLearner = q_learner.myQ()
    #self.qLearner = q_network.myQNetwork()
    self.q_NoState = q_no_state_policy.NoStatePolicy()
    self.agent1 = agent.simpleAgent()
    self.mdpAgent = mdpPolicyWithTB.mdpPolicyAgent()

    # list of the learners with the max number of allowed failures for each
    # and slots for the number of tasks solved and the number of tasks failed for each
    self.learnerList = [ [self.mdpAgent, 300, 0, 0, 'mdpAgent'], [self.q_NoState, 71, 0, 0, 'q_NoState'],\
      [self.agent1, 300, 0, 0, 'agent1'],\
      [self.repeater, 3, 0, 0, 'Repeater'], [self.randomChar, 71, 0, 0, 'Random Character'], \
      [self.alphaNumeric, 15, 0, 0, 'AlphaNumeric'], [self.inputOutputFeedback, 5, 0, 0, 'ioFeedback'] ]
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
    if self.numTries > self.maxTries:
      print('exiting with ', self.numTries, ' tries', self.numRewards, 'rewards')
      for i in range(len(self.learnerList)):
        print("learner ", i, " completed ", self.learnerList[i][2], " and failed ", self.learnerList[i][3], " tasks")
        if self.enablePlotting:
          self.learnerList[self.learnerIndex][0].plotReward(self.learnerList[self.learnerIndex][4])
      exit()

    print('when input was: ', self.inputCharacter, 'we got ')
    printChar = self.outputCharacter
    if self.outputCharacter == self.quietCharacter:
      printChar = "quietCharacter (space)"
    if myReward == -1:
      if self.numConsecutiveRewards >= self.rewardsNeededToFinishTask:
        self.numIndividualTasksCompleted += 1
        self.learnerList[self.learnerIndex][2] += 1
        print("failure after ", self.numConsecutiveRewards, " consecutive rewards for this learner, which means that")
        print(self.learnerList[self.learnerIndex][4], ", learner number ", self.learnerIndex, " completed this task")
        print("The number of tasks it has completed = ", self.learnerList[self.learnerIndex][2])
        self.numConsecutiveFailures = 0
        self.numConsecutiveRewards = 0
        self.numFailures = 0
        print(self.learnerList[self.learnerIndex][4], " succeeded with this task, so we'll use it again, assuming the task is repeated")
        print("closing tf session with a reset in ", self.learnerList[self.learnerIndex][4])
        self.learnerList[self.learnerIndex][0].resetWeights()
        #if self.enablePlotting:
        #    self.learnerList[self.learnerIndex][0].plotReward(self.learnerList[self.learnerIndex][4])
        #self.learnerIndex += 1
        return

      self.numConsecutiveRewards = 0
      self.numConsecutiveFailures += 1
      self.numFailures += 1
      if (self.numFailures > self.maxNumFailures) or (self.numConsecutiveFailures > self.learnerList[self.learnerIndex][1]):
        self.learnerList[self.learnerIndex][3] += 1
        print(self.learnerList[self.learnerIndex][4], ", learner number ", self.learnerIndex, " failed this task with ")
        if self.numConsecutiveFailures > self.learnerList[self.learnerIndex][1]:
          print(self.numConsecutiveFailures, " consecutive failures")
        else:
          print(self.numFailures, " non-consecutive failures")
        print("The number of tasks it has failed = ", self.learnerList[self.learnerIndex][3])

        # move on to next learner
        if self.enablePlotting:
          self.learnerList[self.learnerIndex][0].plotReward(self.learnerList[self.learnerIndex][4])
        # self.learnerList[self.learnerIndex][0].printQ()
        print("closing tf session with a reset in ", self.learnerList[self.learnerIndex][4])
        self.learnerList[self.learnerIndex][0].reset()
        self.learnerIndex += 1

        if self.learnerIndex >= len(self.learnerList):
          self.learnerIndex = 0
        print("moving on to ", self.learnerList[self.learnerIndex][4], ", learner number ", self.learnerIndex)
        self.numConsecutiveFailures = 0
        self.numFailures = 0
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