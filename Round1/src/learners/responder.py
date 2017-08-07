# top class for learners

import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt

class Responder:
    def __init__(self):
        self.characters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', \
            'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', \
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', \
            'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', \
            '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', \
            ',', '.', '!', ';', '?', '-', ':', '"', ' ' ]
        self.specialCharacters = [ ',', '.', '!', ';', '?', '-', ':', '"', ' ' ]
        self.anCharacters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', \
            'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', \
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', \
            'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', \
            '1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
        self.numberCharacters = [ '1', '2', '3', '4', '5', '6', '7', '8', '9', '0' ]
        self.inputCharacter = '-'
        self.previousCharacter = '+'
        self.outputCharacter = '.'
        self.quietCharacter = ' ' # this is the correct response when we get an input that corrects our previous response
        self.netWasReset = True
        self.resetVariables = False
        self.done = False
        self.numResets = 0
        self.numActions = len(self.characters)
        self.numStates = len(self.characters)
        self.initialRandomActionProbability = 0.02
        self.e = self.initialRandomActionProbability
        self.learningRate = 0.001
        self.totalReward = 0
        self.rewardList = []
        self.currentState = 0
        self.previousState = 0
        self.reward = 0
        self.action = 0
        self.recordRewards = True
        self.plotResults = False
        self.useTensorBoard = True
        self.saveModels = True
        self.loadModels = True

    def initializeValues(self):
        self.currentState = 0
        self.previousState = 0
        self.reward = 0
        self.action = 0

    def resetWeights(self):
        self.resetVariables = True

    def closeTF(self):
        self.done = True
        self.getOutput()
        self.initializeValues()

    def rewardIn(self, reward):
        self.reward = reward
        self.totalReward += self.reward
        if self.recordRewards:
            self.rewardList.append(self.reward)

    def charOut(self, charIn):
        self.inputCharacter = charIn
        self.previousState = self.currentState
        #print("character in was = ", charIn)
        # find the index of the character
        for i in range(len(self.characters)):
            if self.characters[i] == charIn:
                self.currentState = i
                break
        self.outputCharacter = self.getOutput()
        #print("character out was = ", self.outputCharacter)
        return self.outputCharacter

    def plotReward(self, learnerName):
        if len(self.rewardList) > 2 and self.recordRewards and self.plotResults:
            print("plotting rewardList.  Its length = ", len(self.rewardList))
            plt.plot(self.rewardList)
            plt.title(learnerName)
            plt.show()
        else:
            print("not plotting rewardList")
            self.rewardList = []
            self.totalReward = 0

    def getOutput(self):
        #override in child classes
        return 0
