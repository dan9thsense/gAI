# a set of simple responders
import random
from responder import Responder

# this solves microtask A.2.1,  Outputting a subset of ASCII characters only
class RandomCharacter(Responder):
    def __init__(self):
        Responder.__init__(self)
        self.characterIndex = 0

    def getOutput(self):
        if self.reward != 1:
            self.characterIndex += 1
            if self.characterIndex >= len(self.characters):
                self.characterIndex = 0
            self.outputCharacter = self.characters[self.characterIndex]
        return self.outputCharacter

# this solves microtask A.2.2,  repeat special characters, otherwise output a particular alphanumeric character
class alphaNumeric(Responder):
    def __init__(self):
      Responder.__init__(self)
      self.characterIndex = 0

    def getOutput(self):
        for i in self.specialCharacters:
          if self.inputCharacter == i:
            return self.inputCharacter

        if self.reward != 1:
            self.characterIndex += 1
            if self.characterIndex >= len(self.anCharacters):
                self.characterIndex = 0
            self.outputCharacter = self.anCharacters[self.characterIndex]
        return self.outputCharacter

#microtask A.2.3 needs an agent to map inputs to outputs, we'll do that with a network.

# this solves microtask A.2.4, Copy input to output
class Repeater(Responder):
    def __init__(self):
        Responder.__init__(self)

    def getOutput(self):
        return self.inputCharacter

# this solves A.2.5.1
class InputOutputFeedback(Responder):
    def __init__(self):
      Responder.__init__(self)
      self.responses = [-1 for x in range(len(self.characters))]
      self.counter = 0
      self.steps = 0

    def getOutput(self):
        print(self.steps)
        self.steps += 1
        if self.done or self.resetVariables:
            self.responses = [-1 for x in range(len(self.characters))]
            self.counter = 0
            print("completed InputOutputFeedback in my_learners.py")
            self.resetVariables = False
            self.done = False
            return self.quietCharacter

        '''
        if self.counter < 3:
            if self.reward != 0:
                self.counter += 1
                return self.quietCharacter
            self.counter = 10
        elif self.counter != 10:
            #this task is not A.2.5.1, so we return an arbitrary value to fail, rather than wasting time
            print('ioFeedback is not able to solve this task')
            return 'b'
        '''

        if self.reward == 1:
            return self.quietCharacter

        if self.reward == 0:
            #we sent a quiet character in response to the answer input
            #and now we have a new input for evaluation
            if self.responses[self.currentState] != -1:
                #we already know the right answer
                return self.characters[self.responses[self.currentState]]
            #if we have not heard the answer to this input yet, we send the quiet character
            #because on the next turn, we should get a negative reward in response to this quiet char and
            #that will then give us the answer to this input
            return self.quietCharacter

        #we got a negative reward, so either we responded to an answer input with something other than the quiet char
        #or we sent the wrong response to a regular input

        if self.outputCharacter == self.quietCharacter:
            #we returned a quiet character and got a -1 reward, so we know
            #that the current state is the answer to the previous state input
            self.responses[self.previousState] = self.currentState
        return self.quietCharacter

# this solves A.2.5.2
class AlphaNumericIOFeedback(Responder):
    def __init__(self):
      Responder.__init__(self)
      self.responses = [-1 for x in range(len(self.characters))]
      self.counter = 0
      self.steps = 0

    def getOutput(self):
        print(self.steps)
        self.steps += 1
        if self.done or self.resetVariables:
            self.responses = [-1 for x in range(len(self.characters))]
            self.counter = 0
            print("completed InputOutputFeedback in my_learners.py")
            self.resetVariables = False
            self.done = False
            return self.quietCharacter

        for i in self.specialCharacters:
            if self.inputCharacter == i:
                return self.quietCharacter

        if self.reward == 1:
            return self.quietCharacter

        if self.reward == 0:
            #we sent a quiet character in response to the answer input
            #and now we have a new input for evaluation
            if self.responses[self.currentState] != -1:
                #we already know the right answer
                return self.characters[self.responses[self.currentState]]
            #if we have not heard the answer to this input yet, we send the quiet character
            #because on the next turn, we should get a negative reward in response to this quiet char and
            #that will then give us the answer to this input
            return self.quietCharacter

        #we got a negative reward, so either we responded to an answer input with something other than the quiet char
        #or we sent the wrong response to a regular input

        if self.outputCharacter == self.quietCharacter:
            #we returned a quiet character and got a -1 reward, so we know
            #that the current state is the answer to the previous state input
            self.responses[self.previousState] = self.currentState
        return self.quietCharacter
