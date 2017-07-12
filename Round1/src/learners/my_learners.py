# a set of simple responders
import random

class Responder:
  def __init__(self):
    self.characters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', \
        'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', \
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', \
        'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', \
        '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', \
        ',', '.', '!', ';', '?', '-', ' ']
    self.specialCharacters = [ ',', '.', '!', ';', '?', '-', ' ' ]
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
    
# this solves microtask A.2.4, Copy input to output    
class Repeater:
  def rewardIn(self, reward):
    pass 
    
  def charOut(self, charIn):
    self.outputCharacter = charIn
    return self.outputCharacter

# this solves microtask A.2.1,  Outputting a subset of ASCII characters only    
class RandomCharacter(Responder):
    def __init__(self):
      Responder.__init__(self)      
      self.characterIndex = -1
      self.foundCharacter = False
      
    def rewardIn(self, reward):
      if reward == 1:
        self.foundCharacter = True
      elif reward == -1:
        self.foundCharacter = False

    def charOut(self, charIn):
      self.inputCharacter = charIn
      # if we were rewarded, then foundCharacter will be true and we want to keep sending the same character
      # otherwise, move on to the next character
      if not self.foundCharacter:
        self.characterIndex += 1
      if self.characterIndex >= len(self.characters):
        self.characterIndex = 0
      
      self.outputCharacter = self.characters[self.characterIndex]
      return self.outputCharacter
      
class InputOutputFeedback(Responder):
    def __init__(self):
      Responder.__init__(self)
      self.reward = 0
      self.correctCharacter = [-1 for x in range(256)]
      
    def rewardIn(self, reward):
      self.reward = reward
      
    def charOut(self, charIn):
      self.inputCharacter = charIn
      if self.reward == 1:
        # this input character is telling us that we got the right answer
        # we got the right answer, record it
        self.correctCharacter[ord(self.previousCharacter)] = self.inputCharacter
        self.outputCharacter = self.quietCharacter
      elif self.reward == 0:
        # we got a new character to respond to
        # check to see if we know the right answer
        if self.correctCharacter[ord(self.inputCharacter)] >= 0:
          self.outputCharacter = self.correctCharacter[ord(self.inputCharacter)]
        else:
          # pick a random character
          r = random.randint(0,len(self.characters) -1)
          self.outputCharacter = self.characters[r]
      else:
        # we got the wrong answer or we spoke when we should have been quiet
        # if we were silent last time and still got a negative, then this input tells us what the correct answer is
        if self.outputCharacter == self.quietCharacter:
          self.correctCharacter[ord(self.previousCharacter)] = self.inputCharacter
        self.outputCharacter = self.quietCharacter
      
      self.previousCharacter = self.inputCharacter
      return self.outputCharacter 
        

