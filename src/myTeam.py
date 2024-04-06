# myTeam.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game

##################
# Game Constants #
##################

# Set TRAINING to True while agents are learning, False if in deployment
# [!] Submit your final team with this set to False!
TRAINING = True

# Name of weights / any agent parameters that should persist between
# games. Should be loaded at the start of any game, training or otherwise
# [!] Replace MY_TEAM with your team name
WEIGHT_PATH = 'weights_MY_TEAM.json'

# Any other constants used for your training (learning rate, discount, etc.)
# should be specified here
# [!] TODO

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'DummyAgent', second = 'DummyAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.
  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class HungryAgent(CaptureAgent):

  def registerInitialState(self, gameState):
    super().registerInitialState(self, gameState)
    self.weights = util.Counter()

  def featureExtractor(self, gameState):
    
    state = gameState.getAgentState(self.index)
    position = state.getPosition()
    features = util.Counter()
    foodList = self.getFood(gameState).asList()

      # Compute distance to the nearest food

    if len(foodList) > 0:  # This should always be True,  but better safe than sorry
          minDistance = min([self.getMazeDistance(position, food)
                              for food in foodList])
          features['distanceToFood'] = minDistance
    
    enemies = [gameState.getAgentState(i)
                   for i in self.getOpponents(gameState)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition()
                    != None]
    
    if len(invaders) > 0:
            dists = [self.getMazeDistance(
                position, a.getPosition()) for a in invaders]
            features['invaderDistance'] = min(dists)
    
    else:
       features['invaderDistance'] = None
    return features

  def reward(self, previousState, action, currentState):
    #successor = gameState.generateSuccessor(self.index, action)
    features = self.featureExtractor(previousState)
    successorFeatures = self.featureExtractor(currentState)
    reward = 0
    reward += features['distanceToFood'] - successorFeatures['distanceToFood']

    if features['invaderDistance'] != None and successorFeatures['invaderDistance'] != None:
        reward += features['invaderDistance'] - successorFeatures['invaderDistance']
      
    return reward
  
  def getQValue(self, gameState, action):
     qValue = sum([self.weights[feature] * value for feature, value in self.featureExtractor(gameState).items()])
     return qValue

  def chooseAction(self, gameState):
    legalActions = gameState.getLegalActions(self.index)
    bestQvalue = -4000
    bestAction = None

    for action in legalActions:
        actionValue = self.getQValue(gameState, action)

        if actionValue > bestQvalue:
            bestQvalue = actionValue
            bestAction = action
    return bestAction


class DummyAgent(CaptureAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """

  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).
    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)
    IMPORTANT: This method may run for at most 15 seconds.
    """

    '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''
    CaptureAgent.registerInitialState(self, gameState)

    '''
    Your initialization code goes here, if you need any.
    '''


  def chooseAction(self, gameState):
    """
    Picks among actions randomly.
    """
    actions = gameState.getLegalActions(self.index)

    '''
    You should change this in your own agent.
    '''

    return random.choice(actions)
