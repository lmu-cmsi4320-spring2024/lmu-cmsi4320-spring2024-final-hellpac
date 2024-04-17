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
import distanceCalculator
import heapq
from capture import GameState
from game import Directions
from game import Actions
from game import Grid
from game import AgentState
from dataclasses import dataclass, field
from typing import Any
import numpy as np
import game
import json
import inspect
import os.path
import math

##################
# Game Constants #
##################

# Set TRAINING to True while agents are learning, False if in deployment
# [!] Submit your final team with this set to False!
TRAINING = True

# Name of weights / any agent parameters that should persist between
# games. Should be loaded at the start of any game, training or otherwise
# [!] Replace MY_TEAM with your team name
WEIGHT_PATH = 'zweights_MY_TEAM.json'

TS_GRAVEYARD_PATH = 'zts_graveyard_MY_TEAM.json'
REGULAR_GRAVEYARD_PATH = 'zregular_graveyard_MY_TEAM.json'

KEEPGRAVEYARD = True

INITALTEAMWEIGHTS =  {"FirstAgent": util.Counter(), "SecondAgent": util.Counter()}
INITALTSGRAVEYARD = {"FirstAgent": dict[int: list[tuple()]](), "SecondAgent": dict[int: list[tuple()]]()}
INITALREGULARGRAVEYARD = {"FirstAgent": util.Counter(dict[str: int]()), "SecondAgent": util.Counter(dict[tuple[str, 1]: int]())}

MOVES = ["North", "South", "East", "West", "Stop"]

# Any other constants used for your training (learning rate, discount, etc.)
# should be specified here
# [!] TODO

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'FirstAgent', second = 'SecondAgent'):
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


class FirstAgent(CaptureAgent):
  def registerInitialState(self, gameState: GameState, training: bool = TRAINING) -> None:
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)
    
    self.weights = util.Counter() 
    self.loadWeights()
    
    self.discount=0.8
    self.learningRate=0.01
    self.epsilon = 0.2
    
    self.livingReward = -0.1
    
    self.useTSChance = 0
    self.isTraining = training
    
    self.movesTaken = 0
    self.initalFoodCount = len(self.getFood(gameState).asList())
    self.spawnLocation = gameState.getAgentPosition(self.index)
    self.enemyIndices = gameState.getBlueTeamIndices() if self.red else gameState.getRedTeamIndices()
    
    #Two Particle Filters for two enemies
    numOfParticles = 1000
    self.particleFilter1 = ParticleFilter(gameState, self.red, self.enemyIndices[0], numOfParticles)
    self.particleFilter2 = ParticleFilter(gameState, self.red, self.enemyIndices[1], numOfParticles)
    
    self.previousActionMemory = None
    
    self.debug = False
    self.getOrSetDebug()
    
    if (self.red):
      self.extractLocs = [(15, 1), (15, 2), (15, 4), (15, 5), (15, 7), (15, 8), (15, 11), (15, 12), (15, 13), (15, 14)]
    else:
      self.extractLocs = [(16, 1), (16, 2), (16, 3), (16, 4), (16, 7), (16, 8), (16, 10), (16, 11), (16, 13), (16, 14)]
    
    # legalPositions = [(0, 0)]
    # numOfParticles = 1000
    # self.particleFilter = ParticleFilter(gameState, legalPositions, numOfParticles)
    
    self.featuresList = list(self.getFeatures(gameState, gameState.getLegalActions(self.index)[0]).keys())

    if (self.isTraining):
      #Thompson Sampling
      self.armsGraveyard = dict[int: list[tuple()]]()
      
      #For Optimistic Sampling
      self.regularGraveyard = util.Counter(dict[int: int]())
      self.optimisticConstant = 0.1
    
      if (KEEPGRAVEYARD):
        self.loadGraveyard()
    
  def getOrSetDebug(self) -> str:
    if (self.index == 0): self.debug = True
    return "FirstAgent"
  
  def loadWeights(self) -> None:
    agentStr = self.getOrSetDebug()
    if os.path.isfile(WEIGHT_PATH):
      with open(WEIGHT_PATH, "r") as jsonFile:
        teamWeights = json.load(jsonFile)
      if agentStr not in teamWeights.keys():
        print("Initalized Team Weights without %s :(" % agentStr)
        teamWeights[agentStr] = self.weights
        with open(WEIGHT_PATH, "w") as jsonFile:
          json.dump(teamWeights, jsonFile)
      else:
        self.weights = util.Counter(teamWeights[agentStr])
    elif not os.path.isfile(WEIGHT_PATH):
      print("Creating new weights from INITALTEAMWEIGHTS")
      newTeamWeights = open(WEIGHT_PATH, "a+")
      newTeamWeights.write(json.dumps(INITALTEAMWEIGHTS))
      newTeamWeights.close()
      
  def updateWeights(self) -> None:
    agentStr = self.getOrSetDebug()
    with open(WEIGHT_PATH, "r") as jsonFile:
      teamWeights = json.load(jsonFile)
  
    teamWeights[agentStr] = self.weights
  
    with open(WEIGHT_PATH, "w") as jsonFile:
      json.dump(teamWeights, jsonFile)
      
  def loadGraveyard(self) -> None:
    agentStr = self.getOrSetDebug()
    
    #LOAD TS_GRAVEYARD
    if os.path.isfile(TS_GRAVEYARD_PATH):
      with open(TS_GRAVEYARD_PATH, "r") as jsonFile:
        teamTSGraveyards = json.load(jsonFile)
      if agentStr not in teamTSGraveyards.keys():
        print("Initalized teamTSGraveyards without %s :(" % agentStr)
        teamTSGraveyards[agentStr] = self.armsGraveyard
        with open(TS_GRAVEYARD_PATH, "w") as jsonFile:
          json.dump(teamTSGraveyards, jsonFile)
      else:
        self.armsGraveyard = util.Counter(teamTSGraveyards[agentStr])
    elif not os.path.isfile(TS_GRAVEYARD_PATH):
      print("Creating new teamTSGraveyards")
      newTeamTSGraveyards = open(TS_GRAVEYARD_PATH, "a+")
      newTeamTSGraveyards.write(json.dumps(INITALTSGRAVEYARD))
      newTeamTSGraveyards.close()
    
    #LOAD REGULAR_GRAVEYARD
    if os.path.isfile(REGULAR_GRAVEYARD_PATH):
      with open(REGULAR_GRAVEYARD_PATH, "r") as jsonFile:
        teamRegularGraveyards = json.load(jsonFile)
      if agentStr not in teamRegularGraveyards.keys():
        print("Initalized teamRegularGraveyards without %s :(" % agentStr)
        teamRegularGraveyards[agentStr] = self.regularGraveyard
        with open(REGULAR_GRAVEYARD_PATH, "w") as jsonFile:
          json.dump(teamRegularGraveyards, jsonFile)
      else:
        self.regularGraveyard = util.Counter(teamRegularGraveyards[agentStr])
    elif not os.path.isfile(REGULAR_GRAVEYARD_PATH):
      print("Creating new teamRegularGraveyards")
      newTeamRegularGraveyards = open(REGULAR_GRAVEYARD_PATH, "a+")
      newTeamRegularGraveyards.write(json.dumps(INITALREGULARGRAVEYARD))
      newTeamRegularGraveyards.close()
      
  def updateGraveyard(self) -> None:
    agentStr = self.getOrSetDebug()
    
    if (self.useTSChance != 0):
      #update ts graveyard
      with open(TS_GRAVEYARD_PATH, "r") as jsonFile:
        teamTSGraveyards = json.load(jsonFile)
    
      teamTSGraveyards[agentStr] = self.armsGraveyard
    
      with open(TS_GRAVEYARD_PATH, "w") as jsonFile:
        json.dump(teamTSGraveyards, jsonFile)
      
    #update regular graveyard
    with open(REGULAR_GRAVEYARD_PATH, "r") as jsonFile:
      teamRegularGraveyards = json.load(jsonFile)
  
    teamRegularGraveyards[agentStr] = self.regularGraveyard
  
    with open(REGULAR_GRAVEYARD_PATH, "w") as jsonFile:
      json.dump(teamRegularGraveyards, jsonFile)
  
  def getMoveIndex(self, move: str) -> int:
    for index, str in enumerate(MOVES):
      if move == str:
        return index
    print("Error Move Not Found in Constants.MOVES")
    return 0
  
  def getFeatureIndex(self, x: str) -> int:
    for index, feature in enumerate(self.featuresList):
      if x == feature:
        return index
    print("Feature: %s not found. Returning 0" % x)
    return 0
  
  def chooseAction(self, gameState: GameState) -> None:
    """
    Picks among the actions with the highest Q(s,a).
    """
    
    if (self.isTraining):
      #Temp saves weights into json file (change this later)
      self.updateWeights()
      #Temp saves graveyard into json file (change this later)
      self.updateGraveyard()
    
    # start = time.time()
    
    #Update Particle Filter to account for observed enemies
    nearbyEnemies = {}
    for opponent in self.getOpponents(gameState):
      nearbyEnemies[opponent] = gameState.getAgentState(opponent)   
      
    for index, enemy in nearbyEnemies.items():
      enemyLoc = enemy.getPosition()
      if enemyLoc != None:
        if self.particleFilter1.getTargetIndex() == index:
          self.particleFilter1.foundAgent(enemyLoc)
        elif self.particleFilter2.getTargetIndex() == index:
          self.particleFilter2.foundAgent(enemyLoc)
    
    #Update Particle Filter for noisy readings
    self.particleFilter1.elapseTime()
    self.particleFilter2.elapseTime()
    noisyReadings = [observed for observed in gameState.getAgentDistances()]
    self.particleFilter1.observe(noisyReadings[self.particleFilter1.getTargetIndex()], gameState, gameState.getAgentPosition(self.index))
    self.particleFilter2.observe(noisyReadings[self.particleFilter2.getTargetIndex()], gameState, gameState.getAgentPosition(self.index))
    
    if self.isTraining:      
      if self.debug: self.particleFilter1.getApproximateLocations()
      
      #if self.debug: print("Belief Distribution: \n%s\n" % str(self.particleFilter.getBeliefDistribution()))
      
      #Update weights and TS
      if (self.previousActionMemory != None):
        #Update weights and all that
        self.update(self.previousActionMemory[0], self.previousActionMemory[1], gameState, self.getReward(self.previousActionMemory[0], self.previousActionMemory[1], gameState))
      
      agentID = hash(gameState.getAgentPosition(self.index))
      foodID = None
      if (self.red): foodID = hash(gameState.getRedFood())
      else: foodID = hash(gameState.getBlueFood())
      currentArmBranch = hash((agentID, foodID))
      
      #Thompson Sampling when training
      if (self.useTSChance != 0):
        if currentArmBranch not in self.armsGraveyard.keys():
          legalActions = gameState.getLegalActions(self.index)
          
          self.armsGraveyard[currentArmBranch] = [(1, 1)] * len(MOVES) 
          
          for move in MOVES:
            if move not in legalActions: self.armsGraveyard[currentArmBranch][self.getMoveIndex(move)] = (0, 0)
      
      #Using a epsilon for usagecase of TS here it is self.useTSChance (I made this up hopefully it works)
      if random.random() < self.useTSChance and self.useTSChance != 0:
        chosenActionIndex = None
        highestDartScore = 0
        
        for actionIndex, ratio in enumerate(self.armsGraveyard[currentArmBranch]):
          if ratio != (0, 0):
            dartScore = np.random.beta(ratio[0], ratio[1])
            if dartScore > highestDartScore:
              chosenActionIndex = actionIndex
              highestDartScore = dartScore
            elif dartScore == highestDartScore and random.randint(0, 1) == 1:
              chosenActionIndex = actionIndex
              
        action = MOVES[chosenActionIndex]
      elif random.random() < self.epsilon:
        action = self.getRandomAction(gameState)
      else:
        action = self.getBestAction(gameState)
    else:
      action = self.getBestAction(gameState)
    self.previousActionMemory = (gameState, action)
    self.movesTaken += 1
    
    # print('eval time for agent %d: %.4f' % (self.index, time.time() - start))
    
    # if self.debug: print("Deadend?: %s" % self.checkIfDeadEnd(gameState, action))

    return action
  
  def getRandomAction(self, state: GameState) -> str:
    legalActions = state.getLegalActions(self.index)
    
    if len(legalActions) == 0:
      return ""  # Terminal state, no legal actions
    
    return random.choice(legalActions)
  
  def getBestAction(self, state: GameState) -> str:
    """
      Compute the best action to take in a state.
    """

    legalActions = state.getLegalActions(self.index)

    if len(legalActions) == 0:
      return ""  # Terminal state, no legal actions

    bestActions = []
    bestQValue = float('-inf')

    for action in legalActions:
      qValue = self.getQValue(state, action)
      if qValue > bestQValue:
        bestActions = [action]
        bestQValue = qValue
      elif qValue == bestQValue:
        bestActions.append(action)
      elif bestQValue == float('-inf'):
        bestActions.append(action)
    
    return random.choice(bestActions)
  
  def getMaxQ_SA(self, state: GameState) -> float:
    """
      Returns max_action Q(state,action)
      where the max is over legal actions.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return a value of 0.0.
    """
    agentID = hash(state.getAgentPosition(self.index))
    foodID = None
    if (self.red): foodID = hash(state.getRedFood())
    else: foodID = hash(state.getBlueFood())
    currentArmBranch = hash((agentID, foodID))
    
    legalActions = state.getLegalActions(self.index)
    if not legalActions:
      return 0.0  # Terminal state, no legal actions

    maxQValue = max(self.getQValue(state, action) +  self.optimisticConstant/(self.regularGraveyard["(%s, %s)" % (currentArmBranch, action)] + 1) for action in legalActions)
    return maxQValue
  
  def update(self, state: GameState, action: str, nextState: GameState, rewards: dict[str: float]):
    """
      Updates the weights, Also updates the thompson samples
    """
    reward = self.getOverallReward(rewards)
    
    maxFutureActionQ_sa = self.getMaxQ_SA(nextState)
    q_sa = self.getQValue(state, action)
    
    featureDifference = util.Counter()
    
    for feature in rewards:
      #if self.debug: print("featureDifference[%s]:  %s  = (%s + ( %s * %s)) - %s" % (feature, (rewards[feature] + (self.discount * maxFutureActionQ_sa)) - q_sa, rewards[feature], self.discount, maxFutureActionQ_sa, q_sa))
      if rewards[feature] != 404: featureDifference[feature] = (rewards[feature] + (self.discount * maxFutureActionQ_sa)) - q_sa
      else: featureDifference[feature] = (reward + (self.discount * maxFutureActionQ_sa)) - q_sa
        
    featureDict = self.getFeatures(state, action)
    for feature in featureDict.keys():
      self.weights[feature] += (self.learningRate * featureDifference[feature] * featureDict[feature]/2)
    
    agentID = hash(state.getAgentPosition(self.index))
    foodID = None
    if (self.red): foodID = hash(state.getRedFood())
    else: foodID = hash(state.getBlueFood())
    currentArmBranch = hash((agentID, foodID))
    
    if (self.useTSChance != 0):
      #Thompson Sampling Update
      moveIndex = self.getMoveIndex(action)
      oldRatio = self.armsGraveyard[currentArmBranch][moveIndex] 
      newRatio = (oldRatio[0] + reward, oldRatio[1]) if reward >= 0 else (oldRatio[0], oldRatio[1] + abs(reward) * 2)
      self.armsGraveyard[currentArmBranch][moveIndex] = newRatio
    
    #For Optimistic Sampling
    self.regularGraveyard["(%s, %s)" % (currentArmBranch, action)] += 1
    
    # if self.debug:
      # print("For Agent %s, %s" % (self.index, self.getOrSetDebug()))
      # print("Last State Location: %s,  Last Action: %s(%sth Repitition)" % (state.getAgentPosition(self.index), action, repetitionCount))
      # print("Overall Reward From State, Action: %s" % reward)
      # print("maxFutureActionQ_sa: %s" % maxFutureActionQ_sa)
      # print("q_sa: %s" % q_sa)
      # print("Features: %s" % self.getFeatures(state, action))
      # print("Weights: %s" % self.weights)
      #print("Thompson Values: %s" % self.armsGraveyard)
      # print()
    
  def getReward(self, state: GameState, action : str, nextState: GameState) -> dict[str: float]:
    """
      Returns reward when given a state, action, and nextState. Should only be called when training
    """
    if not self.isTraining:
      print("WTF error why did u call getReward when ur not training. Returning 0")
      return 0
    
    featuresAtState = self.getFeatures(state, action)
    
    #For Reward Splitting
    reward = {}
    for feature in self.featuresList:
      featureValue = featuresAtState[feature]
      #Please have feature reward functions reserved for all features in the following format. feature = hi, function = hiReward()
      
      #Ignore bottom code was desperate to make this somewhat modular I could have done this all with if statements but here we are
      featureRewardArgs = eval('inspect.getfullargspec(self.%sReward).args' % feature)[1:]
      arg1 = comma1 = arg2 = comma2 = arg3 = " "
      match (len(featureRewardArgs)):
        case 2:
          comma1 = ","
        case 3:
          comma1 = ","
          comma2 = ","
      for index, arg in enumerate(featureRewardArgs):
        ldict = {}
        exec("arg%s = '%s=%s'" % (index +1, arg, arg), globals(), ldict)
        match index:
          case 0:
            arg1 = ldict['arg1']
          case 1:
            arg2 = ldict['arg2']
          case 2:
            arg3 = ldict['arg3']
      
      reward[feature] = eval('self.%sReward(%s%s%s%s%s)' % (feature, arg1, comma1, arg2, comma2, arg3))
    
    return reward
  
  def getOverallReward(self, rewardDict: dict[str: float]) -> float:
    #For Optimistic Sampling
    
    overallReward = 0
    
    #For now I am just gonna add the rewards for overall reward (NOTE subject to change)
    for reward in rewardDict.keys():
      if rewardDict[reward] != 404:
        overallReward += rewardDict[reward]
      
    return overallReward - self.livingReward

  def getSuccessor(self, gameState: GameState, action: str) -> GameState:
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != util.nearestPoint(pos):
        # Only half a grid position was covered
        return successor.generateSuccessor(self.index, action)
    else:
        return successor

  def getQValue(self, gameState: GameState, action: str) -> float:
    """
    Computes a linear combination of features and feature weights
    """
    return self.getFeatures(gameState, action) * self.weights

  def customHash1(self, x: tuple) -> float:
    return x[0]/31 if self.red else (31 - x[0])/31
  
  def customHash2(self, x: tuple) -> float:
    return x[1]/16 if self.red else (16 - x[1])/16

  def checkIfReturning(self, oldPos: tuple, myPos: tuple, gameState: GameState) -> bool:
    if self.red:
      return oldPos[0] > myPos[0]
    else:
      return oldPos[0] < myPos[0]
    
  def checkIfDeadEnd(self, gameState: GameState, action: str) -> bool:
    initalLoc = gameState.getAgentPosition(self.index)
    currentLoc = self.getSuccessor(gameState, action).getAgentPosition(self.index)
    
    #After you finish particle filterer do this
    
    return True

  def checkIfGhostAhead(self, gameState: GameState, action: str) -> bool:
    return True

  def checkIfGhost(self, isRed: bool, loc: tuple) -> bool:
    if isRed: return loc[0] >= 16
    else: return loc[0] <= 15

  def getFeatures(self, gameState: GameState, action: str) -> dict[str: float]:
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    pacmanState = successor.getAgentState(self.index)
    oldFoodList = self.getFood(gameState).asList()
    foodList = self.getFood(successor).asList()
    features['collectedFood'] = 1 if len(foodList) > len(oldFoodList) else 0
    features['lostFood'] = 1 if len(foodList) < len(oldFoodList) else 0
    features['gotScore'] = 1 * pacmanState.numCarrying if successor.getScore() > gameState.getScore() else 0
    features['winMove'] = 1 if successor.isOver() else 0
    
    oldPos = gameState.getAgentState(self.index).getPosition()
    myPos = successor.getAgentState(self.index).getPosition()
    
    features['died'] = 1 if oldPos == self.spawnLocation else 0
    if pacmanState.numCarrying > 0 and self.checkIfReturning(oldPos, myPos, gameState) and pacmanState.isPacman:
      features['returningFoodToBase'] = 0.25 * pacmanState.numCarrying
    elif pacmanState.numCarrying > 0 and not self.checkIfReturning(oldPos, myPos, gameState) and pacmanState.isPacman:
      features['returningFoodToBase'] = -0.25 * pacmanState.numCarrying
    else:
      features['returningFoodToBase'] = 0
      
    #if self.debug and self.movesTaken > 0: print(str([location for location in self.particleFilter1.getApproximateLocations()]))
    
    if len(oldFoodList) > 0 and len(foodList) > 0:  # This should always be True,  but better safe than sorry
      oldMinFoodDistance = min([self.getMazeDistance(oldPos, food) if oldPos != food else 100 for food in oldFoodList])
      newMinFoodDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['closerToFood'] = 0 if newMinFoodDistance >= oldMinFoodDistance else 1
      # features['distanceToFood'] = newMinFoodDistance
    
    enemyLocs = {}
    for enemyIndex in self.getOpponents(gameState):
      if enemyIndex != None:
        if self.particleFilter1.getTargetIndex() == enemyIndex:
          enemyLocs[enemyIndex] = [(loc, prob) for loc, prob in self.particleFilter1.getGhostApproximateLocations().items()]
        elif self.particleFilter2.getTargetIndex() == enemyIndex:
          enemyLocs[enemyIndex] = [(loc, prob) for loc, prob in self.particleFilter2.getGhostApproximateLocations().items()]
    
    oldMinEnemyDistance = 100
    oldAvgEnemyDistance = 0
    currentMinEnemyDistance = 100
    currentAvgEnemyDistance = 0
    for enemyIndex in enemyLocs:
      for locProbTuple in enemyLocs[enemyIndex]:
        oldEnemyDistance = self.distancer.getDistance(oldPos, locProbTuple[0])
        oldAvgEnemyDistance += oldEnemyDistance * locProbTuple[1] /2
        
        if oldEnemyDistance < oldMinEnemyDistance:
          oldMinEnemyDistance = oldEnemyDistance
        
        currentEnemyDistance = self.distancer.getDistance(myPos, locProbTuple[0])
        currentAvgEnemyDistance += currentEnemyDistance * locProbTuple[1] /2
        if currentEnemyDistance < currentMinEnemyDistance:
          currentMinEnemyDistance = currentEnemyDistance
    
    features['closerToGhost'] = oldAvgEnemyDistance - currentAvgEnemyDistance if (currentAvgEnemyDistance < oldAvgEnemyDistance) else 0
        
    if action == Directions.STOP: features['stop'] = 1
    else: features['stop'] = 0
    
    #Use Particle Filter
    # self.particleFilter
    
    return features

  def collectedFoodReward(self, featuresAtState: dict[str: float], featureValue: float) -> float:
    return 0.1 if featureValue > 0 else -0.001
  def lostFoodReward(self, featureValue: float) -> float:
    return -0.75 if featureValue > 0 else 0
  def gotScoreReward(self, featureValue: float) -> float:
    if featureValue > 0:
      #if (self.debug): print("returned to base with pellet")
      return featureValue
    else: 
      return -0.0025
  def winMoveReward(self, featureValue: float) -> float:
    if featureValue > 0:
      return 1
    else:
      return 0
  def diedReward (self, featureValue: float) -> float:
    if featureValue > 0 and self.movesTaken > 1:
      #if (self.debug): print(" I JUST FUCKING DIED ")
      return -2
    else:
      return 0
  def returningFoodToBaseReward(self, featureValue: float) -> float:
    if featureValue > 0:
      #print("returningFoodToBase")
      return 0.25
    elif featureValue < 0:
      #print("refusing to return to base ...")
      return -0.25
    else:
      return 0
  def closerToFoodReward(self, featuresAtState: dict[str: float], featureValue: float) -> float:
    if featuresAtState['died'] == 1: return 0
    elif featureValue == 1: return 0.01
    else: return 0
  def closerToGhostReward(self, featuresAtState: dict[str: float], featureValue: float) -> float:
    if featureValue > 0 and featuresAtState['onDefense'] == 1: return -0.025 * featureValue
    else: return 0 
  def walkedIntoGhostReward(self, featureValue: float) -> float:
    #DUMBASS
    return -1 if featureValue > 0 else 0
  def stopReward(self, featureValue: float) -> float:
    #STOP WASITNG TIME STOPPING NO TIME FOR STOP
    return -0.5 if featureValue > 0 else -0.01
  
class SecondAgent(FirstAgent):
  #PACMAN CHASER PROFESSIONAL
  def getOrSetDebug(self) -> str:
    if (self.red): self.debug = False
    return "SecondAgent"
  
  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    
    oldPos = gameState.getAgentState(self.index).getPosition()
    oldState = gameState.getAgentState(self.index)
    
    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()
    
    features['onDefense'] = 1
    if myState.isPacman:
        features['onDefense'] = 0
      
    features['gotOffDefense'] = 1 if myState.isPacman and not oldState.isPacman else 0
    
    features['died'] = 1 if oldPos == self.spawnLocation else 0
    
    enemyLocs = {}
    for enemyIndex in self.getOpponents(gameState):
      if enemyIndex != None:
        if self.particleFilter1.getTargetIndex() == enemyIndex:
          enemyLocs[enemyIndex] = [(loc, prob) for loc, prob in self.particleFilter1.getPacmanApproximateLocations().items()]
        elif self.particleFilter2.getTargetIndex() == enemyIndex:
          enemyLocs[enemyIndex] = [(loc, prob) for loc, prob in self.particleFilter2.getPacmanApproximateLocations().items()]
    
    oldMinEnemyDistance = 100
    oldAvgEnemyDistance = 0
    currentMinEnemyDistance = 100
    currentAvgEnemyDistance = 0
    for enemyIndex in enemyLocs:
      for locProbTuple in enemyLocs[enemyIndex]:
        oldEnemyDistance = self.distancer.getDistance(oldPos, locProbTuple[0])
        oldAvgEnemyDistance += oldEnemyDistance * locProbTuple[1] /2
        
        if oldEnemyDistance < oldMinEnemyDistance:
          oldMinEnemyDistance = oldEnemyDistance
        
        currentEnemyDistance = self.distancer.getDistance(myPos, locProbTuple[0])
        currentAvgEnemyDistance += currentEnemyDistance * locProbTuple[1] /2
        if currentEnemyDistance < currentMinEnemyDistance:
          currentMinEnemyDistance = currentEnemyDistance
    
    features['closerToGhost'] = oldAvgEnemyDistance - currentAvgEnemyDistance if (currentAvgEnemyDistance < oldAvgEnemyDistance) else 0
    
    center = (15, 7) if self.red else (16, 7)
    oldCenterDistance = self.distancer.getDistance(oldPos, center)
    newCenterDistance = self.distancer.getDistance(myPos, center)
    
    features['closerToCenter'] = 0.01 if newCenterDistance < oldCenterDistance else 0
    
    # if len(ghosts) > 0:
    #   oldMinGhostDistance = min([self.getMazeDistance(oldPos , a.getPosition()) if oldPos != a.getPosition() else 100 for a in ghosts]) + 1
    #   newMinGhostDistance = min([self.getMazeDistance(myPos, a.getPosition()) for a in ghosts]) + 1
    #   if self.red: print(newMinGhostDistance)
    #   features['closerToGhost'] = 0 if newMinGhostDistance >= oldMinGhostDistance else 1
    #   features['distanceToGhost'] = newMinGhostDistance
    #   features['rightNextToGhost'] = 1 if newMinGhostDistance <= 1 else 0
    #   features['walkedIntoGhost'] = 1 if newMinGhostDistance == 0 else 0
        
    if action == Directions.STOP: features['stop'] = 1
    else: features['stop'] = 0
    
    #Use Particle Filter
    # self.particleFilter

    return features
  
  def onDefenseReward(self, featureValue: float) -> float:
    return 0.05 if featureValue == 1 else -0.05
  def gotOffDefenseReward(self, featureValue: float) -> float:
    return -2 if featureValue > 0 else 0
  def diedReward(self, featureValue: float) -> float:
    return -2 if featureValue > 0 else 0
  def closerToGhostReward(self, featuresAtState: dict[str: float], featureValue: float) -> float:
    if featureValue > 0 and featuresAtState['onDefense'] == 1: return 0.025 * featureValue
    else: return 0 
  def closerToCenterReward(self, featureValue: float) -> float:
    if featureValue > 0: return 0.00005 * featureValue
    else: return 0
  def walkedIntoGhostReward(self, featuresAtState: dict[str: float], featureValue: float) -> float:
    return 10 if featureValue > 0 and featuresAtState['onDefense'] == 1 else 0
  def stopReward(self, featureValue: float) -> float:
    #STOP WASITNG TIME STOPPING NO TIME FOR STOP
    return -0.5 if featureValue > 0 else -0.01

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

#Helper Classes

class ParticleFilter():
  """
  A particle filter for approximately tracking a single ghost.
  """

  def __init__(self, gameState: GameState, isRed: bool, index: int, numParticles: int=300) -> None:
    self.numParticles = numParticles
    self.beliefs = util.Counter()
    self.targetIndex = index
    self.spawnLoc = gameState.getInitialAgentPosition(self.targetIndex)
    self.walls = gameState.getWalls()
    self.red = isRed
    
    allLocations = Grid(32, 16, True).asList()
    self.legalPositions = []
    for location in allLocations:
      if self.walls[location[0]][location[1]] == False:
        self.legalPositions.append(location)
    
    self.initializeUniformly()
    
    self.distancer = distanceCalculator.Distancer(gameState.data.layout)
      
  def getPositionDistribution(self, enemyLocation: tuple) -> util.Counter:
    """
    Returns a equally distributed distribution over successor positions of a given location.
    """
    newLocations = Actions.getLegalNeighbors(enemyLocation, self.walls)
    dist = util.Counter()
    possibleNumberOfMoves = len(newLocations)
    stopProb = (possibleNumberOfMoves**2 -1)/100
    avgMoveProb = (100 - stopProb)/(possibleNumberOfMoves -1)
    for location in newLocations:
      dist[location] = avgMoveProb
    return dist

  def initializeUniformly(self) -> None:
    """
    Initializes a list of particles.
    """
    uniformedParticles = [None] * self.numParticles
    for particleNum in range(0, self.numParticles):
      newParticle = self.legalPositions[particleNum % len(self.legalPositions)]
      uniformedParticles[particleNum] = newParticle
    
    self.beliefs = util.Counter()
    for particlePosition in uniformedParticles:
      self.beliefs[particlePosition] = self.beliefs[particlePosition] + 1 if self.beliefs[particlePosition] != None else 1
        
    self.beliefs.normalize()
    
    return uniformedParticles 
  
  def getObservationDistribution(self, noisyDistance: float, gameState: GameState, myPos: tuple) -> dict[int: float]:
    dist = {}
    for possibleEnemyLoc in self.legalPositions:
      distanceToLoc = self.distancer.getDistance(possibleEnemyLoc, myPos)
      dist[distanceToLoc] = gameState.getDistanceProb(distanceToLoc, noisyDistance)
        
    return dist
  
  def observe(self, noisyDistance: float, gameState: GameState, myPos: tuple[int]) -> dict[tuple[int]: float]:
    """
    Update beliefs based on the given distance observation.
    """
    emissionModel = self.getObservationDistribution(noisyDistance, gameState, myPos)
    
    if self.beliefs.totalCount() == 0:
      self.initializeUniformly()
        
    for position in self.beliefs.keys():
      self.beliefs[position] = emissionModel[self.distancer.getDistance(position, myPos)] * self.beliefs[position]
    
    self.beliefs.normalize()
    return self.beliefs
  
  def killedAgent(self) -> None:
    """
    Update beliefs for when player eats the enemy
    """
    for loc in self.legalPositions:
      self.beliefs[loc] = 0.0
    self.beliefs[self.spawnLoc] = 1.0
  
  def foundAgent(self, foundLocation: tuple[int]) -> None:
    """
    Update beliefs for when player finds the enemy location
    """
    for loc in self.beliefs:
      self.beliefs[loc] = 0.0
    self.beliefs[foundLocation] = 1.0
    
  def elapseTime(self) -> None:
    """
    Update beliefs for a time step elapsing.
    """
    bprime = util.Counter() 
    for oldPos in self.legalPositions: 
      newPosDist = self.getPositionDistribution(oldPos) 
      for newPos, prob in newPosDist.items():
        bprime[newPos] = self.beliefs[oldPos] * prob + bprime[newPos] 

    bprime.normalize()
    self.beliefs = bprime

  def getBeliefDistribution(self) -> util.Counter:
    """
    Return the agent's current belief state, a distribution over ghost
    locations conditioned on all evidence and time passage. This method
    essentially converts a list of particles into a belief distribution (a
    Counter object)
    """

    return self.beliefs
  
  def getTargetIndex(self) -> int:
    """
    Return the particle filter's target agent index
    """
    return self.targetIndex
  
  def getApproximateLocations(self, count: int = 50) -> util.Counter():
    """
    Return the approximated locations in dict{location: probability}. Put an int into count to change how much locations you want (default is 10)
    """
    locationArray = []
    for location, prob in self.beliefs.items():
      if location != None and prob != None and prob > 0:
        locationArray.append(HeapNode(prob, location))
          
    largestNodes = heapq.nlargest(count, locationArray)
    topLocs = util.Counter()
    for node in largestNodes:
      if node.prob > 0: topLocs[node.loc] = node.prob
      
    return topLocs
  
  def getPacmanApproximateLocations(self, count: int = 50) -> util.Counter():
    """
    Return the approximated locations when the enemy is in Pacman State in dict{location: probability}. Put an int into count to change how much locations you want (default is 10)
    """
    locationArray = []
    for location, prob in self.beliefs.items():
      if location != None and prob != None and prob > 0:
        if (not self.red and location[0] >= 16) or (self.red and location[0] <= 15): locationArray.append(HeapNode(prob, location))
    
    largestNodes = heapq.nlargest(count, locationArray)
    topLocs = util.Counter()
    for node in largestNodes:
      if node.prob > 0: topLocs[node.loc] = node.prob
      
    return topLocs
  
  def getGhostApproximateLocations(self, count: int = 50) -> util.Counter():
    """
    Return the approximated locations when the enemy is in Ghost State in dict{location: probability}. Put an int into count to change how much locations you want (default is 10)
    """
    locationArray = []
    for location, prob in self.beliefs.items():
      if location != None and prob != None and prob > 0:
        if (self.red and location[0] >= 16) or (not self.red and location[0] <= 15): locationArray.append(HeapNode(prob, location))
    
    largestNodes = heapq.nlargest(count, locationArray)
    topLocs = util.Counter()
    for node in largestNodes:
      if node.prob > 0: topLocs[node.loc] = node.prob
      
    return topLocs

@dataclass(order=True)
class HeapNode:
  prob: float
  loc: Any=field(compare=False)