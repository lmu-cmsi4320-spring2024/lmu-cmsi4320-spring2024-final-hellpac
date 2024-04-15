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
from particleFilter import ParticleFilter
import random, time, util
from capture import GameState
from game import Directions
from game import Actions
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
    
    self.useTSChance = 0.5
    self.isTraining = training
    
    self.movesTaken = 0
    self.initalFoodCount = len(self.getFood(gameState).asList())
    
    self.previousActionMemory = None
    
    self.debug = False
    self.getOrSetDebug()
    
    self.foodInPouch = 0
    
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
      
    #Particle Filter
    numOfParticles = 1000
    #List of all possible cords in tuples
    legalPositions = [(0,0)]
    self.particleFilter = ParticleFilter(gameState, legalPositions, numOfParticles)
    
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
    
    #Thompson Sampling when training
    if self.isTraining:
      #Update weights and TS
      if (self.previousActionMemory != None):
        self.update(self.previousActionMemory[0], self.previousActionMemory[1], gameState, self.getReward(self.previousActionMemory[0], self.previousActionMemory[1], gameState))
      
      agentID = hash(gameState.getAgentPosition(self.index))
      foodID = None
      if (self.red): foodID = hash(gameState.getRedFood())
      else: foodID = hash(gameState.getBlueFood())
      currentArmBranch = hash((agentID, foodID))
      
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
      if feature == 'redHering1' or feature == 'redHering2': featureDifference[feature] = reward
      elif rewards[feature] != 404: featureDifference[feature] = (rewards[feature] + (self.discount * maxFutureActionQ_sa)) - q_sa
      else: featureDifference[feature] = (reward + (self.discount * maxFutureActionQ_sa)) - q_sa
        
    featureDict = self.getFeatures(state, action)
    for feature in featureDict.keys():
      self.weights[feature] += (self.learningRate * featureDifference[feature])
    
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

  def getFeatures(self, gameState: GameState, action: str) -> dict[str: float]:
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    oldFoodList = self.getFood(gameState).asList()
    foodList = self.getFood(successor).asList()
    features['collectedFood'] = 1 if len(foodList) > len(oldFoodList) else 0
    features['lostFood'] = 1 if len(foodList) < len(oldFoodList) else 0
    features['gotScore'] = 1 if successor.getScore() > gameState.getScore() else 0
    features['winMove'] = 1 if successor.isOver() else 0
    features['redHering1'] = self.customHash1(successor.getAgentPosition(self.index))
    features['redHering2'] = self.customHash2(successor.getAgentPosition(self.index))
    
    oldPos = gameState.getAgentState(self.index).getPosition()
    myPos = successor.getAgentState(self.index).getPosition()
    if len(oldFoodList) > 0 and len(foodList) > 0:  # This should always be True,  but better safe than sorry
      oldMinFoodDistance = min([self.getMazeDistance(oldPos, food) if oldPos != food else 100 for food in oldFoodList])
      newMinFoodDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['closerToFood'] = 0 if newMinFoodDistance >= oldMinFoodDistance else 1
      # features['distanceToFood'] = newMinFoodDistance
    
    enemies = [successor.getAgentState(opponent) for opponent in self.getOpponents(successor)]
    ghosts = [a for a in enemies if not a.isPacman and a.getPosition() != None]
    if len(ghosts) > 0:
      oldMinGhostDistance = min([self.getMazeDistance(oldPos , a.getPosition()) if oldPos != a.getPosition() else 100 for a in ghosts]) + 1
      newMinGhostDistance = min([self.getMazeDistance(myPos, a.getPosition()) for a in ghosts]) + 1
      features['closerToGhost'] = 0 if newMinGhostDistance >= oldMinGhostDistance else 1
      features['distanceToGhost'] = newMinGhostDistance
      features['rightNextToGhost'] = 1 if newMinGhostDistance <= 1 else 0
      features['walkedIntoGhost'] = 1 if newMinGhostDistance == 0 else 0
        
    if action == Directions.STOP: features['stop'] = 1
    else: features['stop'] = 0
    
    #Use Particle Filter
    # self.particleFilter
    
    return features

  def redHering1Reward(self, featuresAtState: dict[str: float]) -> float:
    x = self.getOverallReward(featuresAtState)
    if x > 1:
      x = 1
    if x < -1:
      x = -1
    return x/10000
  def redHering2Reward(self, featuresAtState: dict[str: float]) -> float:
    x = self.getOverallReward(featuresAtState)
    if x > 1:
      x = 1
    if x < -1:
      x = -1
    return x/10000
  def collectedFoodReward(self, state: GameState, action : str, featureValue: float) -> float:
    return 0.5 if featureValue > 0 else -0.001
  def lostFoodReward(self, featureValue: float) -> float:
    return -1 if featureValue > 0 else 0
  def gotScoreReward(self, featureValue: float) -> float:
    return 2 if featureValue > 0 else -0.002
  def winMoveReward(self, featureValue: float) -> float:
    if featureValue > 0:
      print("I WON OMG")
      return 1
    else:
      return 0
  def closerToFoodReward(self, state: GameState, action : str, featureValue: float) -> float:
    if featureValue == 1: return 0.075
    else: return 0
  def distanceToFoodReward(self, state: GameState, action : str, featureValue: float) -> float:
    #reward = [log_0.99 of (x) + 300 ] / 400
    return (math.log(featureValue, 0.99) + 300)/200 if featureValue > 1 else 1.5
  def closerToGhostReward(self, state: GameState, action : str, featureValue: float) -> float:
    if featureValue == 1: return -0.075
    else: return 0
  def distanceToGhostReward(self, state: GameState, action : str, featureValue: float) -> float:
    return (-featureValue + 5)/10
  def rightNextToGhostReward(self, featureValue: float) -> float:
    #DUMBASS
    return -1 if featureValue > 0 else 0
  def walkedIntoGhostReward(self, featureValue: float) -> float:
    #DUMBASS
    return -1 if featureValue > 0 else 0
  def stopReward(self) -> float:
    #STOP WASITNG TIME STOPPING NO TIME FOR STOP
    return -0.5
  
class SecondAgent(FirstAgent):
  #PACMAN CHASER PROFESSIONAL
  def getOrSetDebug(self) -> str:
    if (self.red): self.debug = False
    return "SecondAgent"
  
  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    oldFoodList = self.getFood(gameState).asList()
    foodList = self.getFood(successor).asList()
    oldPos = gameState.getAgentState(self.index).getPosition()
    myPos = successor.getAgentState(self.index).getPosition()
    
    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()
    
    features['onDefense'] = 1
    if myState.isPacman:
        features['onDefense'] = 0
    
    if len(oldFoodList) > 0 and len(foodList) > 0:  # This should always be True,  but better safe than sorry
      oldMinFoodDistance = min([self.getMazeDistance(oldPos, food) if oldPos != food else 100 for food in oldFoodList])
      newMinFoodDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['closerToFood'] = 0 if newMinFoodDistance >= oldMinFoodDistance else 1
    
    features['gotScore'] = 1 if successor.getScore() > gameState.getScore() else 0
    
    enemies = [successor.getAgentState(opponent) for opponent in self.getOpponents(successor)]
    ghosts = [a for a in enemies if not a.isPacman and a.getPosition() != None]
    if len(ghosts) > 0:
      oldMinGhostDistance = min([self.getMazeDistance(oldPos , a.getPosition()) if oldPos != a.getPosition() else 100 for a in ghosts]) + 1
      newMinGhostDistance = min([self.getMazeDistance(myPos, a.getPosition()) for a in ghosts]) + 1
      features['closerToGhost'] = 0 if newMinGhostDistance >= oldMinGhostDistance else 1
      features['distanceToGhost'] = newMinGhostDistance
      features['rightNextToGhost'] = 1 if newMinGhostDistance <= 1 else 0
      features['walkedIntoGhost'] = 1 if newMinGhostDistance == 0 else 0
        
    if action == Directions.STOP: features['stop'] = 1
    else: features['stop'] = 0
    
    #Use Particle Filter
    # self.particleFilter

    return features
  
  def onDefenseReward(self, featureValue: float) -> float:
    return 0.01 if featureValue == 1 else -0.01
  
  def closerToFoodReward(self, state: GameState, action : str, featureValue: float) -> float:
    if featureValue == 1: return 0.0001
    else: return 0
    
  def gotScoreReward(self, featureValue: float) -> float:
    return 2 if featureValue > 0 else -0.002
  
  def closerToGhostReward(self, featuresAtState: dict[str: float], featureValue: float) -> float:
    if featureValue == 1 and featuresAtState['onDefense'] == 1: return 0.25
    else: return 0
  def distanceToGhostReward(self, featuresAtState: dict[str: float], featureValue: float) -> float:
    return (featureValue + 5)/5 if featuresAtState['onDefense'] == 1 else 0
  def rightNextToGhostReward(self, featuresAtState: dict[str: float], featureValue: float) -> float:
    return 1 if featureValue > 0 and featuresAtState['onDefense'] == 1 else 0
  def walkedIntoGhostReward(self, featuresAtState: dict[str: float], featureValue: float) -> float:
    return 10 if featureValue > 0 and featuresAtState['onDefense'] == 1 else 0
  def stopReward(self) -> float:
    #STOP WASITNG TIME STOPPING NO TIME FOR STOP
    return -0.5

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

