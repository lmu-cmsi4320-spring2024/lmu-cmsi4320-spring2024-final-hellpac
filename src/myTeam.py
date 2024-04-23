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


import random, time, util
import distanceCalculator
import heapq
from capture import GameState
from game import Directions
from game import Actions
from game import Grid
from game import Agent
from game import AgentState
from dataclasses import dataclass, field
from util import nearestPoint
from typing import Any
import numpy as np
import game
import json
import inspect
import os.path
import math
import copy

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

class CaptureAgent(Agent):
    """
    A base class for capture agents.  The convenience methods herein handle
    some of the complications of a two-team game.

    Recommended Usage:  Subclass CaptureAgent and override chooseAction.
    """

    #############################
    # Methods to store key info #
    #############################

    def __init__(self, index, pf1, pf2, initalizingPf, timeForComputing=.1):
        """
        Lists several variables you can query:
        self.index = index for this agent
        self.red = true if you're on the red team, false otherwise
        self.agentsOnTeam = a list of agent objects that make up your team
        self.distancer = distance calculator (contest code provides this)
        self.observationHistory = list of GameState objects that correspond
            to the sequential order of states that have occurred so far this game
        self.timeForComputing = an amount of time to give each turn for computing maze distances
            (part of the provided distance calculator)
        """
        # Agent index for querying state
        self.index = index

        # Whether or not you're on the red team
        self.red = None

        # Agent objects controlling you and your teammates
        self.agentsOnTeam = None

        # Maze distance calculator
        self.distancer = None

        # A history of observations
        self.observationHistory = []

        # Time to spend each turn on computing maze distances
        self.timeForComputing = timeForComputing

        # Access to the graphics
        self.display = None
        
        self.initalizePF = initalizingPf
        
        #Two Particle Filters for two enemies
        self.particleFilter1: ParticleFilter = pf1
        self.particleFilter2: ParticleFilter = pf2

    def registerInitialState(self, gameState):
        """
        This method handles the initial setup of the
        agent to populate useful fields (such as what team
        we're on).

        A distanceCalculator instance caches the maze distances
        between each pair of positions, so your agents can use:
        self.distancer.getDistance(p1, p2)
        """
        self.red = gameState.isOnRedTeam(self.index)
        self.distancer = distanceCalculator.Distancer(gameState.data.layout)

        # comment this out to forgo maze distance computation and use manhattan distances
        self.distancer.getMazeDistances()

        import __main__
        if '_display' in dir(__main__):
            self.display = __main__._display

    def final(self, gameState):
        self.observationHistory = []

    def registerTeam(self, agentsOnTeam):
        """
        Fills the self.agentsOnTeam field with a list of the
        indices of the agents on your team.
        """
        self.agentsOnTeam = agentsOnTeam

    def observationFunction(self, gameState):
        " Changing this won't affect pacclient.py, but will affect capture.py "
        return gameState.makeObservation(self.index)

    def debugDraw(self, cells, color, clear=False):

        if self.display:
            from captureGraphicsDisplay import PacmanGraphics
            if isinstance(self.display, PacmanGraphics):
                if not type(cells) is list:
                    cells = [cells]
                self.display.debugDraw(cells, color, clear)

    def debugClear(self):
        if self.display:
            from captureGraphicsDisplay import PacmanGraphics
            if isinstance(self.display, PacmanGraphics):
                self.display.clearDebug()

    #################
    # Action Choice #
    #################

    def getAction(self, gameState):
        """
        Calls chooseAction on a grid position, but continues on half positions.
        If you subclass CaptureAgent, you shouldn't need to override this method.  It
        takes care of appending the current gameState on to your observation history
        (so you have a record of the game states of the game) and will call your
        choose action method if you're in a state (rather than halfway through your last
        move - this occurs because Pacman agents move half as quickly as ghost agents).

        """
        self.observationHistory.append(gameState)

        myState = gameState.getAgentState(self.index)
        myPos = myState.getPosition()
        if myPos != nearestPoint(myPos):
            # We're halfway from one position to the next
            return gameState.getLegalActions(self.index)[0]
        else:
            return self.chooseAction(gameState)

    #######################
    # Convenience Methods #
    #######################

    def getFood(self, gameState):
        """
        Returns the food you're meant to eat. This is in the form of a matrix
        where m[x][y]=true if there is food you can eat (based on your team) in that square.
        """
        if self.red:
            return gameState.getBlueFood()
        else:
            return gameState.getRedFood()

    def getFoodYouAreDefending(self, gameState):
        """
        Returns the food you're meant to protect (i.e., that your opponent is
        supposed to eat). This is in the form of a matrix where m[x][y]=true if
        there is food at (x,y) that your opponent can eat.
        """
        if self.red:
            return gameState.getRedFood()
        else:
            return gameState.getBlueFood()

    def getCapsules(self, gameState):
        if self.red:
            return gameState.getBlueCapsules()
        else:
            return gameState.getRedCapsules()

    def getCapsulesYouAreDefending(self, gameState):
        if self.red:
            return gameState.getRedCapsules()
        else:
            return gameState.getBlueCapsules()

    def getOpponents(self, gameState):
        """
        Returns agent indices of your opponents. This is the list of the numbers
        of the agents (e.g., red might be "1,3,5")
        """
        if self.red:
            return gameState.getBlueTeamIndices()
        else:
            return gameState.getRedTeamIndices()

    def getTeam(self, gameState):
        """
        Returns agent indices of your team. This is the list of the numbers
        of the agents (e.g., red might be the list of 1,3,5)
        """
        if self.red:
            return gameState.getRedTeamIndices()
        else:
            return gameState.getBlueTeamIndices()

    def getScore(self, gameState):
        """
        Returns how much you are beating the other team by in the form of a number
        that is the difference between your score and the opponents score.  This number
        is negative if you're losing.
        """
        if self.red:
            return gameState.getScore()
        else:
            return gameState.getScore() * -1

    def getMazeDistance(self, pos1, pos2):
        """
        Returns the distance between two points; These are calculated using the provided
        distancer object.

        If distancer.getMazeDistances() has been called, then maze distances are available.
        Otherwise, this just returns Manhattan distance.
        """
        d = self.distancer.getDistance(pos1, pos2)
        return d

    def getPreviousObservation(self):
        """
        Returns the GameState object corresponding to the last state this agent saw
        (the observed state of the game last time this agent moved - this may not include
        all of your opponent's agent locations exactly).
        """
        if len(self.observationHistory) == 1:
            return None
        else:
            return self.observationHistory[-2]

    def getCurrentObservation(self):
        """
        Returns the GameState object corresponding this agent's current observation
        (the observed state of the game - this may not include
        all of your opponent's agent locations exactly).
        """
        return self.observationHistory[-1]

    def displayDistributionsOverPositions(self, distributions):
        """
        Overlays a distribution over positions onto the pacman board that represents
        an agent's beliefs about the positions of each agent.

        The arg distributions is a tuple or list of util.Counter objects, where the i'th
        Counter has keys that are board positions (x,y) and values that encode the probability
        that agent i is at (x,y).

        If some elements are None, then they will be ignored.  If a Counter is passed to this
        function, it will be displayed. This is helpful for figuring out if your agent is doing
        inference correctly, and does not affect gameplay.
        """
        dists = []
        for dist in distributions:
            if dist != None:
                if not isinstance(dist, util.Counter):
                    raise Exception("Wrong type of distribution")
                dists.append(dist)
            else:
                dists.append(util.Counter())
        if self.display != None and 'updateDistributions' in dir(self.display):
            self.display.updateDistributions(dists)
        else:
            self._distributions = dists  # These can be read by pacclient.py

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
  if firstIndex == 1 or firstIndex == 3:
    enemyIndex = (2, 4)
  else:
    enemyIndex = (1, 3)
  
  particleFilter1 = ParticleFilter(isRed, enemyIndex[0])
  particleFilter2 = ParticleFilter(isRed, enemyIndex[1])
  return [eval(first)(firstIndex, particleFilter1, particleFilter2, True), eval(second)(secondIndex, particleFilter1, particleFilter2, False)]

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
    
    self.discount=0.9
    self.learningRate=0.01
    self.epsilon = 0.2
    
    self.livingReward = -0.1
    
    self.useTSChance = 0
    self.isTraining = training
    
    self.movesTaken = 0
    self.initalFoodCount = len(self.getFood(gameState).asList())
    self.spawnLocation = gameState.getAgentPosition(self.index)
    self.enemyIndices = gameState.getBlueTeamIndices() if self.red else gameState.getRedTeamIndices()
    
    if self.initalizePF:
      self.particleFilter1.initalizeGivenGameState(gameState)
      self.particleFilter2.initalizeGivenGameState(gameState)
    
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
    
    firstAction = gameState.getLegalActions(self.index)[0]
    firstState = self.getSuccessor(gameState, firstAction)
    self.featuresList = list(self.getFeatures(gameState, firstAction, firstState, self.particleFilter1, self.particleFilter2))

    if (self.isTraining):
      #Thompson Sampling
      self.armsGraveyard = dict[int: list[tuple()]]()
      
      #For Optimistic Sampling
      self.regularGraveyard = util.Counter(dict[int: int]())
      self.optimisticConstant = 0.1
    
      if (KEEPGRAVEYARD):
        self.loadGraveyard()
    
  def getOrSetDebug(self) -> str:
    if (self.index == 0): self.debug = False
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
    
    if self.debug and self.red: start = time.time()
    
    
    
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
    if self.initalizePF:
      self.particleFilter1.elapseTime()
      self.particleFilter2.elapseTime()
      noisyReadings = [observed for observed in gameState.getAgentDistances()]
      self.particleFilter1.observe(noisyReadings[self.particleFilter1.getTargetIndex()], gameState, gameState.getAgentPosition(self.index))
      self.particleFilter2.observe(noisyReadings[self.particleFilter2.getTargetIndex()], gameState, gameState.getAgentPosition(self.index))
    
    if self.debug and self.red:
      first = True
      approxLocs = self.particleFilter1.getApproximateLocations()
      
      for loc in approxLocs:
        if first:
          self.debugDraw(loc, [0, 0, approxLocs[loc]], True)
          first = False
        else:
          self.debugDraw(loc, [0, 0, approxLocs[loc]], False)

      approxLocs = self.particleFilter2.getApproximateLocations()
      for loc in approxLocs:
        self.debugDraw(loc, [0, approxLocs[loc], approxLocs[loc]], False)
    
    if not self.debug and self.red: print("\n\n\n-----------------------------------Move %s-----------------------------------\n"% self.movesTaken)
    
    if (self.isTraining):
      #Temp saves weights into json file (change this later)
      self.updateWeights()
      #Temp saves graveyard into json file (change this later)
      self.updateGraveyard()
         
      #if self.debug: print("Belief Distribution: \n%s\n" % str(self.particleFilter.getBeliefDistribution()))
      
      #Update weights and TS
      if (self.previousActionMemory != None):
        featureDict = self.getFeatures(self.previousActionMemory[0], self.previousActionMemory[1], gameState, self.particleFilter1, self.particleFilter2)
        
        #Update weights and all that
        self.update(self.previousActionMemory[0], self.previousActionMemory[1], gameState, self.getReward(self.previousActionMemory[0], self.previousActionMemory[1], gameState, featureDict), featureDict)
      
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
        if self.red and self.debug: print("Got Random Action: %s" % action)
      else:
        action = self.getBestAction(gameState)
        if self.red and self.debug: print("Got Best Action: %s" % action)
    else:
      action = self.getBestAction(gameState)
    self.previousActionMemory = (gameState, action)
    self.movesTaken += 1
    
    # if self.debug: print("Deadend?: %s" % self.checkIfDeadEnd(gameState, action))

    if self.debug and self.red: print('eval time for chooseAction %d: %.4f' % (self.index, time.time() - start))
    
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

    if self.red and self.debug: print("\nGetting Best Action:")
    for action in legalActions:
      qValue = self.getQValue(state, action)
      if self.red and self.debug: print("Action: %s,  QValue: %s" % (action, qValue))
      if qValue > bestQValue:
        bestActions = [action]
        bestQValue = qValue
      elif qValue == bestQValue:
        bestActions.append(action)
      elif bestQValue == float('-inf'):
        bestActions.append(action)
    
    return random.choice(bestActions)
  
  def getMaxQ_SA(self, oldPos: tuple, state: GameState) -> float:
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
    
    optimalValuesFeature = self.getOptimalValues(oldPos, state.getAgentPosition(self.index), state, self.particleFilter1, self.particleFilter2)
    
    legalActions = state.getLegalActions(self.index)
    if not legalActions:
      return 0.0  # Terminal state, no legal actions

    maxQValue = max(self.getQValue(state, action) + optimalValuesFeature[action] * self.weights['onOptimalPath'] + self.optimisticConstant/(self.regularGraveyard["(%s, %s)" % (currentArmBranch, action)] + 1) for action in legalActions)
    return maxQValue
  
  def update(self, state: GameState, action: str, nextState: GameState, rewards: dict[str: float], featureDict: util.Counter):
    """
      Updates the weights, Also updates the thompson samples
    """
    reward = self.getOverallReward(rewards)
    
    maxFutureActionQ_sa = self.getMaxQ_SA(state.getAgentPosition(self.index), nextState)
    q_sa = self.getQValue(state, action)
    
    featureDifference = util.Counter()
    
    for feature in rewards:
      if rewards[feature] != 404: featureDifference[feature] = (rewards[feature] + (self.discount * maxFutureActionQ_sa)) - q_sa
      else: featureDifference[feature] = (reward/100 + (self.discount * maxFutureActionQ_sa)) - q_sa
    
    if self.red and self.debug: print("\nUpdating Features,  maxQ`_s`a`: %s,  Q_sa: %s" % (maxFutureActionQ_sa, q_sa))
    
    for feature in featureDict.keys():
      if self.red and self.debug and abs(self.learningRate * featureDifference[feature] * featureDict[feature]) > 0: print("%s Update: = %s = difference{%s} * featureValue{%s} * learningRate{%s}" % (feature, str(self.learningRate * featureDifference[feature] * featureDict[feature]), featureDifference[feature], featureDict[feature], self.learningRate))
      self.weights[feature] += (self.learningRate * featureDifference[feature] * featureDict[feature])
    
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
    
  def getReward(self, state: GameState, action : str, nextState: GameState, featureDict: util.Counter) -> dict[str: float]:
    """
      Returns reward when given a state, action, and nextState. Should only be called when training
    """
    if not self.isTraining:
      print("WTF error why did u call getReward when ur not training. Returning None")
      return None
    
    featuresAtState = featureDict
    
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
    pf1 = copy.deepcopy(self.particleFilter1)
    pf2 = copy.deepcopy(self.particleFilter2)
    pf1.elapseTime()
    pf2.elapseTime()
    featureVector = self.getFeatures(gameState, action, self.getSuccessor(gameState, action), pf1, pf2, optimal=False) 
    return featureVector * self.weights

  def customHash1(self, x: tuple) -> float:
    return x[0]/31 if self.red else (31 - x[0])/31
  
  def customHash2(self, x: tuple) -> float:
    return x[1]/16 if self.red else (16 - x[1])/16

  def getOptimalValues(self, oldPos: tuple, myPos: tuple, gameState: GameState, pf1: any, pf2: any) -> util.Counter:
    """
    Returns action -> featureValue dictionary for onOptimalPath feature  (exists to optimize the use of this function)
    Basically same function as checkIfOptimal but it returns a util.Counter of the optimal values instead of a bool
    """
    
    self.checkIfOptimal(oldPos, myPos, gameState, pf1, pf2)
    
    return util.Counter()

  def checkIfOptimal(self, oldPos: tuple, myPos: tuple, gameState: GameState, pf1: any, pf2: any) -> bool:
    """
    An optimality checker for the agent that also takes into account enemy locations, pellet locations, capsule locations (Goal is to atleast get one pellet)
    """
    if self.debug and self.red: start = time.time()
    
    walls = gameState.getWalls()
    pellets = gameState.getRedFood() if self.red else gameState.getBlueFood()
    capsules = gameState.getRedCapsules() if self.red else gameState.getBlueCapsules()
    particleFilter1 = copy.deepcopy(pf1)
    particleFilter2 = copy.deepcopy(pf2)
    
    paths = [(oldPos, myPos, ), ]
    finishedPaths = list[tuple[tuple[int]]]()
    
    while(len(paths) != 0):
      newPaths = list[tuple[(0, 0)]]()
      for path in paths:
        neighbors = [neighbor for neighbor in Actions.getLegalNeighbors(path[-1], walls) if neighbor != path[-1] and neighbor not in path]
        for newLoc in neighbors:
          newPath = path + (newLoc,)
          if newLoc in self.extractLocs:
            finishedPaths.append(newPath)
          else:
            newPaths.append(newPath)
      
      paths = newPaths
      particleFilter1.elapseTime()
      particleFilter2.elapseTime()
      
      
      
    print("finishedPaths length: %s" % len(finishedPaths))
    
    if self.debug and self.red: print('eval time for checkIfOptimal %d: %.4f' % (self.index, time.time() - start))
    
    return True

  def getFeatures(self, gameState: GameState, action: str, nextState: GameState, pf1: any, pf2: any, optimal=True) -> util.Counter:
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    currentState = nextState.getAgentState(self.index)
    oldFoodList = self.getFood(gameState).asList()
    foodList = self.getFood(nextState).asList()
    features['collectedFood'] = 1 if len(foodList) > len(oldFoodList) else 0
    features['gotScore'] = 1 * currentState.numCarrying if nextState.getScore() > gameState.getScore() else 0
    features['lostFood'] = 1 if len(foodList) < len(oldFoodList) and features['gotScore'] != 1 else 0
    features['winMove'] = 1 if nextState.isOver() else 0
    
    features['onAttack'] = 0
    if currentState.isPacman:
        features['onAttack'] = 1
    
    oldPos = gameState.getAgentState(self.index).getPosition()
    myPos = nextState.getAgentState(self.index).getPosition()
    
    #if self.debug and self.red and optimal: self.checkIfOptimal(oldPos, myPos, gameState, pf1, pf2)
    
    features['died'] = 1 if oldPos == self.spawnLocation else 0
    # if currentState.isPacman and optimal:
    #   if self.checkIfOptimal(oldPos, myPos, gameState, pf1, pf2): features['onOptimalPath'] = 1 
    #   else: features['onOptimalPath'] = -1 
    # else:
    #   features['onOptimalPath'] = 0
      
    #if self.debug and self.movesTaken > 0: print(str([location for location in self.particleFilter1.getApproximateLocations()]))
    
    if len(oldFoodList) > 0 and len(foodList) > 0:  # This should always be True,  but better safe than sorry
      oldMinFoodDistance = min([self.getMazeDistance(oldPos, food) if oldPos != food else 100 for food in oldFoodList])
      newMinFoodDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['closerToFood'] = 0 if newMinFoodDistance >= oldMinFoodDistance else 1
      # features['distanceToFood'] = newMinFoodDistance
    
    enemyLocs = {}
    for enemyIndex in self.getOpponents(gameState):
      if enemyIndex != None:
        if pf1.getTargetIndex() == enemyIndex:
          enemyLocs[enemyIndex] = [(loc, prob) for loc, prob in pf1.getGhostApproximateLocations().items()]
        elif pf2.getTargetIndex() == enemyIndex:
          enemyLocs[enemyIndex] = [(loc, prob) for loc, prob in pf2.getGhostApproximateLocations().items()]
    
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
    
    if (currentAvgEnemyDistance < oldAvgEnemyDistance) and currentState.isPacman:
      features['closerToGhost'] = 1
    else: features['closerToGhost'] = 0
        
    if action == Directions.STOP: features['stop'] = 1
    else: features['stop'] = 0
    
    #Use Particle Filter
    # self.particleFilter
    
    return features
  
  def onAttackReward(self) -> float:
    #This feature's effect will be reduced
    return 404
  def collectedFoodReward(self, featuresAtState: dict[str: float], featureValue: float) -> float:
    if self.debug and self.red and featureValue > 0: print("  event: Collected Food")
    return 0.1 if featureValue > 0 else -0.001
  def lostFoodReward(self, featureValue: float) -> float:
    if self.debug and self.red and featureValue > 0: print("  event: Lost Food")
    return -2 if featureValue > 0 else 0
  def gotScoreReward(self, featureValue: float) -> float:
    if featureValue > 0:
      if self.debug and self.redand and featureValue > 0: print(" event: Got Score")
      return featureValue
    else: 
      return -0.0025
  def winMoveReward(self, featureValue: float) -> float:
    if featureValue > 0:
      if self.debug and self.redand: print("  event: Got Win Move")
      return 1
    else:
      return 0
  def diedReward (self, featureValue: float) -> float:
    if featureValue > 0 and self.movesTaken > 1:
      #if (self.debug): print(" I JUST FUCKING DIED ")
      return -2
    else:
      return 0
  def onOptimalPathReward(self, featureValue: float) -> float:
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
    if featureValue > 0 and featuresAtState['onAttack'] == 1: return -0.05 * featureValue
    else: return 0 
  def stopReward(self, featureValue: float) -> float:
    #STOP WASITNG TIME STOPPING NO TIME FOR STOP
    return -0.5 if featureValue > 0 else -0.01
  
class SecondAgent(FirstAgent):

  
  #PACMAN CHASER PROFESSIONAL
  def getOrSetDebug(self) -> str:
    if (self.red): self.debug = True
    return "SecondAgent"
  
  def getMaxQ_SA(self, oldPos: tuple, state: GameState) -> float:
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

    maxQValue = max(self.getQValue(state, action) + self.optimisticConstant/(self.regularGraveyard["(%s, %s)" % (currentArmBranch, action)] + 1) for action in legalActions)
    return maxQValue
  
  def getQValue(self, gameState: GameState, action: str) -> float:
    """
    Computes a linear combination of features and feature weights
    """
    pf1 = copy.deepcopy(self.particleFilter1)
    pf2 = copy.deepcopy(self.particleFilter2)
    pf1.elapseTime()
    pf2.elapseTime()
    featureVector = self.getFeatures(gameState, action, self.getSuccessor(gameState, action), pf1, pf2) 
    return featureVector * self.weights
  
  def getFeatures(self, gameState: GameState, action: str, nextState: GameState, pf1: any, pf2: any) -> util.Counter:
    features = util.Counter()
        
    oldPos = gameState.getAgentState(self.index).getPosition()
    oldState = gameState.getAgentState(self.index)
    
    myState = nextState.getAgentState(self.index)
    myPos = myState.getPosition()
    
    features['onDefense'] = 1
    if myState.isPacman:
        features['onDefense'] = 0
      
    features['gotOffDefense'] = 1 if myState.isPacman and not oldState.isPacman else 0
    
    features['died'] = 1 if oldPos == self.spawnLocation else 0
    
    enemyOneLocs = [(loc, prob) for loc, prob in pf1.getPacmanApproximateLocations(count=10).items()]
    enemyTwoLocs = [(loc, prob) for loc, prob in pf2.getPacmanApproximateLocations(count=10).items()]
    
    enemyOneAvgDist = sum([prob * self.distancer.getDistance(myPos, loc) 
                                      if (self.red and loc[0] <= 15) or (not self.red and loc[0] >= 16) else 0 
                                      for loc, prob in enemyOneLocs])
    enemyTwoAvgDist = sum([prob * self.distancer.getDistance(myPos, loc) 
                                      if (self.red and loc[0] <= 15) or (not self.red and loc[0] >= 16) else 0 
                                      for loc, prob in enemyTwoLocs])
    
    if enemyOneAvgDist < enemyTwoAvgDist or enemyTwoAvgDist == 0:
      oldEnemyOneAvgDist = sum([prob * self.distancer.getDistance(oldPos, loc) 
                                      if (self.red and loc[0] <= 15) or (not self.red and loc[0] >= 16) else 0 
                                      for loc, prob in enemyOneLocs])
      features['closerToPacman'] = oldEnemyOneAvgDist - enemyOneAvgDist if (enemyOneAvgDist < oldEnemyOneAvgDist) else 0
    elif enemyTwoAvgDist < enemyOneAvgDist or enemyOneAvgDist == 0:
      oldEnemyTwoAvgDist = sum([prob * self.distancer.getDistance(oldPos, loc) 
                                      if (self.red and loc[0] <= 15) or (not self.red and loc[0] >= 16) else 0 
                                      for loc, prob in enemyTwoLocs])
      features['closerToPacman'] = oldEnemyTwoAvgDist - enemyTwoAvgDist if (enemyTwoAvgDist < oldEnemyTwoAvgDist) else 0
    
    
    
    # for enemyIndex in enemyLocs:
    #   for locProbTuple in enemyLocs[enemyIndex]:
    #     if (self.red and (locProbTuple[0][0] <= 15)) or (not self.red and (locProbTuple[0][0] >= 16)):
    #       oldEnemyDistance = self.distancer.getDistance(oldPos, locProbTuple[0])
    #       oldAvgEnemyDistance += oldEnemyDistance * locProbTuple[1] /2
          
    #       if oldEnemyDistance < oldMinEnemyDistance:
    #         oldMinEnemyDistance = oldEnemyDistance
          
    #       currentEnemyDistance = self.distancer.getDistance(myPos, locProbTuple[0])
    #       currentAvgEnemyDistance += currentEnemyDistance * locProbTuple[1] /2
    #       if currentEnemyDistance < currentMinEnemyDistance:
    #         currentMinEnemyDistance = currentEnemyDistance
    
    
    center = (15, 7) if self.red else (16, 7)
    oldCenterDistance = self.distancer.getDistance(oldPos, center)
    newCenterDistance = self.distancer.getDistance(myPos, center)
    
    features['closerToCenter'] = 0.01 if newCenterDistance < oldCenterDistance else 0
        
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
  def closerToPacmanReward(self, featuresAtState: dict[str: float], featureValue: float) -> float:
    if featureValue > 0 and featuresAtState['onDefense'] == 1: return 0.025 * featureValue
    else: return 0 
  def closerToCenterReward(self, featureValue: float) -> float:
    if featureValue > 0: return 0.000005 * featureValue
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

  def __init__(self, isRed: bool, index: int) -> None:
    self.beliefs = util.Counter()
    self.targetIndex = index
    self.red = isRed
    
    self.spawnLoc = None
    self.walls = None
    self.legalPositions = None
    self.distancer = None
    
  def initalizeGivenGameState(self, gameState: GameState) -> None:
    self.spawnLoc = gameState.getInitialAgentPosition(self.targetIndex)
    self.walls = gameState.getWalls()
    
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
    
    self.beliefs = util.Counter()
    for cord in self.legalPositions:
      self.beliefs[cord] = 0
      if cord == self.spawnLoc:
        self.beliefs[cord] = 1
        
    self.beliefs.normalize()
  
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
    
  def noAgentNear(self, playerLocation: tuple[int]) -> None:
    """
    Update beliefs for when player's personal detection does not detect enemy
    """
    nearbyTiles = []
    
    
    for loc in self.beliefs:
      if loc in nearbyTiles: self.beliefs[loc] = 0
    
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
  
