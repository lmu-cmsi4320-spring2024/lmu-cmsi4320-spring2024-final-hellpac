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

STARINGEPSILON = 0

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
               first = 'SecondAgent', second = 'FirstAgent'):
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

    self.livingReward = -0.1
    
    self.useTSChance = 0
    self.isTraining = training
    
    self.movesTaken = 0
    self.gameCount = 1
    self.initalFoodCount = len(self.getFood(gameState).asList())
    self.spawnLocation = gameState.getAgentPosition(self.index)
    self.enemyIndices = gameState.getBlueTeamIndices() if self.red else gameState.getRedTeamIndices()
    
    if self.initalizePF:
      self.particleFilter1.initalizeGivenGameState(gameState)
      self.particleFilter2.initalizeGivenGameState(gameState)
    
    self.previousActionMemory = None
    
    self.debug = False
    self.showNoise = False
    self.getOrSetDebug()
    
    if (self.red):
      self.extractLocs = [(15, 1), (15, 2), (15, 4), (15, 5), (15, 7), (15, 8), (15, 11), (15, 12), (15, 13), (15, 14)]
    else:
      self.extractLocs = [(16, 1), (16, 2), (16, 3), (16, 4), (16, 7), (16, 8), (16, 10), (16, 11), (16, 13), (16, 14)]
    
    self.featuresList = self.getFeatureList()
    self.postiveFeatures = self.getPositiveFeatures()
    self.negativeFeatures = self.getNegativeFeatures()
    
    self.averageEvalTime = []
    self.maxEvalTime = 0

    if (self.isTraining):
      #Thompson Sampling
      self.armsGraveyard = dict[int: list[tuple()]]()
      
      #For Optimistic Sampling
      self.regularGraveyard = util.Counter(dict[int: int]())
      self.optimisticConstant = 0.1
    
      if (KEEPGRAVEYARD):
        self.loadGraveyard()
    
  def getOrSetDebug(self) -> str:
    if (self.red): 
      self.debug = True
      self.showNoise = True
    return "FirstAgent"
  
  def getFeatureList(self) -> list[str]:
    return ['died', 'collectedFood', 'closerToGhost', 'closerToExtraction', 'closerToUnguardedFood', 'collectedCapsule', 'closerToCapsule', 'gotScore']
  def getPositiveFeatures(self) -> list[str]:
    return ['collectedFood', 'closerToExtraction', 'closerToUnguardedFood', 'collectedCapsule', 'closerToCapsule', 'gotScore']
  def getNegativeFeatures(self) -> list[str]:
    return ['died', 'closerToGhost']
  
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

  def clamp(self, n, min, max): 
    if n < min: 
      return min
    elif n > max: 
      return max
    else: 
      return n 
  
  def updateParticleFilters(self, gameState: GameState) -> None:
    #Update Particle Filter for noisy readings
    if self.initalizePF:
      self.particleFilter1.elapseTime()
      self.particleFilter2.elapseTime()
    
    #Update Particle Filter to account for nearby enemies
    nearbyEnemies = {}
    for opponent in self.getOpponents(gameState):
      nearbyEnemies[opponent] = gameState.getAgentState(opponent)   
      
    enemyOneFound = False
    enemyOneLocation = None
    enemyTwoFound = False
    enemyTwoLocation = None
    
    for index, enemy in nearbyEnemies.items():
      enemyLoc = enemy.getPosition()
      if enemyLoc != None:
        if self.particleFilter1.getTargetIndex() == index:
          self.particleFilter1.foundAgent(enemyLoc)
          enemyOneFound = True
          enemyOneLocation = enemyLoc
        elif self.particleFilter2.getTargetIndex() == index:
          self.particleFilter2.foundAgent(enemyLoc)
          enemyTwoFound = True
          enemyTwoLocation = enemyLoc
          
    enemyOneIsPacman = gameState.getAgentState(self.enemyIndices[0]).isPacman
    enemyTwoIsPacman = gameState.getAgentState(self.enemyIndices[1]).isPacman
    
    oldEnemyOneIsPacman = self.previousActionMemory[0].getAgentState(self.enemyIndices[0]).isPacman
    oldEnemyTwoIsPacman = self.previousActionMemory[0].getAgentState(self.enemyIndices[1]).isPacman
    
    #Update Particle Filter to account for observed/unobserved enemies
    if not enemyOneFound:
      self.particleFilter1.noAgentNear(gameState.getAgentPosition(self.index))
    if not enemyTwoFound:
      self.particleFilter2.noAgentNear(gameState.getAgentPosition(self.index))
    
    #Update Particle Filter to account for food changes
    foodLocation = None
    if (self.red):
      oldFood = self.previousActionMemory[0].getRedFood().asList()
      currentFood = gameState.getRedFood().asList()
      if len(oldFood) > len(currentFood):
        for location in oldFood:
          if location not in currentFood:
            foodLocation = location
    elif(not self.red):
      oldFood = self.previousActionMemory[0].getBlueFood().asList()
      currentFood = gameState.getBlueFood().asList()
      if len(oldFood) > len(currentFood):
        for location in oldFood:
          if location not in currentFood:
            foodLocation = location
    
    if foodLocation != None:
      if (enemyOneIsPacman and not enemyTwoIsPacman) or (enemyOneIsPacman and (enemyTwoFound and enemyTwoLocation != foodLocation)):
        if self.particleFilter1.targetIndex == self.enemyIndices[0]:
          self.particleFilter1.allParticlesToLoc(foodLocation)
        else:
          self.particleFilter2.allParticlesToLoc(foodLocation)
      elif (enemyTwoIsPacman and not enemyOneIsPacman) or (enemyTwoIsPacman and (enemyOneFound and enemyOneLocation != foodLocation)):
        if self.particleFilter2.targetIndex == self.enemyIndices[1]:
          self.particleFilter2.allParticlesToLoc(foodLocation)
        else:
          self.particleFilter1.allParticlesToLoc(foodLocation)
      elif (enemyOneIsPacman and enemyTwoIsPacman) and (not enemyOneFound and not enemyTwoFound):
        self.particleFilter1.halfParticlesToLoc(foodLocation)
        self.particleFilter2.halfParticlesToLoc(foodLocation)
    
    #Update Particle Filter to account for capsule changes
    capsuleLocation = None
    if (self.red):
      oldCapsules = self.previousActionMemory[0].getRedCapsules()
      currentCapsules = gameState.getRedCapsules()
      if len(oldCapsules) > len(currentCapsules):
        for location in oldCapsules:
          if location not in currentCapsules:
            capsuleLocation = location
    elif(not self.red):
      oldCapsules = self.previousActionMemory[0].getBlueCapsules()
      currentCapsules = gameState.getBlueCapsules()
      if len(oldCapsules) > len(currentCapsules):
        for location in oldCapsules:
          if location not in currentCapsules:
            capsuleLocation = location
    
    if capsuleLocation != None:
      if (enemyOneIsPacman and not enemyTwoIsPacman) or (enemyOneIsPacman and (enemyTwoFound and enemyTwoLocation != capsuleLocation)):
        if self.particleFilter1.targetIndex == self.enemyIndices[0]:
          self.particleFilter1.allParticlesToLoc(capsuleLocation)
        else:
          self.particleFilter2.allParticlesToLoc(capsuleLocation)
      elif (enemyTwoIsPacman and not enemyOneIsPacman) or (enemyTwoIsPacman and (enemyOneFound and enemyOneLocation != capsuleLocation)):
        if self.particleFilter2.targetIndex == self.enemyIndices[1]:
          self.particleFilter2.allParticlesToLoc(capsuleLocation)
        else:
          self.particleFilter1.allParticlesToLoc(capsuleLocation)
      elif (enemyOneIsPacman and enemyTwoIsPacman) and (not enemyOneFound and not enemyTwoFound):
        self.particleFilter1.halfParticlesToLoc(capsuleLocation)
        self.particleFilter2.halfParticlesToLoc(capsuleLocation)
    
    #Update Particle Filter based on isPacman information
    if oldEnemyOneIsPacman:
      self.particleFilter1.clearGhostParticles()
    else:
      self.particleFilter1.clearPacmanParticles()
    if oldEnemyTwoIsPacman:
      self.particleFilter2.clearGhostParticles()
    else:
      self.particleFilter2.clearPacmanParticles()
    
    noisyReadings = [observed for observed in gameState.getAgentDistances()]
    self.particleFilter1.observe(noisyReadings[self.particleFilter1.getTargetIndex()], gameState, gameState.getAgentPosition(self.index))
    self.particleFilter2.observe(noisyReadings[self.particleFilter2.getTargetIndex()], gameState, gameState.getAgentPosition(self.index))
  
    if self.showNoise and self.red:
      pf1 = copy.deepcopy(self.particleFilter1)
      pf2 = copy.deepcopy(self.particleFilter2)
      pf1.elapseTime()
      pf2.elapseTime()
      multipleConstant = 1
      approxLocs1 = pf1.getApproximateLocations(50)
      approxLocs2 = pf2.getApproximateLocations(50)
      
      delList = list()
      for loc in approxLocs1:
        if loc in approxLocs2:
          if approxLocs1[loc] > approxLocs2[loc]:
            del approxLocs2[loc]
          else:
            delList.append(loc)
      
      for loc in delList:
        del approxLocs1[loc]
        
      self.debugDraw(self.spawnLocation, [0, 0, 0], True)
      
      for loc in approxLocs1:
        self.debugDraw(loc, [0, 0, self.clamp(approxLocs1[loc] * multipleConstant, 0, 1)], False)

      for loc in approxLocs2:
        self.debugDraw(loc, [0, self.clamp(approxLocs2[loc] * multipleConstant, 0, 1), self.clamp(approxLocs2[loc] * multipleConstant, 0, 1)], False)
  
  def chooseAction(self, gameState: GameState) -> None:
    """
    Picks among the actions with the highest Q(s,a).
    """
    
    if self.debug and self.red: start = time.time()
    
    if self.previousActionMemory != None:
      self.updateParticleFilters(gameState)
    
      if self.debug and self.red: print("\n\n\n-----------------------------------Move %s-----------------------------------\noldLoc: %s, Action: %s, Loc: %s\n"% (self.movesTaken, self.previousActionMemory[0].getAgentPosition(self.index), self.previousActionMemory[1], gameState.getAgentPosition(self.index)))
    
    if (self.isTraining):
      #Temp saves weights into json file (change this later)
      self.updateWeights()
      #Temp saves graveyard into json file (change this later)
      self.updateGraveyard()
      
      #Update weights and TS
      if (self.previousActionMemory != None):
        #Update weights and all that
        self.update(self.previousActionMemory[0], self.previousActionMemory[1], gameState)
      
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
      elif random.random() < STARINGEPSILON ** self.gameCount:
        action = self.getRandomAction(gameState)
      else:
        action = self.getBestAction(gameState)
    else:
      action = self.getBestAction(gameState)
    self.previousActionMemory = (gameState, action)
    self.movesTaken += 1
    
    if self.debug and self.red: 
      timeTaken = time.time() - start
      self.averageEvalTime.append(timeTaken)
      if timeTaken > self.maxEvalTime:
        self.maxEvalTime = timeTaken
        
      avgEvalTime = sum(self.averageEvalTime)/len(self.averageEvalTime)
      print('\navg eval time: %s' % avgEvalTime)
      print('max eval time: %s' % self.maxEvalTime)
    
    return action
  
  def getRandomAction(self, state: GameState) -> str:
    legalActions = state.getLegalActions(self.index)
    
    if len(legalActions) == 0:
      return ""  # Terminal state, no legal actions
    
    if len(legalActions) == 1:
      return 'Stop'
    
    action = None
    
    while action != Directions.STOP:
      action = random.choice(legalActions)
    
    return action
  
  def getBestAction(self, state: GameState) -> str:
    """
      Compute the best action to take in a state.
    """

    legalActions = state.getLegalActions(self.index)

    if len(legalActions) == 0:
      return ""  # Terminal state, no legal actions
    
    if len(legalActions) == 1:
      return 'Stop'

    bestActions = []
    bestQValue = float('-inf')

    for action in legalActions:
      if action != Directions.STOP:
        qValue = self.getQValue(state, action)
        if qValue > bestQValue:
          bestActions = [action]
          bestQValue = qValue
        elif qValue == bestQValue:
          bestActions.append(action)
        elif bestQValue == float('-inf'):
          print("Error")
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

    maxQValue = max(self.getQValue(state, action) + self.optimisticConstant/(self.regularGraveyard["(%s, %s)" % (currentArmBranch, action)] + 1) for action in legalActions)
    return maxQValue
  
  def update(self, state: GameState, action: str, nextState: GameState):
    """
      Updates the weights, Also updates the thompson samples
    """
    rewards = self.getReward(state, action, nextState)    
    featureDict = self.getFeatures(state, action)
    maxFutureActionQ_sa = self.getMaxQ_SA(nextState)
    q_sa = self.getQValue(state, action)
    
    reward = rewards[0] + rewards[1]
    
    featureDifference = util.Counter()
    
    debugReward = util.Counter()
    
    for feature in self.featuresList:
      if feature in self.postiveFeatures:
        featureDifference[feature] = (rewards[0] + (self.discount * maxFutureActionQ_sa)) - q_sa
        debugReward[feature] = rewards[0]
      elif feature in self.negativeFeatures:
        featureDifference[feature] = (rewards[1] + (self.discount * maxFutureActionQ_sa)) - q_sa
        debugReward[feature] = rewards[1]
        
    if self.red and self.debug: print("\nUpdating Features")
    
    for feature in featureDict.keys():
      if self.red and self.debug and featureDict[feature] > 0: 
        print("\n%s: %s, %s Change after update: %s\n              = %s * %s * %s\n                            featureDiff = (%s + (%s * %s)) - %s" % (feature, featureDict[feature], feature, str(self.learningRate * featureDifference[feature] * featureDict[feature]), self.learningRate, featureDifference[feature], featureDict[feature], debugReward[feature], self.discount, maxFutureActionQ_sa, q_sa))
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
    
  def getReward(self, state: GameState, action : str, nextState: GameState) -> tuple[int]:
    """
      Returns reward when given a state, action, and nextState. Should only be called when training
    """
    if not self.isTraining:
      print("WTF error why did u call getReward when ur not training. Returning None")
      return None
    
    #For Reward Splitting
    positiveReward = self.positiveRewards(state, action, nextState)
    negativeReward = self.negativeRewards(state, action, nextState)
    
    return (positiveReward, negativeReward)

  def getSuccessor(self, gameState: GameState, action: str) -> tuple[GameState, any, any]:
    """
    Finds the next successor which is a grid position (location tuple).
    """
    pf1 = copy.deepcopy(self.particleFilter1)
    pf2 = copy.deepcopy(self.particleFilter2)
    pf1.elapseTime()
    pf2.elapseTime()
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != util.nearestPoint(pos):
        # Only half a grid position was covered
        return (successor.generateSuccessor(self.index, action), pf1, pf2)
    else:
        return (successor, pf1, pf2)

  def getQValue(self, gameState: GameState, action: str) -> float:
    """
    Computes a linear combination of features and feature weights
    """
    featureVector = self.getFeatures(gameState, action)
    return featureVector * self.weights

  def final(self, gameState: GameState) -> None:
    self.gameCount += 1
    if (self.isTraining):
      #Update weights and TS
      if (self.previousActionMemory != None):
        #Update weights and all that
        self.update(self.previousActionMemory[0], self.previousActionMemory[1], gameState)
      
      self.movesTaken = 0
      
      #Temp saves weights into json file (change this later)
      self.updateWeights()
      #Temp saves graveyard into json file (change this later)
      self.updateGraveyard()

  def getFeatures(self, gameState: GameState, action: str) -> util.Counter:
    """
    Returns a counter of features for the state
    """
    
    features = util.Counter()
    agentState = gameState.getAgentState(self.index)
    nextStateTuple = self.getSuccessor(gameState, action)
    nextState = nextStateTuple[0]
    pf1 = nextStateTuple[1]
    pf2 = nextStateTuple[2]
    nextAgentState = nextState.getAgentState(self.index)
    pos = agentState.getPosition()
    nextPos = nextAgentState.getPosition()
    
    if (pos == self.spawnLocation and self.movesTaken > 20) or (nextPos == self.spawnLocation and ((self.red and pos[0] > 15) or (not self.red and pos[0] < 16))):
      features['died'] = 1
    else:
      features['died'] = 0
    
    foodList = self.getFood(gameState).asList()
    
    features['collectedFood'] = 1 if nextPos in foodList else 0
    
    oldCapsulesList = self.getCapsules(gameState)
    newCapsulesList = self.getCapsules(nextState)
    
    foodDistanceScore = -100
    # Will get the closest pellet and weigh score by how close pellet is and whether it got closer or not
    
    opponents = self.getOpponents(gameState)
    enemyLocs = {}
    for enemyIndex in opponents:
      if enemyIndex != None:
        if pf1.getTargetIndex() == enemyIndex:
          enemyLocs[enemyIndex] = [(loc, prob) for loc, prob in pf1.getGhostApproximateLocations().items()]
        elif pf2.getTargetIndex() == enemyIndex:
          enemyLocs[enemyIndex] = [(loc, prob) for loc, prob in pf2.getGhostApproximateLocations().items()]
    
    distanceIncreasePercent = 0
    mostProbableDistance = None
    highestProb = 0
    for enemyIndex in enemyLocs:
      for locProbTuple in enemyLocs[enemyIndex]:
        oldEnemyDistance = self.distancer.getDistance(pos, locProbTuple[0])
        currentEnemyDistance = self.distancer.getDistance(nextPos, locProbTuple[0])
        if currentEnemyDistance < oldEnemyDistance:
          distanceIncreasePercent += locProbTuple[1] /2
        
        if locProbTuple[1] > highestProb:
          highestProb = locProbTuple[1]
          mostProbableDistance = self.distancer.getDistance(pos, locProbTuple[0])
    
    features['closerToGhost'] = 0
    
    if highestProb > 0 and nextAgentState.isPacman:
      if features['died'] == 1:
        features['closerToGhost'] = 1
      elif mostProbableDistance <= 1:
        features['closerToGhost'] = 1
      else:
        features['closerToGhost'] = distanceIncreasePercent * (-1/3.5 * math.log(mostProbableDistance, 3) +1)
    
    if len(foodList) > 0:
      for food in foodList:
        oldDistanceToFood = self.distancer.getDistance(pos, food)
        newDistanceToFood = self.distancer.getDistance(nextPos, food)
        currentFoodDistanceScore = 0
        
        distanceDelta = oldDistanceToFood - newDistanceToFood
        if newDistanceToFood <= 1 and distanceDelta > 0:
          currentFoodDistanceScore = distanceDelta
        else:
          currentFoodDistanceScore = distanceDelta * (-1/6 * math.log2(newDistanceToFood) +1)
        
        if currentFoodDistanceScore > foodDistanceScore: foodDistanceScore = currentFoodDistanceScore
    
    features['closerToExtraction'] = 0
    
    if nextAgentState.numCarrying <= 3:
      if features['collectedFood'] == 1:
        features['closerToUnguardedFood'] = 1 
      elif foodDistanceScore > 0:
        features['closerToUnguardedFood'] = foodDistanceScore
    else:
      extractDistanceScore = 0
      for loc in self.extractLocs:
        oldDistanceToLoc = self.distancer.getDistance(pos, loc)
        newDistanceToLoc = self.distancer.getDistance(nextPos, loc)
        currentDistanceScore = 0
        
        distanceDelta = oldDistanceToLoc - newDistanceToLoc
        if newDistanceToLoc <= 1 and distanceDelta > 0:
          currentDistanceScore = distanceDelta
        else:
          currentDistanceScore = distanceDelta * (-1/3.5 * math.log(newDistanceToLoc, 3) +1)
        
        if currentDistanceScore > extractDistanceScore: extractDistanceScore = currentDistanceScore
      
      features['closerToExtraction'] = extractDistanceScore if extractDistanceScore > 0 else 0
    
    features['collectedCapsule'] = 1 if len(newCapsulesList) > len(oldCapsulesList) else 0
    
    if len(oldCapsulesList) > 0:
      capsule = oldCapsulesList[0]
      oldDistanceToCapsule = self.distancer.getDistance(pos, capsule)
      newDistanceToCapsule = self.distancer.getDistance(nextPos, capsule)
      currentFoodDistanceScore = 0
      
      distanceDelta = oldDistanceToCapsule - newDistanceToCapsule
      
    features['closerToCapsule'] = 1 if features['collectedCapsule'] == 1 else distanceDelta
    
    features['gotScore'] = nextAgentState.numCarrying/len(foodList) if nextState.getScore() > gameState.getScore() else 0
    
    return features
  
  def negativeRewards(self, state: GameState, action: str, nextState: GameState) -> float:
    isDead = 0
    oldPos = state.getAgentPosition(self.index)
    newPos = nextState.getAgentPosition(self.index)
    
    if self.red:
      isDead = -2 if newPos == self.spawnLocation and oldPos[0] > 15 else 0
    if not self.red:
      isDead = -2 if newPos == self.spawnLocation and oldPos[0] < 16 else 0
    
    closerToGhost = 0
    
    enemyLocs = {}
    opponents = self.getOpponents(nextState)
    for enemyIndex in opponents:
      if enemyIndex != None:
        if self.particleFilter1.getTargetIndex() == enemyIndex:
          enemyLocs[enemyIndex] = [(loc, prob) for loc, prob in self.particleFilter1.getGhostApproximateLocations().items()]
        elif self.particleFilter2.getTargetIndex() == enemyIndex:
          enemyLocs[enemyIndex] = [(loc, prob) for loc, prob in self.particleFilter2.getGhostApproximateLocations().items()]
    
    distanceIncreasePercent = 0
    mostProbableDistance = None
    highestProb = 0
    for enemyIndex in enemyLocs:
      for locProbTuple in enemyLocs[enemyIndex]:
        oldEnemyDistance = self.distancer.getDistance(oldPos, locProbTuple[0])
        currentEnemyDistance = self.distancer.getDistance(newPos, locProbTuple[0])
        if currentEnemyDistance < oldEnemyDistance:
          distanceIncreasePercent += locProbTuple[1] /2
        
        if locProbTuple[1] > highestProb:
          highestProb = locProbTuple[1]
          mostProbableDistance = self.distancer.getDistance(newPos, locProbTuple[0])
    
    if highestProb > 0 and nextState.getAgentState(self.index).isPacman:
      if isDead == -2:
        closerToGhost = -2
      elif mostProbableDistance <= 1:
        closerToGhost = -2
      else:
        closerToGhost = distanceIncreasePercent * -0.25 * (-1/3.5 * math.log(mostProbableDistance, 3) +1)
    
    negativeReward = isDead + closerToGhost
    if self.debug and self.red: print("negativeReward = isDead + closerToGhost\n    %s = %s + %s" % (negativeReward, isDead, closerToGhost))
    
    return negativeReward
    
  def positiveRewards(self, state: GameState, action: str, nextState: GameState) -> float:
    oldPos = state.getAgentPosition(self.index)
    newPos = nextState.getAgentPosition(self.index)
    
    oldFoodList = self.getFood(state).asList()
    foodList = self.getFood(nextState).asList()
    
    pelletReward = 0.5 if len(foodList) > len(oldFoodList) and newPos in oldFoodList else 0
    
    foodList = self.getFood(nextState).asList()
    
    foodDistanceScore = 0
    
    if len(foodList) > 0:
      for food in foodList:
        oldDistanceToFood = self.distancer.getDistance(oldPos, food)
        newDistanceToFood = self.distancer.getDistance(newPos, food)
        currentFoodDistanceScore = 0
        
        distanceDelta = oldDistanceToFood - newDistanceToFood
        if newDistanceToFood <= 1 and distanceDelta > 0:
          currentFoodDistanceScore = distanceDelta
        else:
          currentFoodDistanceScore = distanceDelta * (-1/6 * math.log2(newDistanceToFood) +1)
        
        if currentFoodDistanceScore > foodDistanceScore: foodDistanceScore = currentFoodDistanceScore
        
    pelletReward += 0.05 * foodDistanceScore if foodDistanceScore > 0 else 0
    
    oldCapsulesList = self.getCapsules(state)
    newCapsulesList = self.getCapsules(nextState)

    capsuleReward = 1 if len(newCapsulesList) > len(oldCapsulesList) else 0
    
    if len(oldCapsulesList) > 0:
      capsule = oldCapsulesList[0]
      oldDistanceToCapsule = self.distancer.getDistance(oldPos, capsule)
      newDistanceToCapsule = self.distancer.getDistance(newPos, capsule)
      
      distanceDelta = oldDistanceToCapsule - newDistanceToCapsule
      
    capsuleReward += distanceDelta * 0.1 if distanceDelta > 0 and len(oldCapsulesList) > 0 else 0
    
    closerToExtraction = 0
    needToExtract = False
    
    pelletsNeeded = 1 - nextState.getScore() if nextState.getScore() < 1 else 1
    
    if state.getAgentState(self.index).numCarrying >= pelletsNeeded:
      needToExtract = True
      extractDistanceScore = 0
      for loc in self.extractLocs:
        oldDistanceToLoc = self.distancer.getDistance(oldPos, loc)
        newDistanceToLoc = self.distancer.getDistance(newPos, loc)
        currentDistanceScore = 0
        
        capsuleDistanceDelta = oldDistanceToLoc - newDistanceToLoc
        if newDistanceToLoc <= 1 and capsuleDistanceDelta > 0:
          currentDistanceScore = capsuleDistanceDelta
        else:
          currentDistanceScore = capsuleDistanceDelta * (-1/3.5 * math.log(newDistanceToLoc, 3) +1)
        
        if currentDistanceScore > extractDistanceScore: extractDistanceScore = currentDistanceScore
        
      closerToExtraction = extractDistanceScore * 0.25 if extractDistanceScore > 0 else 0
    
    scoreReward = nextState.getScore() - state.getScore() if nextState.getScore() > state.getScore() and nextState.getAgentState(self.index).numReturned > 0 else 0
    
    if needToExtract:
      positiveReward = closerToExtraction + scoreReward + capsuleReward
      if self.debug and self.red: print("Positive Reward = closerToExtraction + scoreReward + capsuleReward\n    %s = %s + %s + %s" %(positiveReward, closerToExtraction, scoreReward, capsuleReward))
    else:
      positiveReward = pelletReward + capsuleReward + scoreReward
      if self.debug and self.red: print("Positive Reward = pelletReward + capsuleReward + scoreReward\n    %s = %s + %s + %s" %(positiveReward, pelletReward, capsuleReward, scoreReward))
    
    return positiveReward - self.livingReward if positiveReward - self.livingReward > 0 else 0
  
class SecondAgent(FirstAgent):
  #PACMAN CHASER PROFESSIONAL
  def getOrSetDebug(self) -> str:
    if (self.red): self.debug = False
    return "SecondAgent"
  
  def getFeatureList(self) -> list[str]:
    return ['gotOffDefense', 'died', 'closerToPacman', 'ateGhost', 'closerToCenter']
  def getPositiveFeatures(self) -> list[str]:
    return ['closerToPacman', 'ateGhost', 'closerToCenter']
  def getNegativeFeatures(self) -> list[str]:
    return ['gotOffDefense', 'died']
  
  def getFeatures(self, gameState: GameState, action: str) -> util.Counter:
    features = util.Counter()
        
    oldPos = gameState.getAgentState(self.index).getPosition()
    oldState = gameState.getAgentState(self.index)
    
    nextStateTuple = self.getSuccessor(gameState, action)
    nextState = nextStateTuple[0]
    pf1 = nextStateTuple[1]
    pf2 = nextStateTuple[2]
    myState = nextState.getAgentState(self.index)
    myPos = myState.getPosition()
      
    features['gotOffDefense'] = 1 if myState.isPacman and not oldState.isPacman else 0
    
    features['died'] = 1 if myPos == self.spawnLocation else 0
    
    enemyOneLocs = [(loc, prob) for loc, prob in pf1.getPacmanApproximateLocations(count=10).items()]
    enemyTwoLocs = [(loc, prob) for loc, prob in pf2.getPacmanApproximateLocations(count=10).items()]
    
    enemyOneAvgDist = sum([prob * self.distancer.getDistance(myPos, loc) 
                                      if (self.red and loc[0] <= 15) or (not self.red and loc[0] >= 16) else 0 
                                      for loc, prob in enemyOneLocs])
    enemyTwoAvgDist = sum([prob * self.distancer.getDistance(myPos, loc) 
                                      if (self.red and loc[0] <= 15) or (not self.red and loc[0] >= 16) else 0 
                                      for loc, prob in enemyTwoLocs])
    
    if enemyOneAvgDist < enemyTwoAvgDist or enemyTwoAvgDist == 0:
      # oldEnemyOneAvgDist = sum([prob * self.distancer.getDistance(oldPos, loc) 
      #                                 if (self.red and loc[0] <= 15) or (not self.red and loc[0] >= 16) else 0 
      #                                 for loc, prob in enemyOneLocs])
      distanceIncreasePercent = 0
      for locProbTuple in enemyOneLocs:
        if locProbTuple != None:
          oldEnemyDistance = self.distancer.getDistance(oldPos, locProbTuple[0])
          currentEnemyDistance = self.distancer.getDistance(myPos, locProbTuple[0])
          
          if currentEnemyDistance < oldEnemyDistance:
            distanceIncreasePercent += locProbTuple[1]
      features['closerToPacman'] = distanceIncreasePercent
    elif enemyTwoAvgDist < enemyOneAvgDist or enemyOneAvgDist == 0:
      # oldEnemyTwoAvgDist = sum([prob * self.distancer.getDistance(oldPos, loc) 
      #                                 if (self.red and loc[0] <= 15) or (not self.red and loc[0] >= 16) else 0 
      #                                 for loc, prob in enemyTwoLocs])
      distanceIncreasePercent = 0
      for locProbTuple in enemyTwoLocs:
        if locProbTuple != None:
          oldEnemyDistance = self.distancer.getDistance(oldPos, locProbTuple[0])
          currentEnemyDistance = self.distancer.getDistance(myPos, locProbTuple[0])
          
          if currentEnemyDistance < oldEnemyDistance:
            distanceIncreasePercent += locProbTuple[1]
      features['closerToPacman'] = distanceIncreasePercent

    # print("Agent Position %s: %s\nAgent Position %s: %s" % (self.enemyIndices[0], gameState.getAgentPosition(self.enemyIndices[0]), self.enemyIndices[0], gameState.getAgentPosition(self.enemyIndices[0])))
    
    if (gameState.getAgentPosition(self.enemyIndices[0]) == myPos and ((self.red and myPos[0] <= 15) or (not self.red and myPos[0] >= 16))) or (gameState.getAgentPosition(self.enemyIndices[1]) == myPos and ((self.red and myPos[0] <= 15) or (not self.red and myPos[0] >= 16))):
      features['ateGhost'] = 1

    center = (15, 7) if self.red else (16, 7)
    oldCenterDistance = self.distancer.getDistance(oldPos, center)
    newCenterDistance = self.distancer.getDistance(myPos, center)
    
    features['closerToCenter'] = 0.01 if newCenterDistance < oldCenterDistance else 0

    return features
  
  def negativeRewards(self, state: GameState, action: str, nextState: GameState) -> float:
    isDead = 0
    oldPos = state.getAgentPosition(self.index)
    newPos = nextState.getAgentPosition(self.index)
    
    if self.red:
      isDead = -2 if newPos == self.spawnLocation and oldPos[0] > 15 else 0
    if not self.red:
      isDead = -2 if newPos == self.spawnLocation and oldPos[0] < 16 else 0
    
    gotOffDefense = -1 if state.getAgentState(self.index).isPacman and not nextState.getAgentState(self.index).isPacman else 0
    
    negativeReward = isDead + gotOffDefense
    if self.debug and self.red: print("negativeReward = isDead + gotOffDefense\n    %s = %s + %s" % (negativeReward, isDead, gotOffDefense))
    
    return negativeReward
  
  def positiveRewards(self, state: GameState, action: str, nextState: GameState) -> float:
    closerToGhost = 0
    
    oldPos = state.getAgentPosition(self.index)
    newPos = nextState.getAgentPosition(self.index)
    
    if nextState.getAgentPosition(self.index) == state.getAgentPosition(self.enemyIndices[0]) or state.getAgentPosition(self.index) == nextState.getAgentPosition(self.enemyIndices[1]):
      closerToGhost = 1
    else:
      closerToGhost = 0
    
    enemyLocs = {}
    opponents = self.getOpponents(nextState)
    for enemyIndex in opponents:
      if enemyIndex != None:
        if self.particleFilter1.getTargetIndex() == enemyIndex:
          enemyLocs[enemyIndex] = [(loc, prob) for loc, prob in self.particleFilter1.getPacmanApproximateLocations().items()]
        elif self.particleFilter2.getTargetIndex() == enemyIndex:
          enemyLocs[enemyIndex] = [(loc, prob) for loc, prob in self.particleFilter2.getPacmanApproximateLocations().items()]
    
    distanceIncreasePercent = 0
    mostProbableDistance = None
    highestProb = 0
    for enemyIndex in enemyLocs:
      for locProbTuple in enemyLocs[enemyIndex]:
        oldEnemyDistance = self.distancer.getDistance(oldPos, locProbTuple[0])
        currentEnemyDistance = self.distancer.getDistance(newPos, locProbTuple[0])
        if currentEnemyDistance < oldEnemyDistance:
          distanceIncreasePercent += locProbTuple[1] /2
        
        if locProbTuple[1] > highestProb:
          highestProb = locProbTuple[1]
          mostProbableDistance = self.distancer.getDistance(newPos, locProbTuple[0])
    
    if highestProb > 0 and nextState.getAgentState(self.index).isPacman:
      if mostProbableDistance <= 1:
        closerToGhost += 1
      else:
        closerToGhost += distanceIncreasePercent * 0.25 * (-1/3.5 * math.log(mostProbableDistance, 3) +1)        
    
    positiveReward = closerToGhost
    
    return positiveReward - self.livingReward if positiveReward - self.livingReward > 0 else 0
  
  def gotOffDefenseReward(self, state: GameState, action : str, nextState: GameState) -> float:
    return -5
  def diedReward(self, state: GameState, action : str, nextState: GameState) -> float:
    return -10
  def closerToPacmanReward(self, state: GameState, action : str, nextState: GameState) -> float:
    return 
  def closerToCenterReward(self, state: GameState, action : str, nextState: GameState) -> float:
    return 
  def ateGhostReward(self, state: GameState, action : str, nextState: GameState) -> float:
    return 10
  def stopReward(self, state: GameState, action : str, nextState: GameState) -> float:
    return -1

#Helper Classes
class ParticleFilter():
  """
  A particle filter for approximately tracking a single ghost.
  """

  def __init__(self, isRed: bool, index: int, particleMult: int = 1) -> None:
    #When ghost eats pellet on your side you know their location
    self.particles = None
    self.targetIndex = index
    self.red = isRed
    self.particleMult = particleMult
    
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
    
    self.allParticlesToLoc(self.spawnLoc)
    
    self.distancer = distanceCalculator.Distancer(gameState.data.layout)
      
  def getPositionDistribution(self, enemyLocation: tuple) -> util.Counter:
    """
    Returns a equally distributed distribution over successor positions of a given location.
    """
    newLocations = Actions.getLegalNeighbors(enemyLocation, self.walls)
    dist = util.Counter()
    possibleNumberOfMoves = len(newLocations) -1
    moveProb = 99
    if possibleNumberOfMoves > 0:
      moveProb = 100/possibleNumberOfMoves
    for location in newLocations:
      if location != enemyLocation: dist[location] = moveProb
    return dist
  
  def allParticlesToLoc(self, loc: tuple) -> None:
    self.particles = []
    
    for x in range(len(self.legalPositions) * self.particleMult):
      self.particles.append(loc)
      
  def halfParticlesToLoc(self, loc: tuple) -> None:
    self.newParticles = []
    
    for x in range(len(self.legalPositions) * self.particleMult):
      self.newParticles.append(loc)
    
    self.particles += self.newParticles
    self.resample()
  
  def initializeUniformly(self) -> None:
    """
    Initializes a list of particles.
    """
    
    self.particles = []
    for x in range(self.particleMult):
      for cord in self.legalPositions:
        self.particles.append(cord)
  
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
    weights = util.Counter()
    for particle in self.particles:
      weights[particle] += gameState.getDistanceProb(util.manhattanDistance(myPos, particle), noisyDistance)
    
    if weights.totalCount() == 0:
      self.initializeUniformly()
      weights = util.Counter()
      for particle in self.particles:
        weights[particle] += gameState.getDistanceProb(util.manhattanDistance(myPos, particle), noisyDistance)
      
    #Resample
    newParticles = [0] * len(self.legalPositions) * self.particleMult
    for x in range(len(self.legalPositions) * self.particleMult):
      newParticles[x] = util.sample(weights)
      
    self.particles = newParticles
  
  def elapseTime(self) -> None:
    """
    Update beliefs for a time step elapsing.
    """
    newParticles = list[tuple[int]]()
    for particle in self.particles: 
      particleDist = Actions.getLegalNeighbors(particle, self.walls) 
      particleDist.remove(particle)
      newParticles.append(random.choice(particleDist))

    self.particles = newParticles

  def normalize(self, vectorOrCounter):
    """
    normalize a vector or counter by dividing each value by the sum of all values
    """
    normalizedCounter = util.Counter()
    if type(vectorOrCounter) == type(normalizedCounter):
        counter = vectorOrCounter
        total = float(counter.totalCount())
        if total == 0: return counter
        for key in counter.keys():
            value = counter[key]
            normalizedCounter[key] = value / total
        return normalizedCounter
    else:
        vector = vectorOrCounter
        s = float(sum(vector))
        if s == 0: return vector
        return [el / s for el in vector]
  
  def killedAgent(self) -> None:
    """
    Update beliefs for when player eats the enemy
    """
    self.particles = []
    self.particles.append(self.spawnLoc)
    
    self.resample()
  
  def foundAgent(self, foundLocation: tuple[int]) -> None:
    """
    Update beliefs for when player finds the enemy location
    """
    self.particles = []
    self.particles.append(foundLocation)
    
    self.resample()
    
  def noAgentNear(self, playerLocation: tuple[int]) -> None:
    """
    Update beliefs for when player's personal detection does not detect enemy
    """
    nearbyTiles = [(playerLocation[0] + a - 5, playerLocation[1] + b - 5)
                   for a in range(11) for b in range (11)
                   if abs(a - 5) + abs(b - 5) <= 5]
    
    particlesToBeRemoved = []
    
    for particle in self.particles:
      if particle in nearbyTiles and particle not in particlesToBeRemoved: 
        particlesToBeRemoved.append(particle)
        
    newParticles = [particle for particle in self.particles if particle not in particlesToBeRemoved]
        
    if len(newParticles) != 0:
      self.particles = newParticles
      self.resample()

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
  
  def getApproximateLocations(self, count: int = 10) -> util.Counter():
    """
    Return the approximated locations in dict{location: probability}. Put an int into count to change how much locations you want (default is 10)
    """
    locationParticles = util.Counter()
    for particle in self.particles:
      locationParticles[particle] += 1
    
    locationNodes = [HeapNode(particleAmount, particleLoc) for particleLoc, particleAmount in locationParticles.items()]
          
    largestNodes = heapq.nlargest(count, locationNodes)
    topLocs = util.Counter()
    for node in largestNodes:
      if node.prob > 0: topLocs[node.loc] = node.prob
    
    topLocs.normalize()
    
    return topLocs
  
  def getPacmanApproximateLocations(self, count: int = 10) -> util.Counter():
    """
    Return the approximated locations when the enemy is in Pacman State in dict{location: probability}. Put an int into count to change how much locations you want (default is 10)
    """
    locationParticles = util.Counter()
    for particle in self.particles:
      locationParticles[particle] += 1
    
    locationNodes = [HeapNode(particleAmount, particleLoc) for particleLoc, particleAmount in locationParticles.items()
                     if (not self.red and particleLoc[0] >= 16) or (self.red and particleLoc[0] <= 15)]
          
    largestNodes = heapq.nlargest(count, locationNodes)
    topLocs = util.Counter()
    for node in largestNodes:
      if node.prob > 0: topLocs[node.loc] = node.prob
      
    topLocs.normalize()
      
    return topLocs  
  
  def getGhostApproximateLocations(self, count: int = 50) -> util.Counter():
    """
    Return the approximated locations when the enemy is in Ghost State in dict{location: probability}. Put an int into count to change how much locations you want (default is 10)
    """
    locationParticles = util.Counter()
    for particle in self.particles:
      locationParticles[particle] += 1
    
    locationNodes = [HeapNode(particleAmount, particleLoc) for particleLoc, particleAmount in locationParticles.items()
                     if (self.red and particleLoc[0] >= 16) or (not self.red and particleLoc[0] <= 15)]
          
    largestNodes = heapq.nlargest(count, locationNodes)
    topLocs = util.Counter()
    for node in largestNodes:
      if node.prob > 0: topLocs[node.loc] = node.prob
      
    topLocs.normalize()
      
    return topLocs  
  
  def clearGhostParticles(self) -> None:
    particlesToBeRemoved = []
    if self.red:
      for particle in self.particles:
        if particle[0] >= 16 and particle not in particlesToBeRemoved:
          particlesToBeRemoved.append(particle)
    elif not self.red:
      for particle in self.particles:
        if particle[0] <= 15 and particle not in particlesToBeRemoved:
          particlesToBeRemoved.append(particle)
        
    newParticles = [particle for particle in self.particles if particle not in particlesToBeRemoved]
    
    if len(newParticles) != 0:
      self.particles = newParticles
      self.resample()
        
  def clearPacmanParticles(self) -> None:
    particlesToBeRemoved = []
    if self.red:
      for particle in self.particles:
        if particle[0] <= 15 and particle not in particlesToBeRemoved:
          particlesToBeRemoved.append(particle)
    elif not self.red:
      for particle in self.particles:
        if particle[0] >= 16 and particle not in particlesToBeRemoved:
          particlesToBeRemoved.append(particle)
          
    newParticles = [particle for particle in self.particles if particle not in particlesToBeRemoved]
        
    if len(newParticles) != 0:
      self.particles = newParticles
      self.resample()
  
  def resample(self) -> None:
    weights = util.Counter()
    for particle in self.particles:
      weights[particle] += 1
    
    newParticles = [0] * len(self.legalPositions) * self.particleMult
    for x in range(len(self.legalPositions) * self.particleMult):
      newParticles[x] = util.sample(weights)
      
    self.particles = newParticles
  
@dataclass(order=True)
class HeapNode:
  prob: float
  loc: Any=field(compare=False)
  
