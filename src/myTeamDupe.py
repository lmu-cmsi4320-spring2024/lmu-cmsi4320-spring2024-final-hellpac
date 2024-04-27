from myTeam import FirstAgent
from myTeam import SecondAgent
from myTeam import ParticleFilter
from capture import GameState
import util


WEIGHT_PATH = 'zdupeweights_MY_TEAM.json'
INITALTEAMWEIGHTS =  {"FirstAgentDupe": util.Counter(), "SecondAgentDupe": util.Counter()}

def createTeam(firstIndex, secondIndex, isRed,
               first = 'SecondAgentDupe', second = 'FirstAgentDupe'):
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
  print(isRed)
  print(firstIndex)
  print(secondIndex)
  if firstIndex == 1 or firstIndex == 3:
    enemyIndex = (0, 2)
  else:
    enemyIndex = (1, 3)
  
  particleFilter1 = ParticleFilter(isRed, enemyIndex[0])
  particleFilter2 = ParticleFilter(isRed, enemyIndex[1])
  return [eval(first)(firstIndex, particleFilter1, particleFilter2, True), eval(second)(secondIndex, particleFilter1, particleFilter2, False)]

class FirstAgentDupe(FirstAgent):
  def getOrSetDebug(self) -> str:
    self.debug = False
    self.showNoise = False
    return "FirstAgentDupe"
  
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

  def getFeatures(self, gameState: GameState, action: str) -> util.Counter:
    """
    Returns a counter of features for the state
    """
    start = time.time()
    
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
    
    foodDistanceScore = -100
    # Will get the closest pellet and weigh score by how close pellet is and whether it got closer or not
    
    opponents = self.getOpponents(gameState)
    enemyLocs = {}
    for enemyIndex in opponents:
      if enemyIndex != None:
        if pf1.getTargetIndex() == enemyIndex:
          enemyLocs[enemyIndex] = [(loc, prob) for loc, prob in pf1.getGhostApproximateLocations(10).items()]
        elif pf2.getTargetIndex() == enemyIndex:
          enemyLocs[enemyIndex] = [(loc, prob) for loc, prob in pf2.getGhostApproximateLocations(10).items()]
    
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
    
    if highestProb > 0:
      if features['died'] == 1:
        features['closerToGhost'] = 1
      elif mostProbableDistance <= 1:
        features['closerToGhost'] = 1
      else:
        if distanceIncreasePercent > 1:
          distanceIncreasePercent = 1
        features['closerToGhost'] = distanceIncreasePercent * ((-1/3.5 * math.log(mostProbableDistance, 3)) +1)
    
    if (nextState.getAgentPosition(self.index) == gameState.getAgentPosition(self.enemyIndices[0]) or nextState.getAgentPosition(self.index) == gameState.getAgentPosition(self.enemyIndices[1])):
      features['closerToGhost'] = 1
      
    if (gameState.getAgentState(self.enemyIndices[0]).scaredTimer > 0 and gameState.getAgentState(self.enemyIndices[1]).scaredTimer > 0):
      features['closerToGhost'] = 0
    
    if len(foodList) > 0:
      for food in foodList:
        oldDistanceToFood = self.distancer.getDistance(pos, food)
        newDistanceToFood = self.distancer.getDistance(nextPos, food)
        currentFoodDistanceScore = 0
        
        distanceDelta = oldDistanceToFood - newDistanceToFood
        if newDistanceToFood <= 1:
          currentFoodDistanceScore = distanceDelta if distanceDelta > 0 else 0
        else:
          currentFoodDistanceScore = distanceDelta * (-1/6 * math.log2(newDistanceToFood) +1)
        
        if currentFoodDistanceScore > foodDistanceScore: foodDistanceScore = currentFoodDistanceScore
    
    pelletsNeeded = 1 - gameState.getScore() if gameState.getScore() < 1 else 1
    
    if agentState.numCarrying < pelletsNeeded:
      if foodDistanceScore > 0:
        features['closerToUnguardedFood'] = foodDistanceScore
    else:
      extractDistanceScore = 0
      for loc in self.extractLocs:
        oldDistanceToLoc = self.distancer.getDistance(pos, loc)
        newDistanceToLoc = self.distancer.getDistance(nextPos, loc)
        currentDistanceScore = 0
        
        distanceDelta = oldDistanceToLoc - newDistanceToLoc
        if newDistanceToLoc <= 1:
          currentDistanceScore = distanceDelta if distanceDelta > 0 else 0
        else:
          currentDistanceScore = distanceDelta * (-1/3.5 * math.log(newDistanceToLoc, 3) +1)
        
        if currentDistanceScore > extractDistanceScore: extractDistanceScore = currentDistanceScore
      
      features['closerToExtraction'] = extractDistanceScore if extractDistanceScore > 0 else 0
      
      if nextPos in self.extractLocs:
        features['closerToExtraction'] = 1
    
    if features['collectedFood'] == 1:
        features['closerToUnguardedFood'] = 0
    
    if len(oldCapsulesList) > 0:
      features['collectedCapsule'] = 1 if nextPos == oldCapsulesList[0] else 0
    
    if len(oldCapsulesList) > 0:
      capsule = oldCapsulesList[0]
      oldDistanceToCapsule = self.distancer.getDistance(pos, capsule)
      newDistanceToCapsule = self.distancer.getDistance(nextPos, capsule)
      currentFoodDistanceScore = 0
      
      distanceDelta = oldDistanceToCapsule - newDistanceToCapsule
      
      features['closerToCapsule'] = 0 if features['collectedCapsule'] == 1 else distanceDelta
      if features['closerToCapsule'] < 0:
        features['closerToCapsule'] = 0

    features['gotScore'] = agentState.numCarrying/len(foodList) if not nextAgentState.isPacman and agentState.numCarrying > 0 else 0
    
    timeTaken = time.time() - start
    self.getFeatureTime.append(timeTaken)
    
    return features

class SecondAgentDupe(SecondAgent):
  def getOrSetDebug(self) -> str:
    self.debug = False
    self.showNoise = False
    return "SecondAgentDupe"
  
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