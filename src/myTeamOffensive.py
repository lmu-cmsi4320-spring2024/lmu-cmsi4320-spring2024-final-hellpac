from myTeam import FirstAgent
from myTeam import SecondAgent
from myTeam import ParticleFilter
from capture import GameState
import os
import util
import json
import time
import math


WEIGHT_PATH = 'zOffensiveWeights_MY_TEAM.json'
INITALTEAMWEIGHTS =  {"MainOffensiveAgent": util.Counter(), "SecondaryOffensiveAgent": util.Counter()}

def createTeam(firstIndex, secondIndex, isRed,
               first = 'MainOffensiveAgent', second = 'SecondaryOffensiveAgent'):
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
  if firstIndex == 1 or firstIndex == 3:
    enemyIndex = (0, 2)
  else:
    enemyIndex = (1, 3)
  
  particleFilter1 = ParticleFilter(isRed, enemyIndex[0])
  particleFilter2 = ParticleFilter(isRed, enemyIndex[1])
  return [eval(first)(firstIndex, particleFilter1, particleFilter2, True), eval(second)(secondIndex, particleFilter1, particleFilter2, False)]

class MainOffensiveAgent(FirstAgent):
  def getOrSetDebug(self) -> str:
    self.debug = False
    self.showNoise = False
    self.pelletNeed = 1
    return "MainOffensiveAgent"
  
  def registerCustomValues(self) -> None:
    self.foodView = [0, 1, 2, 3, 4, 5]
  
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

class SecondaryOffensiveAgent(MainOffensiveAgent):
  def getOrSetDebug(self) -> str:
    self.debug = False
    self.showNoise = False
    self.pelletNeed = 1
    return "SecondaryOffensiveAgent"
  
  def registerCustomValues(self) -> None:
    self.foodView = [11, 12, 13, 14]