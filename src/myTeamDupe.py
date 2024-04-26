from myTeam import FirstAgent
from myTeam import SecondAgent
from myTeam import ParticleFilter
import util

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

class SecondAgentDupe(SecondAgent):
  def getOrSetDebug(self) -> str:
    self.debug = False
    self.showNoise = False
    return "SecondAgentDupe"