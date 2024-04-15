from myTeam import FirstAgent
from myTeam import SecondAgent
import util

def createTeam(firstIndex, secondIndex, isRed,
               first = 'FirstAgentDupe', second = 'SecondAgentDupe'):
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

class FirstAgentDupe(FirstAgent):
  def getOrSetDebug(self) -> str:
    self.debug = False
    return "FirstAgentDupe"

class SecondAgentDupe(SecondAgent):
  def getOrSetDebug(self) -> str:
    self.debug = False
    return "SecondAgentDupe"