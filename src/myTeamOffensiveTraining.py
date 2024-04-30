import random, time, util, sys
import textDisplay
import capture
from capture import GameState
from capture import AgentRules
import layout

if __name__ == '__main__':
    """
    Training Loop For Professional Agents
    """
    start = time.time()
    myAgents = capture.loadAgents(True, 'myTeamOffensive', False, '')
    
    NUMBER_OF_GAMES = 150
    enemyAgents = capture.loadAgents(False, 'baselineTeam', False, '')
    
    agents = [myAgents[0], enemyAgents[0], myAgents[1], enemyAgents[1]]
    
    layouts = []
    for i in range(NUMBER_OF_GAMES):
        l = layout.getLayout('defaultCapture')

        layouts.append(l)
    
    optionsArgs = {'display': textDisplay.NullGraphics(), 'redTeamName': 'ALLIED FORCES', 'blueTeamName': 'ENEMIES', 'agents': agents, 'layouts': layouts, 'length': 1200, 'numGames': NUMBER_OF_GAMES, 'numTraining': 0, 'record': False, 'catchExceptions': False}
    games = capture.runGames(**optionsArgs)
    
    
    NUMBER_OF_GAMES = 150
    enemyAgents = capture.loadAgents(False, 'betterBaselineTeam', False, '')
    
    agents = [myAgents[0], enemyAgents[0], myAgents[1], enemyAgents[1]]
    
    layouts = []
    for i in range(NUMBER_OF_GAMES):
        l = layout.getLayout('defaultCapture')

        layouts.append(l)
    
    optionsArgs = {'display': textDisplay.NullGraphics(), 'redTeamName': 'ALLIED FORCES', 'blueTeamName': 'ENEMIES', 'agents': agents, 'layouts': layouts, 'length': 1200, 'numGames': NUMBER_OF_GAMES, 'numTraining': 0, 'record': False, 'catchExceptions': False}
    games = capture.runGames(**optionsArgs)
    
    
    NUMBER_OF_GAMES = 150
    enemyAgents = capture.loadAgents(False, 'standoffTeam', False, '')
    
    agents = [myAgents[0], enemyAgents[0], myAgents[1], enemyAgents[1]]
    
    layouts = []
    for i in range(NUMBER_OF_GAMES):
        l = layout.getLayout('defaultCapture')

        layouts.append(l)
    
    optionsArgs = {'display': textDisplay.NullGraphics(), 'redTeamName': 'ALLIED FORCES', 'blueTeamName': 'ENEMIES', 'agents': agents, 'layouts': layouts, 'length': 1200, 'numGames': NUMBER_OF_GAMES, 'numTraining': 0, 'record': False, 'catchExceptions': False}
    games = capture.runGames(**optionsArgs)
    
    timeTaken = time.time() - start
    
    print("\n\nTraining Loop Completed in %s seconds or %s minutes or %s hours" % (timeTaken, str(timeTaken/60), str(timeTaken/60/60)))