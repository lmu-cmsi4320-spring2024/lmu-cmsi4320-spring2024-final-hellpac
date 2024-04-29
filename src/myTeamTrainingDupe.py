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
    #First Training Session
    start = time.time()
    
    NUMBER_OF_GAMES = 50
    myAgents = capture.loadAgents(True, 'myTeamDupe', False, '')
    enemyAgents = capture.loadAgents(False, 'myTeamDummy', False, '')
    
    agents = [myAgents[0], enemyAgents[0], myAgents[1], enemyAgents[1]]
    
    layouts = []
    for i in range(NUMBER_OF_GAMES):
        l = layout.getLayout('defaultCapture')

        layouts.append(l)
    
    optionsArgs = {'display': textDisplay.NullGraphics(), 'redTeamName': 'ALLIED FORCES', 'blueTeamName': 'ENEMIES', 'agents': agents, 'layouts': layouts, 'length': 1200, 'numGames': NUMBER_OF_GAMES, 'numTraining': 0, 'record': False, 'catchExceptions': False}
    games = capture.runGames(**optionsArgs)
    
    #Second Training Session
    NUMBER_OF_GAMES = 50
    enemyAgents = capture.loadAgents(False, 'baselineTeam', False, '')
    
    agents = [myAgents[0], enemyAgents[0], myAgents[1], enemyAgents[1]]
    
    layouts = []
    for i in range(NUMBER_OF_GAMES):
        l = layout.getLayout('defaultCapture')

        layouts.append(l)
    
    optionsArgs = {'display': textDisplay.NullGraphics(), 'redTeamName': 'ALLIED FORCES', 'blueTeamName': 'ENEMIES', 'agents': agents, 'layouts': layouts, 'length': 1200, 'numGames': NUMBER_OF_GAMES, 'numTraining': 0, 'record': False, 'catchExceptions': False}
    games = capture.runGames(**optionsArgs)
    
    
    #Third Training Session
    NUMBER_OF_GAMES = 50
    enemyAgents = capture.loadAgents(False, 'betterBaselineTeam', False, '')
    
    agents = [myAgents[0], enemyAgents[0], myAgents[1], enemyAgents[1]]
    
    layouts = []
    for i in range(NUMBER_OF_GAMES):
        l = layout.getLayout('defaultCapture')

        layouts.append(l)
    
    optionsArgs = {'display': textDisplay.NullGraphics(), 'redTeamName': 'ALLIED FORCES', 'blueTeamName': 'ENEMIES', 'agents': agents, 'layouts': layouts, 'length': 1200, 'numGames': NUMBER_OF_GAMES, 'numTraining': 0, 'record': False, 'catchExceptions': False}
    games = capture.runGames(**optionsArgs)
    
    
    #Forth Training Session
    NUMBER_OF_GAMES = 50
    enemyAgents = capture.loadAgents(False, 'standoffTeam', False, '')
    
    agents = [myAgents[0], enemyAgents[0], myAgents[1], enemyAgents[1]]
    
    layouts = []
    for i in range(NUMBER_OF_GAMES):
        l = layout.getLayout('defaultCapture')

        layouts.append(l)
    
    optionsArgs = {'display': textDisplay.NullGraphics(), 'redTeamName': 'ALLIED FORCES', 'blueTeamName': 'ENEMIES', 'agents': agents, 'layouts': layouts, 'length': 1200, 'numGames': NUMBER_OF_GAMES, 'numTraining': 0, 'record': False, 'catchExceptions': False}
    games = capture.runGames(**optionsArgs)
    
    #Fifth Training Session
    NUMBER_OF_GAMES = 50
    enemyAgents = capture.loadAgents(False, 'betterBaselineTeam', False, '')
    
    agents = [myAgents[0], enemyAgents[0], myAgents[1], enemyAgents[1]]
    
    layouts = []
    for i in range(NUMBER_OF_GAMES):
        l = layout.getLayout('defaultCapture')

        layouts.append(l)
    
    optionsArgs = {'display': textDisplay.NullGraphics(), 'redTeamName': 'ALLIED FORCES', 'blueTeamName': 'ENEMIES', 'agents': agents, 'layouts': layouts, 'length': 1200, 'numGames': NUMBER_OF_GAMES, 'numTraining': 0, 'record': False, 'catchExceptions': False}
    games = capture.runGames(**optionsArgs)
    
    
    #Sixth Training Session
    NUMBER_OF_GAMES = 100
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