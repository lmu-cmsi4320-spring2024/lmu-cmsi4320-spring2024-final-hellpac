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
    
    #Tests
    
    start = time.time()

    myAgents = capture.loadAgents(True, 'myTeam', False, '')

    NUMBER_OF_GAMES = 100
    enemyAgents = capture.loadAgents(False, 'myTeamOtherDupe', False, '')
    
    agents = [myAgents[0], enemyAgents[0], myAgents[1], enemyAgents[1]]
    
    layouts = []
    for i in range(NUMBER_OF_GAMES):
        l = layout.getLayout('defaultCapture')

        layouts.append(l)
    
    optionsArgs = {'display': textDisplay.NullGraphics(), 'redTeamName': 'ALLIED FORCES', 'blueTeamName': 'ENEMIES', 'agents': agents, 'layouts': layouts, 'length': 1200, 'numGames': NUMBER_OF_GAMES, 'numTraining': 0, 'record': False, 'catchExceptions': False}
    firstGame = capture.runGames(**optionsArgs)
    
    NUMBER_OF_GAMES = 100
    enemyAgents = capture.loadAgents(False, 'myTeamDupe', False, '')
    
    agents = [myAgents[0], enemyAgents[0], myAgents[1], enemyAgents[1]]
    
    layouts = []
    for i in range(NUMBER_OF_GAMES):
        l = layout.getLayout('defaultCapture')

        layouts.append(l)
    
    optionsArgs = {'display': textDisplay.NullGraphics(), 'redTeamName': 'ALLIED FORCES', 'blueTeamName': 'ENEMIES', 'agents': agents, 'layouts': layouts, 'length': 1200, 'numGames': NUMBER_OF_GAMES, 'numTraining': 0, 'record': False, 'catchExceptions': False}
    secondGame = capture.runGames(**optionsArgs)
    
    totalGames = firstGame + secondGame
    scores = [game.state.data.score for game in totalGames]
    redWinRate = [s > 0 for s in scores].count(True) / float(len(scores))
    blueWinRate = [s < 0 for s in scores].count(True) / float(len(scores))
    print('\n\n\nAverage Score:', sum(scores) / float(len(scores)))
    print('Scores:       ', ', '.join([str(score) for score in scores]))
    print('Red Win Rate:  %d/%d (%.2f)' % (
        [s > 0 for s in scores].count(True), len(scores), redWinRate))
    print('Blue Win Rate: %d/%d (%.2f)' % (
        [s < 0 for s in scores].count(True), len(scores), blueWinRate))
    print('Record:       ', ', '.join(
        [('Blue', 'Tie', 'Red')[max(0, min(2, 1 + s))] for s in scores]))
    
    timeTaken = time.time() - start
    
    print("\n\nTraining Loop Completed in %s seconds or %s minutes or %s hours" % (timeTaken, str(timeTaken/60), str(timeTaken/60/60)))