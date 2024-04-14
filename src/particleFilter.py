import itertools
import util

class ParticleFilter():
    """
    A particle filter for approximately tracking a single ghost.

    Useful helper functions will include random.choice, which chooses an element
    from a list uniformly at random, and util.sample, which samples a key from a
    Counter by treating its values as probabilities.
    """

    def __init__(self, gameState, legalPositions, numParticles=300):
        self.numParticles = numParticles
        self.beliefs = util.Counter()
        self.initalGameState = gameState
        self.legalPositions = legalPositions
        self.initializeUniformly()
        
        
    def getPositionDistribution(self, gameState):
        """
        Returns a distribution over successor positions of the ghost from the
        given gameState.

        You must first place the ghost in the gameState, using setGhostPosition
        below.
        """
        ghostPosition = gameState.getGhostPosition(self.index)
        actionDist = self.ghostAgent.getDistribution(gameState)
        dist = util.Counter()
        for action, prob in actionDist.items():
            successorPosition = game.Actions.getSuccessor(ghostPosition, action)
            dist[successorPosition] = prob
        return dist

    def setGhostPosition(self, gameState, ghostPosition):
        """
        Sets the position of the ghost for this inference module to the
        specified position in the supplied gameState.

        Note that calling setGhostPosition does not change the position of the
        ghost in the GameState object used for tracking the true progression of
        the game.  The code in inference.py only ever receives a deep copy of
        the GameState object which is responsible for maintaining game state,
        not a reference to the original object.  Note also that the ghost
        distance observations are stored at the time the GameState object is
        created, so changing the position of the ghost will not affect the
        functioning of observeState.
        """
        conf = game.Configuration(ghostPosition, game.Directions.STOP)
        gameState.data.agentStates[self.index] = game.AgentState(conf, False)
        return gameState

    def initializeUniformly(self):
        """
        Initializes a list of particles. Use self.numParticles for the number of
        particles. Use self.legalPositions for the legal board positions where a
        particle could be located.  Particles should be evenly (not randomly)
        distributed across positions in order to ensure a uniform prior.

        Note: the variable you store your particles in must be a list; a list is
        simply a collection of unweighted variables (positions in this case).
        Storing your particles as a Counter (where there could be an associated
        weight with each position) is incorrect and may produce errors.
        """
        uniformedParticles = [None] * self.numParticles
        for particleNum in range(0, self.numParticles):
            newParticle = self.legalPositions[particleNum % len(self.legalPositions)]
            uniformedParticles[particleNum] = newParticle
        
        self.beliefs = util.Counter()
        for particlePosition in uniformedParticles:
            self.beliefs[particlePosition] = self.beliefs[particlePosition] + 1 if self.beliefs[particlePosition] != None else 1
            
        self.beliefs.normalize()
        
        return uniformedParticles 
    
    def observe(self, observation, gameState):
        """
        Update beliefs based on the given distance observation. Make sure to
        handle the special case where all particles have weight 0 after
        reweighting based on observation. If this happens, resample particles
        uniformly at random from the set of legal positions
        (self.legalPositions).

        A correct implementation will handle two special cases:
          1) When a ghost is captured by Pacman, all particles should be updated
             so that the ghost appears in its prison cell,
             self.getJailPosition()

             As before, you can check if a ghost has been captured by Pacman by
             checking if it has a noisyDistance of None.

          2) When all particles receive 0 weight, they should be recreated from
             the prior distribution by calling initializeUniformly. The total
             weight for a belief distribution can be found by calling totalCount
             on a Counter object

        util.sample(Counter object) is a helper method to generate a sample from
        a belief distribution.

        You may also want to use util.manhattanDistance to calculate the
        distance between a particle and Pacman's position.
        """
        noisyDistance = observation
        emissionModel = busters.getObservationDistribution(noisyDistance)
        pacmanPosition = gameState.getPacmanPosition()
        legalPositions = [p for p in gameState.getWalls().asList(False) if p[1] > 1]
        
        if self.beliefs.totalCount() == 0:
            self.initializeUniformly(gameState)
            
        for position in self.beliefs.keys():
            # print("beliefDist[%s] = emissionsModel[%s] * beliefDist[%s]" % (position, util.manhattanDistance(pacmanPosition, position), position))
            # print("beliefDist[%s] = %s * %s" % (position, emissionModel[util.manhattanDistance(pacmanPosition, position)], self.beliefs[position]))
            self.beliefs[position] = emissionModel[util.manhattanDistance(pacmanPosition, position)] * self.beliefs[position]
            # print(str(position) + ": " + str(self.beliefs[position]) + "\n")
        
        if noisyDistance is None:
            for p in legalPositions:
                self.beliefs[p] = 0.0
            self.beliefs[self.getJailPosition()] = 1.0
        
        
        self.beliefs.normalize()
        return self.beliefs

    def elapseTime(self, gameState):
        """
        Update beliefs for a time step elapsing.

        As in the elapseTime method of ExactInference, you should use:

          newPosDist = self.getPositionDistribution(self.setGhostPosition(gameState, oldPos))

        to obtain the distribution over new positions for the ghost, given its
        previous position (oldPos) as well as Pacman's current position.

        util.sample(Counter object) is a helper method to generate a sample from
        a belief distribution.
        """
        legalPositions = [p for p in gameState.getWalls().asList(False) if p[1] > 1]
        
        bprime = util.Counter() #initialize bprime
        for oldPos in legalPositions: #summing over all old positions
            newPosDist = self.getPositionDistribution(self.setGhostPosition(gameState, oldPos)) #
            for newPos, prob in newPosDist.items():
                bprime[newPos] = self.beliefs[oldPos] * prob + bprime[newPos] 

        bprime.normalize()
        self.beliefs = bprime

    def getBeliefDistribution(self):
        """
        Return the agent's current belief state, a distribution over ghost
        locations conditioned on all evidence and time passage. This method
        essentially converts a list of particles into a belief distribution (a
        Counter object)
        """
        

        return self.beliefs