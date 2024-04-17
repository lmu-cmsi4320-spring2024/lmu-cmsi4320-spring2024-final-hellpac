# import itertools
# import util
# import distanceCalculator
# import heapq
# from capture import GameState
# from game import Grid
# from game import Actions
# from game import Directions
# from dataclasses import dataclass, field
# from typing import Any

# class ParticleFilter():
#     """
#     A particle filter for approximately tracking a single ghost.

#     Useful helper functions will include random.choice, which chooses an element
#     from a list uniformly at random, and util.sample, which samples a key from a
#     Counter by treating its values as probabilities.
#     """

#     def __init__(self, gameState: GameState, isRed: bool, index: int, numParticles: int=300) -> None:
#         self.numParticles = numParticles
#         self.beliefs = util.Counter()
#         self.initalGameState = gameState
#         self.targetIndex = index
        
#         self.spawnLoc = gameState.getInitialAgentPosition(self.targetIndex)
        
#         self.red = isRed
#         self.walls = gameState.getWalls()
        
#         allLocations = Grid(32, 16, True).asList()
#         legalPositions = []
#         walls = gameState.getWalls()
#         for location in allLocations:
#             if walls[location[0]][location[1]] == False:
#                 legalPositions.append(location)
#         enemyIndices = gameState.getBlueTeamIndices() if self.red else gameState.getRedTeamIndices()
        
#         # self.possibleEnemyPositions = [gameState.getInitialAgentPosition(index) for index in enemyIndices]
        
#         self.legalPositions = legalPositions
#         self.initializeUniformly()
        
#         self.distancer = distanceCalculator.Distancer(gameState.data.layout)
        
#     def getPositionDistribution(self, enemyLocation: tuple) -> util.Counter:
#         """
#         Returns a equally distributed distribution over successor positions of a given location.
#         """
#         newLocations = Actions.getLegalNeighbors(enemyLocation, self.walls)
#         dist = util.Counter()
#         prob = 1/len(newLocations)
#         for location in newLocations:
#             dist[location] = prob
#         return dist

#     def initializeUniformly(self) -> None:
#         """
#         Initializes a list of particles. Use self.numParticles for the number of
#         particles. Use self.legalPositions for the legal board positions where a
#         particle could be located.  Particles should be evenly (not randomly)
#         distributed across positions in order to ensure a uniform prior.

#         Note: the variable you store your particles in must be a list; a list is
#         simply a collection of unweighted variables (positions in this case).
#         Storing your particles as a Counter (where there could be an associated
#         weight with each position) is incorrect and may produce errors.
#         """
#         uniformedParticles = [None] * self.numParticles
#         for particleNum in range(0, self.numParticles):
#             newParticle = self.legalPositions[particleNum % len(self.legalPositions)]
#             uniformedParticles[particleNum] = newParticle
        
#         self.beliefs = util.Counter()
#         for particlePosition in uniformedParticles:
#             self.beliefs[particlePosition] = self.beliefs[particlePosition] + 1 if self.beliefs[particlePosition] != None else 1
            
#         self.beliefs.normalize()
        
#         return uniformedParticles 
    
#     def getObservationDistribution(self, noisyDistance: float, gameState: GameState, myPos: tuple) -> dict[int: float]:
#         dist = {}
#         for possibleEnemyLoc in self.legalPositions:
#             distanceToLoc = self.distancer.getDistance(possibleEnemyLoc, myPos)
#             dist[distanceToLoc] = gameState.getDistanceProb(distanceToLoc, noisyDistance)
            
#         return dist
    
#     def observe(self, noisyDistance: float, gameState: GameState, myPos: tuple[int]) -> dict[tuple[int]: float]:
#         """
#         Update beliefs based on the given distance observation. Make sure to
#         handle the special case where all particles have weight 0 after
#         reweighting based on observation. If this happens, resample particles
#         uniformly at random from the set of legal positions
#         (self.legalPositions).

#         A correct implementation will handle two special cases:
#           1) When a ghost is captured by Pacman, all particles should be updated
#              so that the ghost appears in its prison cell,
#              self.getJailPosition()

#              As before, you can check if a ghost has been captured by Pacman by
#              checking if it has a noisyDistance of None.

#           2) When all particles receive 0 weight, they should be recreated from
#              the prior distribution by calling initializeUniformly. The total
#              weight for a belief distribution can be found by calling totalCount
#              on a Counter object

#         util.sample(Counter object) is a helper method to generate a sample from
#         a belief distribution.

#         You may also want to use util.manhattanDistance to calculate the
#         distance between a particle and Pacman's position.
#         """
#         emissionModel = self.getObservationDistribution(noisyDistance, gameState, myPos)
        
#         if self.beliefs.totalCount() == 0:
#             self.initializeUniformly()
            
#         for position in self.beliefs.keys():
#             self.beliefs[position] = emissionModel[self.distancer.getDistance(position, myPos)] * self.beliefs[position]
        
#         self.beliefs.normalize()
#         return self.beliefs
    
#     def killedAgent(self) -> None:
#         for loc in self.legalPositions:
#             self.beliefs[loc] = 0.0
#         self.beliefs[self.spawnLoc] = 1.0
    
#     def foundAgent(self, foundLocation: tuple[int]) -> None:
#         for loc in self.legalPositions:
#             self.beliefs[loc] = 0.0
#         self.beliefs[foundLocation] = 1.0

#     def elapseTime(self) -> None:
#         """
#         Update beliefs for a time step elapsing.

#         As in the elapseTime method of ExactInference, you should use:

#           newPosDist = self.getPositionDistribution(self.setGhostPosition(gameState, oldPos))

#         to obtain the distribution over new positions for the ghost, given its
#         previous position (oldPos) as well as Pacman's current position.

#         util.sample(Counter object) is a helper method to generate a sample from
#         a belief distribution.
#         """
#         # newLocations = []
#         # for location in self.possibleEnemyPositions:
#         #     adjLocations = Actions.getLegalNeighbors(location, gameState.getWalls())
#         #     for adjLoc in adjLocations:
#         #         if adjLoc not in newLocations and adjLoc not in self.possibleEnemyPositions:
#         #             newLocations.append(adjLoc)
#         # for newLoc in newLocations:
#         #     self.possibleEnemyPositions.append(newLoc)
        
#         bprime = util.Counter() #initialize bprime
#         for oldPos in self.legalPositions: #summing over all old positions
#             newPosDist = self.getPositionDistribution(oldPos) #
#             for newPos, prob in newPosDist.items():
#                 bprime[newPos] = self.beliefs[oldPos] * prob + bprime[newPos] 

#         bprime.normalize()
#         self.beliefs = bprime

#     def getBeliefDistribution(self):
#         """
#         Return the agent's current belief state, a distribution over ghost
#         locations conditioned on all evidence and time passage. This method
#         essentially converts a list of particles into a belief distribution (a
#         Counter object)
#         """

#         return self.beliefs
    
#     def getTargetIndex(self) -> int:
#         return self.targetIndex
    
#     def getApproximateLocations(self, count: int = 10) -> list[tuple]:
#         locationArray = [None] * len(self.beliefs)
#         for index, (location, prob) in enumerate(self.beliefs.items()):
#             locationArray[index] = HeapNode(prob, location)
        
#         largestNodes = heapq.nlargest(count, locationArray)
#         topLocs = [None] * 10
#         for index, node in enumerate(largestNodes):
#             topLocs[index] = node.loc
            
#         return topLocs
            
# @dataclass(order=True)
# class HeapNode:
#     prob: int
#     loc: Any=field(compare=False)