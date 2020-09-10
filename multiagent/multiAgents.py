# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        minDist = -1
        food = newFood.asList()
        for each in food:
            dist = util.manhattanDistance(newPos, each)
            if minDist == -1 or minDist >= dist:
                minDist = dist
        ghostProx = 0
        ghostDist = 1
        for ghostState in successorGameState.getGhostPositions():
            dist = util.manhattanDistance(newPos, ghostState)
            ghostDist += dist
            if dist <= 1:
                ghostProx += 1
        return successorGameState.getScore() + (1 / float(minDist)) - (1 / float(ghostDist)) - ghostProx

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        def minimax(agent, depth, gameState):
            if gameState.isLose() or gameState.isWin() or depth == self.depth:
                return self.evaluationFunction(gameState)
            if agent == 0:
                return max(minimax(1, depth, gameState.generateSuccessor(agent, state)) for state in gameState.getLegalActions(agent))
            else:
                next_agent = agent + 1
                if gameState.getNumAgents() == next_agent:
                    next_agent = 0
                if next_agent == 0:
                   depth += 1
                return min(minimax(next_agent, depth, gameState.generateSuccessor(agent, state)) for state in gameState.getLegalActions(agent))

        action = Directions.WEST
        maximum = float("-inf")
        for state in gameState.getLegalActions(0):
            util = minimax(1, 0, gameState.generateSuccessor(0, state))
            if util > maximum or maximum == float("-inf"):
                action = state
                maximum = util
        return action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        pacman = 0
        def maxAgent(state, depth, a, b):
            if state.isWin() or state.isLose():
                return state.getScore()
            bestScore = float("-inf")
            score = bestScore
            actions = state.getLegalActions(pacman)
            bestAction = Directions.STOP
            for action in actions:
                score = minAgent(state.generateSuccessor(pacman, action), depth, 1, a, b)
                if score > bestScore:
                    bestScore = score
                    bestAction = action
                a = max(a, bestScore)
                if bestScore > b:
                    return bestScore
            if depth == 0:
                return bestAction
            else:
                return bestScore

        def minAgent(state, depth, ghost, a, b):
            if state.isLose() or state.isWin():
                return state.getScore()
            nextGhost = ghost + 1
            if ghost == state.getNumAgents() - 1:
                nextGhost = pacman
            actions = state.getLegalActions(ghost)
            bestScore = float("inf")
            score = bestScore
            for action in actions:
                if nextGhost == pacman:
                    if depth == self.depth - 1:
                        score = self.evaluationFunction(state.generateSuccessor(ghost, action))
                    else:
                        score = maxAgent(state.generateSuccessor(ghost, action), depth + 1, a, b)
                else:
                    score = minAgent(state.generateSuccessor(ghost, action), depth, nextGhost, a, b)
                if score < bestScore:
                    bestScore = score
                b = min(b, bestScore)
                if bestScore < a:
                    return bestScore
            return bestScore
        return maxAgent(gameState, 0, float("-inf"), float("inf"))


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, game_state):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        def expectimax(agent, depth, game_state):
            if game_state.isLose() or game_state.isWin() or depth == self.depth:
                return self.evaluationFunction(game_state)
            if agent == 0:
                return max(expectimax(1, depth, game_state.generateSuccessor(agent, new_state)) for new_state in game_state.getLegalActions(agent))
            else:
                next_agent = agent + 1
                if game_state.getNumAgents() == next_agent:
                    next_agent = 0
                if next_agent == 0:
                    depth += 1
                return sum(expectimax(next_agent, depth, game_state.generateSuccessor(agent, new_state)) for new_state in game_state.getLegalActions(agent)) / float(len(game_state.getLegalActions(agent)))

        maximum = float("-inf")
        action = Directions.WEST
        for agent_state in game_state.getLegalActions(0):
            util = expectimax(1, 0, game_state.generateSuccessor(0, agent_state))
            if util > maximum or maximum == float("-inf"):
                action = agent_state
                maximum = util
        return action

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    min_dist = -1
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    foodList = newFood.asList()
    for food in foodList:
        dist = util.manhattanDistance(newPos, food)
        if min_dist >= dist or min_dist == -1:
            min_dist = dist
    ghost_prox = 0
    ghost_dist = 1
    for ghost_state in currentGameState.getGhostPositions():
        dist = util.manhattanDistance(newPos, ghost_state)
        ghost_dist += dist
        if dist <= 1:
            ghost_prox += 1
    newCap = currentGameState.getCapsules()
    numCaps = len(newCap)
    return currentGameState.getScore() + (1 / float(min_dist)) - (1 / float(ghost_dist)) - ghost_prox - numCaps


# Abbreviation
better = betterEvaluationFunction
