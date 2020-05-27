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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

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

        "*** YOUR CODE HERE ***"
        # Initializing distance value to infinite
        dist = float("inf")
        ghost_position = list()
        ghost_score = 0
        food_score = 0

        # Get ghost position
        for ghost in newGhostStates:
            ghost_position.append(ghost.getPosition())

        # calculate ghost score
        for position in ghost_position:
            dist = min(dist, manhattanDistance(position, newPos))
            if dist == 0:
                ghost_score = ghost_score - 1.0 / (2 * 1.0)
            else:
                ghost_score = ghost_score - 1.0 / (2 * dist)

        # food list of pacman
        food_list = currentGameState.getFood().asList()
        # calculate food score
        for food in food_list:
            # distance between pacman and food
            dist = min(dist, manhattanDistance(food, newPos))
            food_score = 1.0 / (1.0 + dist)

        # calculate total score
        total_score = successorGameState.getScore() + food_score + ghost_score

        return total_score


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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        # Minimizer function
        def minimizer(gameState, cur_depth, ghost):

            # initializing minimum_value to infinite
            minimum_value = float('inf')

            # evaluation function  is returned if the game is in win or lose state
            if gameState.isLose() or gameState.isWin():
                return self.evaluationFunction(gameState)

            # legal moves  for ghost
            legal_moves = gameState.getLegalActions(ghost)
            for beta in legal_moves:
                # total agents in game
                total_agents = gameState.getNumAgents()
                if ghost == (total_agents - 1):
                    successor = gameState.generateSuccessor(ghost, beta)
                    # gets the minimum value
                    minimum_value = min(minimum_value, maximizer(successor, cur_depth))
                else:
                    successor = gameState.generateSuccessor(ghost, beta)
                    # gets the minimum value
                    minimum_value = min(minimum_value, minimizer(successor, cur_depth, ghost + 1))
            return minimum_value

        # Maximizer function
        def maximizer(gameState, cur_depth):

            # initializing maximum_value to infinite
            maximum_value = float('-inf')
            cur_depth = cur_depth + 1

            # evaluation function is returned if the game is in win or lose state o
            if gameState.isLose() or gameState.isWin() or cur_depth == self.depth:
                return self.evaluationFunction(gameState)

            # agent pacman
            agent_index = 0

            # legal moves of agent
            legal_moves = gameState.getLegalActions(agent_index)
            for alpha in legal_moves:
                successor = gameState.generateSuccessor(agent_index, alpha)
                # gets the maximum value
                maximum_value = max(maximum_value, minimizer(successor, cur_depth, 1))
            return maximum_value

        moves = ""
        # initializing maximum_value to infinite
        maximum_value = float('-inf')

        # agent pacman
        agent_index = 0
        legal_moves = gameState.getLegalActions(agent_index)

        # Minimax search with pacman
        for pacman_move in legal_moves:
            cur_depth = 0
            successor = gameState.generateSuccessor(agent_index, pacman_move)
            max_value = minimizer(successor, cur_depth, 1)

            if max_value > maximum_value:
                maximum_value = max_value
                moves = pacman_move
        return moves


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        # agent pacman
        agent_index = 0

        # initializing alpha and beta to infinite
        alpha = float('-inf')
        beta = float('inf')
        maximum_score = float('-inf')
        moves = ''

        # total agents in game
        total_agents = gameState.getNumAgents()
        # legal moves of agent
        legal_moves = gameState.getLegalActions(agent_index)
        # minimax search with pacman
        for move in legal_moves:
            successor = gameState.generateSuccessor(agent_index, move)
            max_value = self.alpha_beta_pruning(successor, self.depth, alpha, beta, agent_index + 1, total_agents)

            if max_value > maximum_score:
                moves = move
                maximum_score = max_value

            if maximum_score >= beta:
                return moves

            alpha = max(alpha, maximum_score)

        return moves

    # alpha beta pruning
    def alpha_beta_pruning(self, gameState, cur_depth, alpha, beta, agent_index, total_agents):

        # evaluation function  is returned if the game is in win or lose state or depth<=0
        if gameState.isLose() or gameState.isWin() or cur_depth <= 0:
            return self.evaluationFunction(gameState)

        # initializing v to infinite
        v = float('inf')

        if agent_index == 0:
            v = float('-inf')

        # legal moves of agent
        legal_moves = gameState.getLegalActions(agent_index)

        for move in legal_moves:
            if agent_index == 0:
                # max-value functionality
                successor = gameState.generateSuccessor(agent_index, move)
                # max search value
                v = max(v, self.alpha_beta_pruning(successor, cur_depth, alpha, beta, agent_index + 1, total_agents))
                if v > beta:
                    return v
                alpha = max(alpha, v)

            elif agent_index == total_agents - 1:
                # min-value functionality
                successor = gameState.generateSuccessor(agent_index, move)
                # min search value
                v = min(v, self.alpha_beta_pruning(successor, cur_depth - 1, alpha, beta, 0, total_agents))
                if v < alpha:
                    return v
                beta = min(beta, v)

            else:
                successor = gameState.generateSuccessor(agent_index, move)
                v = min(v, self.alpha_beta_pruning(successor, cur_depth, alpha, beta, agent_index + 1, total_agents))
                if v < alpha:
                    return v
                beta = min(beta, v)
        return v


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        def expectimax(gameState, cur_depth, ghost):
            # initializing v to 0
            v = 0
            # evaluation function  is returned if the game is in win or lose state
            if gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            # ghost action
            ghost_legal_move = gameState.getLegalActions(ghost)
            for ghost_move in ghost_legal_move:
                # total agents in game
                total_agents = gameState.getNumAgents()
                if ghost == (total_agents - 1):
                    ghost_successor = gameState.generateSuccessor(ghost, ghost_move)
                    v = v + maximizer(ghost_successor, cur_depth) / len(ghost_legal_move)
                else:
                    ghost_successor = gameState.generateSuccessor(ghost, ghost_move)
                    v = v + expectimax(ghost_successor, cur_depth, ghost + 1) / len(ghost_legal_move)
            return v

        def maximizer(gameState, cur_depth):

            # initializing max_value to infinite
            max_value = float('-inf')
            cur_depth = cur_depth + 1

            # evaluation function  is returned if the game is in win or lose state
            # or agent depth is equal to node depth
            if gameState.isWin() or gameState.isLose() or cur_depth == self.depth:
                return self.evaluationFunction(gameState)

            # agent pacman
            agent_index = 0
            legal_moves = gameState.getLegalActions(agent_index)

            for alpha in legal_moves:
                successor = gameState.generateSuccessor(agent_index, alpha)
                max_value = max(max_value, expectimax(successor, cur_depth, 1))
            return max_value

        # agent pacman
        agent_index = 0
        legal_moves = gameState.getLegalActions(agent_index)

        # initializing max_value to infinite
        max_value = float('-inf')
        max_moves = ''

        for move in legal_moves:
            cur_depth = 0
            successor = gameState.generateSuccessor(agent_index, move)
            maximum_value = expectimax(successor, cur_depth, 1)
            if maximum_value > max_value:
                max_value = maximum_value
                max_moves = move
        return max_moves


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # initialization
    dist = float("inf")
    ghost_position = list()
    ghost_score = 0
    food_score = 0
    capsule_score = 0

    # ghost state
    cur_ghost_states = currentGameState.getGhostStates()
    for ghost in cur_ghost_states:
        ghost_position.append(ghost.getPosition())

    # current position of pacman
    cur_pos = currentGameState.getPacmanPosition()

    # ghost score calculation
    for position in ghost_position:
        dist = min(dist, manhattanDistance(position, cur_pos))
        if dist == 0:
            ghost_score = ghost_score - 1.0 / (2 * 1.0)
        else:
            ghost_score = ghost_score - 1.0 / (2 * dist)

    # food list of pacman
    food_list = currentGameState.getFood().asList()
    # calculate food score and distance between pacman and food
    for food in food_list:
        dist = min(dist, manhattanDistance(food, cur_pos))
        food_score = 1.0 / (1.0 + dist)

    # capsules in the game
    cur_capsules = currentGameState.getCapsules()
    for capsule in cur_capsules:
        dist = min(dist, manhattanDistance(capsule, cur_pos))
        if dist == 0:
            capsule_score = capsule_score + 3
        else:
            capsule_score = 1.0 / (1.0 + capsule_score)

    # calculate the total score
    total_score = currentGameState.getScore() + food_score + ghost_score + capsule_score

    return total_score


# Abbreviation
better = betterEvaluationFunction
