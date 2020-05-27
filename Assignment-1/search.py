# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """

    "*** YOUR CODE HERE ***"

     # Using Stack data structure for DFS
    fringe_stack = util.Stack()
    # list of visited states
    visited = list()
    fringe_stack.push((problem.getStartState(), [], []))
    while not fringe_stack.isEmpty():  # loop until there is nothing in the stack
        #pop from the stack
        cur_node = fringe_stack.pop()
        cur_state = cur_node[0]
        cur_action = cur_node[1]
        cur_path = cur_node[2]

        # print('state',cur_state)
    
        if cur_state not in visited:
            visited.append(cur_state)
            # if goal state is reached return the path
            if problem.isGoalState(cur_state):
                return cur_path

            # keep successors of current state in a new list and continue the search
            successor_list = list(problem.getSuccessors(cur_state))  # new list of successors

            # loop for every successor in the new list
            for successor_node in successor_list:
                if successor_node[0] not in visited:  # If 1st successor is not visited
                    new_path = cur_path + [successor_node[1]]
                    fringe_stack.push((successor_node[0], successor_node[1], new_path))
    return list()



def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    # Using Queue data structure for BFS
    fringe_queue = util.Queue()
    # list of visited states
    visited = list()
    fringe_queue.push((problem.getStartState(), [], []))

    while not fringe_queue.isEmpty():  # loop until there is nothing in the queue
        # pop from the queue
        cur_node = fringe_queue.pop()
        cur_state = cur_node[0]
        cur_action = cur_node[1]
        cur_path = cur_node[2]
        # print('state',cur_state)
        if cur_state not in visited:
            visited.append(cur_state)
            if problem.isGoalState(cur_state):
                return cur_path  # return the path if the goal is reached

            # keep successors of current state in a new list and continue the search
            successor_list = list(problem.getSuccessors(cur_state))

            # loop for every successor in the new list
            for successor_node in successor_list:
                if successor_node[0] not in visited:
                    new_path = cur_path + [successor_node[1]]
                    fringe_queue.push((successor_node[0], successor_node[1],new_path))
    return list()


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    # Using Priority Queue for UCS
    fringe_priority_queue = util.PriorityQueue()
    # list of visited states
    visited = list()
    fringe_priority_queue.push(((problem.getStartState(), [], 0), [], 0), [])

    while not fringe_priority_queue.isEmpty(): # loop until there is nothing in the queue
        # pop from priority queue
        cur_node = fringe_priority_queue.pop()
        cur_state = cur_node[0][0]
        cur_action = cur_node[0][1]
        cur_path = cur_node[1]
        cur_cost = cur_node[2]

        if cur_state not in visited:
            visited.append(cur_state)
            if problem.isGoalState(cur_state):
                return cur_path # return the path if the goal is reached

            # keep successors of current state in a new list and continue the search
            successor_list = list(problem.getSuccessors(cur_state))

            #loop for every successor in the new list
            for node in successor_list:
                if node[0] not in visited:
                    new_path = cur_path + [node[1]]
                    fringe_priority_queue.push((node, new_path, cur_cost + node[2]), cur_cost + node[2])

    return list()


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    # using priority queue for A* search as we are using cost and heuristic
    fringe_priority_queue = util.PriorityQueue()
    cost = 0
    # list of visited states
    visited = list()
    heuristic_f = heuristic(problem.getStartState(), problem)
    comb = cost + heuristic_f
    fringe_priority_queue.push((problem.getStartState(), [], cost, []), comb)

    while not fringe_priority_queue.isEmpty(): # loop until there is nothing in the priority queue
        #pop from priority queue
        cur_node = fringe_priority_queue.pop()
        cur_state = cur_node[0]
        cur_action = cur_node[1]
        cur_cost = cur_node[2]

        if cur_state not in visited: # if the current state not the list of visited states
            cur_path = cur_node[3]
            visited.append(cur_state)
            if (problem.isGoalState(cur_state)): # if goal state is reached return the path
                return cur_path

            # keep successors of current state in a new list and continue the search
            successor_list = list(problem.getSuccessors(cur_state))

            # loop for every successor in the new list
            for node in successor_list:
                if node[0] not in visited:
                    cost = cur_cost + node[2]
                    heuristic_f = heuristic(node[0], problem)
                    comb = cost + heuristic_f
                    fringe_priority_queue.push((node[0], node[1], cost, cur_path + [node[1]]), comb)
    return list()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
