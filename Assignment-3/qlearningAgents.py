# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random, util, math


class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """

    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        # Initalise Q values
        self.q_values = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        # returns the q values with state and action
        if (state, action) in self.q_values:
            return self.q_values[(state, action)]

        else:
            # return 0.0 if never seen a state
            self.q_values[(state, action)] = 0.0
            return self.q_values[(state, action)]

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        # Actions from state
        actions = self.getLegalActions(state)
        # if actions are not in actions
        if not actions:
            return 0.0

        else:
            # Initialise max_q_value to -infinity
            max_q_value = -float('inf')
            # Iterate for every action in the list of actions
            for action in actions:

                max_q_value = max(max_q_value, self.getQValue(state, action))
            if max_q_value != float('-inf'):
                # return the max_q_values
                return max_q_value

            else:
                return 0

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        # Actions from state
        actions = self.getLegalActions(state)
        # If not in actions
        if not actions:
            return None  # Returning None

        else:
            # empty temp list
            temp_actions = list()
            max_q_value = self.computeValueFromQValues(state)

            # Iterate for every actions
            for action in actions:

                q_value = self.getQValue(state, action)
                if q_value > max_q_value:
                    max_q_value = q_value

                elif q_value == max_q_value:
                    temp_actions.append(action)

            #actions using random choice
            return random.choice(temp_actions)

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        # if not in actions
        if not legalActions:
            return action

        if util.flipCoin(self.epsilon):
            action = random.choice(legalActions)  # actions using random choice

        else:
            action = self.computeActionFromQValues(state)

        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        # current and previous values
        temp_q_values = self.computeValueFromQValues(nextState)
        previous_q_values = self.q_values[(state, action)]
        # alpha (learning rate)
        alpha = self.alpha
        # discount factor
        discount = self.discount

        self.q_values[(state, action)] = ((1 - alpha) * previous_q_values) + alpha * (reward + discount * temp_q_values)

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05, gamma=0.8, alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self, state)
        self.doAction(state, action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """

    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        # Initialization
        value = 0
        weights = self.getWeights()
        features = self.featExtractor.getFeatures(state, action)

        # Iterate over every feature and return q value
        for feature, val in features.items():
            value += (weights[feature] * val)

        return value

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        # current and previous q values
        temp_q_value = self.computeValueFromQValues(nextState)
        previous_q_value = self.getQValue(state, action)
        weights = self.getWeights()
        features = self.featExtractor.getFeatures(state, action)
        # discount factor
        discount = self.discount
        # alpha(learning rate)
        alpha = self.alpha
        # iterate over every feature and update weights

        for feature in features:
            weights[feature] += alpha * ((reward + discount * temp_q_value) - previous_q_value) * features[feature]

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
