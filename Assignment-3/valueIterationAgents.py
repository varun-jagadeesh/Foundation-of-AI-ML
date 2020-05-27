# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections


class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()  # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here

        "*** YOUR CODE HERE ***"

        # Iterate all states and actions from every state
        for i in range(0, self.iterations):
            # Stores new value for the state
            new_values = util.Counter()

            mdp_states = self.mdp.getStates()
            # Iterate over all state
            for state in mdp_states:

                terminal = self.mdp.isTerminal(state)
                if terminal:
                    new_values[state] = 0

                else:
                    # Temporary variable
                    temp = list()

                    # Iterate over all actions and compute qvalue for every action
                    actions = self.mdp.getPossibleActions(state)
                    for action in actions:
                        # Finding the qvalue for this action and this state
                        q_value = self.computeQValueFromValues(state, action)
                        temp.append(q_value)

                    new_values[state] = max(temp)
            self.values = new_values

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        # Initialize q_value to 0
        q_value = 0

        transitions = self.mdp.getTransitionStatesAndProbs(state, action)
        # Iterate over transition for each state
        for next_state, transition in transitions:
            value_next_state = self.values[next_state]
            # Discount factor
            discount = self.discount
            # Calculate reward
            reward = self.mdp.getReward(state, action, next_state)
            # Finding the Q value
            q_value += transition * (reward + discount * value_next_state)

        return q_value

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        # Checking terminal state
        terminal = self.mdp.isTerminal(state)
        if terminal:
            return None

        else:
            # Initialise max_value to -infinity
            max_value = -float("inf")

            actions = self.mdp.getPossibleActions(state)
            max_action = None
            # Iterate over all actions and compute q value in a given state and return max_action
            for action in actions:

                # q value of a given state and action
                value = self.computeQValueFromValues(state, action)
                if value > max_value:
                    max_value = value
                    max_action = action

            # return max_action in a given state
            return max_action

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        # Iterate over all possible actions
        for i in range(self.iterations):
            # Stores new value for the state
            new_values = util.Counter()
            # an empty temp  list
            temp = list()
            mdp_states = self.mdp.getStates()

            state = mdp_states[i % len(mdp_states)]

            # Iterate over all actions and compute qvalue for every action
            actions = self.mdp.getPossibleActions(state)
            for action in actions:
                q_value = self.computeQValueFromValues(state, action)
                temp.append(q_value)

            if temp:
                new_values[state] = max(temp)

            for key in new_values.keys():
                self.values[key] = new_values[key]


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """

    def __init__(self, mdp, discount=0.9, iterations=100, theta=1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        # Initialization  of values and temp
        values = util.Counter()
        temp = util.Counter()
        # An empty predecessors list
        predecessors = list()
        # Counter
        count = 0
        # Initialize an empty priority queue
        priority_queue = util.PriorityQueue()

        # Computing predecessors of all states
        mdp_states = self.mdp.getStates()
        for state in mdp_states:
            # Computing predecessor of a state and I am making sure to store it as a set to avoid duplicated
            predecessor = set()

            for states in mdp_states:

                # Iterate over all actions
                actions = self.mdp.getPossibleActions(states)
                for action in actions:

                    transitions = self.mdp.getTransitionStatesAndProbs(states, action)
                    for next_state, transition in transitions:

                        if transition > 0 and next_state == state:
                            predecessor.add(states)

            values[state] = count
            count += 1
            predecessors.append(predecessor)

        # Iterate over all states for each non terminal states
        for state in mdp_states:

            # terminal state
            terminal = self.mdp.isTerminal(state)
            # if condition for terminal
            if terminal:
                continue
            # find the current value of the state
            cur_value = self.getValue(state)

            actions = self.computeActionFromValues(state)
            if actions:
                # Find the absolute value of the difference between the current value and the highest q value (temp
                # value)
                temp_value = self.computeQValueFromValues(state, actions)
                temp[state] = temp_value
                difference_value = abs(cur_value - temp_value)
                # pushing the state s into the priority queue with -diff
                priority_queue.push(state, -difference_value)

            else:
                temp[state] = cur_value

        # Iterate over all iterations from 0
        for i in range(0, self.iterations):

            # if priority queue is empty the terminate
            if priority_queue.isEmpty():
                break
                # pop a state from priority queue
            front = priority_queue.pop()

            # Update the value of state if it not the terminal state
            terminal = self.mdp.isTerminal(front)
            if terminal:
                continue

            else:
                self.values[front] = temp[front]

            # Iterate for each predecessor
            for predecessor in predecessors[values[front]]:
                actions = self.computeActionFromValues(predecessor)
                current_value = self.getValue(predecessor)

                if actions:
                    # finding the absolute value of the difference between the current value and highest value(temp)
                    # acroos all actions
                    temp_value = self.computeQValueFromValues(predecessor, actions)
                    difference_value = abs(current_value - temp_value)
                    temp[predecessor] = temp_value
                    # Checking if the difference is greater than theta and updating the priority quest with -diff

                    if difference_value > self.theta:
                        priority_queue.update(predecessor, -difference_value)
