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

from json.encoder import INFINITY
from util import manhattanDistance
from game import Directions
import random, util
import pdb
from itertools import cycle

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
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

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
        currPos = currentGameState.getPacmanPosition()
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood().asList()
        currFood = currentGameState.getFood().asList() 
        newGhostStates = successorGameState.getGhostStates()
  
        #newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        

        "*** YOUR CODE HERE ***"

        # 1st Implementation: (closest ghost*2) + closest food => takes too long to finish game
        # Calculations
        closest_ghost = min(manhattanDistance(newPos, ghost.configuration.pos) for ghost in newGhostStates)
        game_score = successorGameState.getScore()


        if len(newFood) == 0:
            nextGame_closest_food = 0
        else:
            nextGame_closest_food = min(manhattanDistance(newPos, food) for food in newFood) 

        if len(currFood) == 0:
            currGame_closest_food = 1
        else:
            currGame_closest_food = min(manhattanDistance(currPos, food) for food in currFood)
                
        # If distance from closest food is shorter for next state, take this action
        if nextGame_closest_food < currGame_closest_food:
            closest_food = 50
        else:
             closest_food = 0

        # if ghost is far away move towards food
        if closest_ghost > 5 and closest_food == 1:
            closest_food += 100
        
        # if ghost is close to pacman runaway from it (BEST REFLEXES I HAVE EVER SEEN TO DODGE GHOST)
        if closest_ghost < 2:
            closest_ghost -= 100


        return closest_food + closest_ghost + game_score


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.()
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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        r = self.minimax(gameState, agentIndex=0, depth=self.depth)[1]
        
        return r

    # def value(self, state, agentIndex, depth):
    #     # If game is finished
    #     if depth == 0 or state.isWin() or state.isLose():
    #         score = self.evaluationFunction(state)
    #         #print('score:',score)
    #         return  score, Directions.STOP
        
    #     # Pacman's turn
    #     if agentIndex == 0:
    #         return self.max_value(state, agentIndex, depth)
    #     else: # Gohsts` turn 
    #         return self.min_value(state, agentIndex, depth)


    # def max_value(self, state, agentIndex, depth):
    #     if agentIndex == state.getNumAgents() - 1:
    #         agentIndex = 0
    #         depth -= 1

    #     max_eval = float("-inf")
    #     best_action = Directions.STOP
    #     legalActions = state.getLegalActions(agentIndex)

    #     for action in legalActions:
    #         child = state.generateSuccessor(agentIndex, action)
    #         eval = self.value(child, agentIndex+ 1, depth)[0]
    #         #print(eval)
    #         if eval > max_eval:
    #             max_eval, best_action = eval, action
    #         #print(max_eval)
    #     return max_eval, best_action


    # def min_value(self, state, agentIndex, depth):
    #     if agentIndex == state.getNumAgents() - 1:
    #         agentIndex = 0
    #         depth -= 1

    #     min_eval = float("inf")
    #     best_action = Directions.STOP
    #     legalActions = state.getLegalActions(agentIndex)

    #     for action in legalActions:
    #         child = state.generateSuccessor(agentIndex, action)
    #         eval = self.value(child, agentIndex+ 1, depth)[0]
    #         #print(eval)
    #         if eval > min_eval:
    #             min_eval, best_action= eval,action
    #         #print(min_eval)
    #     return min_eval, best_action

    # Recursive Implementation: easier to manage important variables
    def minimax(self, state, agentIndex, depth):
        
        # If game is finished
        if depth == 0 or state.isWin() or state.isLose():
            return  self.evaluationFunction(state), Directions.STOP
        
        # Pacman's turn
        if agentIndex == 0:
            if agentIndex == state.getNumAgents() - 1:
                next_agentMax = 0
                depth -= 1
            else:
                next_agentMax = agentIndex + 1
            
            max_eval = float("-inf")
            best_action_max = Directions.STOP
            legalActions = state.getLegalActions(agentIndex)

            for action in legalActions:
                child = state.generateSuccessor(agentIndex, action)
                eval = self.minimax(child, next_agentMax, depth)[0]
                #print(eval)
                if eval > max_eval:
                    max_eval, best_action_max = eval, action
            #print(max_eval)

            return max_eval, best_action_max
        
        # Gohsts` turn 
        else:
            if agentIndex == state.getNumAgents() - 1:
                next_agentMin = 0
                depth -= 1
            else:
                next_agentMin = agentIndex + 1
            
            min_eval = float("inf")
            best_action_min = Directions.STOP
            legalActions = state.getLegalActions(agentIndex)

            for action in legalActions:
                child = state.generateSuccessor(agentIndex, action)
                eval = self.minimax(child, next_agentMin, depth)[0]
                #print(eval)
                if eval < min_eval:
                    min_eval, best_action_min = eval,action
                #print(min_eval)

            return min_eval, best_action_min
            


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        
        r = self.alphaBeta(gameState, 0, self.depth, -1e9, 1e9)[1]

        return r

    def alphaBeta(self, state, agentIndex, depth, alpha, beta):
        
        # If game is finished
        if depth == 0 or state.isWin() or state.isLose():
            return  self.evaluationFunction(state), Directions.STOP
        
        # Pacman's turn
        if agentIndex == 0:
            if agentIndex == state.getNumAgents() - 1:
                next_agentMax = 0
                depth -= 1
            else:
                next_agentMax = agentIndex + 1
            
            max_eval = float("-inf")
            best_action_max = Directions.STOP
            legalActions = state.getLegalActions(agentIndex)

            for action in legalActions:
                child = state.generateSuccessor(agentIndex, action)
                eval = self.alphaBeta(child, next_agentMax, depth, alpha, beta)[0]
                if eval > max_eval:
                    max_eval, best_action_max = eval, action
        
                alpha = max(alpha, eval)
                if beta < alpha:
                    break
                
        
            return max_eval, best_action_max
        
        # Gohsts` turn 
        else:
            if agentIndex == state.getNumAgents() - 1:
                next_agentMin = 0
                depth -= 1
            else:
                next_agentMin = agentIndex + 1
            
            min_eval = float("inf")
            best_action_min = Directions.STOP
            legalActions = state.getLegalActions(agentIndex)

            for action in legalActions:
                child = state.generateSuccessor(agentIndex, action)
                eval = self.alphaBeta(child, next_agentMin, depth, alpha, beta)[0]
                if eval < min_eval:
                    min_eval, best_action_min = eval, action
                
                beta = min(beta, eval)
                if beta < alpha:
                    break
                
                
            return min_eval, best_action_min

    # def value(self, state, agentIndex, depth, alpha, beta):
    #     if agentIndex >= state.getNumAgents():
    #         agentIndex = 0

    #     # If game is finished
    #     if depth == 0 or state.isWin() or state.isLose():
    #         return self.evaluationFunction(state)
        
    #     # Pacman's turn
    #     if agentIndex == 0:
    #         return self.max_value(state, agentIndex, depth, alpha, beta)
    #     else: # Gohsts` turn 
    #         return self.min_value(state, agentIndex, depth, alpha, beta)


    # def max_value(self, state, agentIndex, depth, alpha, beta):
    #     depth -= 1
    #     evalScore = float("-inf")
    #     legalActions = state.getLegalActions(agentIndex)

    #     for action in legalActions:
    #         child = state.generateSuccessor(agentIndex, action)
    #         eval = self.value(child, agentIndex+1, depth, alpha, beta)
    #         evalScore = max(evalScore,eval)
    #         alpha = max(alpha, eval)
    #         if beta <= alpha:
    #             break

    #     return evalScore


    # def min_value(self, state, agentIndex, depth, alpha, beta):
    #     depth -= 1
    #     evalScore = float("inf")
    #     legalActions = state.getLegalActions(agentIndex)

    #     for action in legalActions:
    #         child = state.generateSuccessor(agentIndex, action)
    #         eval = self.value(child, agentIndex+1, depth, alpha, beta)
    #         evalScore = min(evalScore, eval)
    #         beta = min(beta, eval)
    #         if beta <= alpha:
    #             break

    #     return evalScore


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
        r = self.expectimax(gameState, agentIndex=0, depth=self.depth)[1]

        return r

    def expectimax(self, state, agentIndex, depth):
        
        # If game is finished
        if depth == 0 or state.isWin() or state.isLose():
            return  self.evaluationFunction(state), Directions.STOP
        
        # Pacman's turn
        if agentIndex == 0:
            if agentIndex == state.getNumAgents() - 1:
                next_agentMax, depth  = 0, depth - 1
            else:
                next_agentMax = agentIndex + 1
            
            max_eval = float("-inf")
            best_action_max = Directions.STOP
            legalActions = state.getLegalActions(agentIndex)

            for action in legalActions:
                child = state.generateSuccessor(agentIndex, action)
                eval = self.expectimax(child, next_agentMax, depth)[0]
                if eval > max_eval:
                    max_eval, best_action_max = eval, action

            return max_eval, best_action_max
        
        # Gohsts` turn 
        else:
            if agentIndex == state.getNumAgents() - 1:
                next_agentMin = 0
                depth -= 1
            else:
                next_agentMin = agentIndex + 1
            
            eval = 0.0
            legalActions = state.getLegalActions(agentIndex)
            prob =  1.0 / float(len(legalActions))
            
            for action in legalActions:
                child = state.generateSuccessor(agentIndex, action)
                eval += prob * self.expectimax(child, next_agentMin, depth)[0]
    
            return eval, Directions.STOP

    # def value(self, state, agentIndex, depth):
    #     if agentIndex >= state.getNumAgents():
    #         agentIndex = 0
        
    #     # If game is finished
    #     if depth == 0 or state.isWin() or state.isLose():
    #         return self.evaluationFunction(state),  Directions.STOP
        
    #     # Pacman's turn
    #     if agentIndex == 0:
    #         return self.max_value(state, agentIndex, depth)
    #     else: # Gohsts` turn 
    #         return self.min_value(state, agentIndex, depth)


    # def max_value(self, state, agentIndex, depth):
    #     depth -= 1
    #     evalScore = float("-inf")
    #     legalActions = state.getLegalActions(agentIndex)

    #     for action in legalActions:
    #         child = state.generateSuccessor(agentIndex, action)
    #         eval = self.value(child, agentIndex+ 1, depth)
    #         if type(eval) is tuple:
    #             eval = eval[0]
    #         evalScore = max(evalScore,eval)

    #     return evalScore


    # def exp_value(self, state, agentIndex, depth):
    #     depth -= 1
    #     evalScore = float("inf")
    #     legalActions = state.getLegalActions(agentIndex)

    #     for action in legalActions:
    #         child = state.generateSuccessor(agentIndex, action)
    #         eval = self.value(child, agentIndex+ 1, depth)
    #         if type(eval) is tuple:
    #             eval = eval[0]
    #         prob = self.expectimaxsearch(child,agentIndex, depth)
    #         evalScore = min(evalScore, eval)

    #     return evalScore

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    # Useful information you can extract from a GameState (pacman.py)
    currGhostStates = currentGameState.getGhostStates()
    currPos = currentGameState.getPacmanPosition()
    currFood = currentGameState.getFood().asList() 
    
    if len(currFood) == 0:
        closest_food = 0
    else:
        closest_food = min(manhattanDistance(currPos, food) for food in currFood)

    closest_ghost = min(manhattanDistance(currPos, ghost.configuration.pos) for ghost in currGhostStates)
    game_score = currentGameState.getScore()

    

    # if len(currFood) == 0:
    #     currGame_closest_food = 1
    # else:
    #     currGame_closest_food = min(manhattanDistance(currPos, food) for food in currFood)
               
    #     # If distance from closest food is shorter for next state, take this action
    # if nextGame_closest_food < currGame_closest_food:
    #     closest_food = 50
    # else:
    #     closest_food = 0

    
    if closest_ghost > 5 and closest_food > 1:
        closest_food += 50

    # if ghost is far away move towards food
    if closest_ghost > 5 and closest_food == 1:
        closest_food += 100
        
    # # if ghost is close to pacman runaway from it (BEST REFLEXES I HAVE EVER SEEN TO DODGE GHOST)
    if closest_ghost < 2:
        closest_ghost -= 100

    return closest_food + closest_ghost + game_score

# Abbreviation
better = betterEvaluationFunction
