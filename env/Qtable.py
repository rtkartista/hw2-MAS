import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import pdb
class QTable():
    def __init__(self, shape=[5,10], terminalStates=[[2, 7], [3, 7]], n_ep=10, epsilon=0.2, lr=0.01, gamma=0.90) -> None:
        self.epsilon = epsilon
        self.epsilon2 = 0.2
        self.lr = lr
        self.lr2 = 0.01
        self.gamma = gamma
        self.gamma2 = 0.9
        self.actionL = [0, -1]
        self.actionR = [0, 1]
        self.actionU = [-1, 0]
        self.actionD = [1, 0]
        self.actionS = [0, 0]
        self.actions = ['l', 'r', 'u', 'd', 's']
        self.terminationStates = [[2, 7], [3,7]]
        self.gridr = shape[0]
        self.gridc = shape[1]
        self.Q = {}
        self.Q2 = {}
        self.n_episodes = n_ep
        self.episodeLenght = [0 for i in range(self.n_episodes)]
        self.agentReward = [0 for i in range(self.n_episodes)]
        self.agentReward2 = [0 for i in range(self.n_episodes)]
        self.isGoalReached = False

        # create a grid
        self.createStates()
        self.initalState()
        # initialize Q(s,a)
        for s in self.states:
            self.Q[(s[0], s[1])] = {}
            for a in self.actions:
                self.Q[s[0],s[1]][a] = 0

        # initialize deltas
        self.deltas = {}
        for s in self.states:
            self.deltas[(s[0], s[1])] = {}
            for a in self.actions:
                self.deltas[(s[0], s[1])][a] = {}

        # initialize Q2(s,a)
        for s in self.states:
            self.Q2[(s[0], s[1])] = {}
            for a in self.actions:
                self.Q2[s[0],s[1]][a] = 0

        # initialize deltas
        self.deltas2 = {}
        for s in self.states:
            self.deltas2[(s[0], s[1])] = {}
            for a in self.actions:
                self.deltas2[(s[0], s[1])][a] = {}

    def createStates(self):
        self.states = [[i, j] for i in range(self.gridr) for j in range(self.gridc)]

    def initalState(self):
        # generate random initial state
        self.initState = random.choice(self.states[1:-1])
        while list(self.initState) in self.terminationStates:
            self.initState = random.choice(self.states[1:-1])

        # # generate random initial state
        # self.initState2 = random.choice(self.states[1:-1])
        # while list(self.initState2) in self.terminationStates:
        #     self.initState2 = random.choice(self.states[1:-1])
        self.initState2 = self.initState

    def max_dict(self, d):
        # returns the argmax (key) and max (value) from a dictionary
        max_key = None
        max_val = float('-inf')
        for k, v in d.items():
            if v > max_val:
                max_val = v
                max_key = k
        return max_key, max_val

    def nextAction(self, bestAction):
        if random.uniform(0,1) > self.epsilon:
            self.actionNext = random.choice(self.actions)
        else:
            # action with the max Qvalue
            self.actionNext = bestAction

    def terminalGoal(self):
        self.terminalGoal1 = random.choice(self.terminationStates)
        goal2 = list(self.terminationStates)
        goal2.remove(self.terminalGoal1)
        self.terminalGoal2 = goal2[0]
        
    def nextStateReward(self, state, currA):
        if state in self.terminationStates:
            return 20, state

        self.nextAction(currA)

        if self.actionNext == "l":
            currAction = self.actionL
        elif self.actionNext == "r":
            currAction = self.actionR
        elif self.actionNext == "u":
            currAction = self.actionU
        elif self.actionNext == "d":
            currAction = self.actionD
        else:
            currAction = self.actionS
        finalState = np.array(state) + np.array(currAction)
        # if robot crosses wall
        if (-1 in list(finalState)) or (self.gridc == finalState[1]) or (self.gridr == finalState[0]):
            finalState = state

        if list(finalState) in self.terminationStates:
            return 20, list(finalState)

        return -1, list(finalState)

    def nextStateReward2(self, state, state2, currA, currA2):
        if (state or state2) in self.terminationStates:
            return 20

        self.nextAction(currA)

        if self.actionNext == "l":
            currAction = self.actionL
        elif self.actionNext == "r":
            currAction = self.actionR
        elif self.actionNext == "u":
            currAction = self.actionU
        elif self.actionNext == "d":
            currAction = self.actionD
        else:
            currAction = self.actionS
        finalState = np.array(state) + np.array(currAction)

        self.nextAction(currA2)

        if self.actionNext == "l":
            currAction = self.actionL
        elif self.actionNext == "r":
            currAction = self.actionR
        elif self.actionNext == "u":
            currAction = self.actionU
        elif self.actionNext == "d":
            currAction = self.actionD
        else:
            currAction = self.actionS
        finalState2 = np.array(state2) + np.array(currAction)
        
        # if robot crosses wall
        if (-1 in list(finalState)) or (self.gridc == finalState[1]) or (self.gridr == finalState[0]):
            finalState = state
        if (-1 in list(finalState2)) or (self.gridc == finalState2[1]) or (self.gridr == finalState2[0]):
            finalState2 = state2

        if list(finalState) or list(finalState2) in self.terminationStates:
            return 20
        return -1

    def updateQ(self):
        for it in tqdm(range(self.n_episodes)):
            self.initalState()
            state = self.initState
            i = 0
            while True:
            # for i in range(500):
                # we reached the end
                finalState = state
                if finalState in self.terminationStates:
                    # print("Terminal Goal Reached.")
                    break

                a, _ = self.max_dict(self.Q[(state[0], state[1])])
                reward, finalState = self.nextStateReward(state, a)
                self.agentReward[it] += reward 
                # modify Q function
                old_qsa = self.Q[(state[0], state[1])][a]
                _, max_q_s2a2 = self.max_dict(self.Q[(finalState[0], finalState[1])])
                self.Q[(state[0], state[1])][a] = old_qsa + self.lr*(reward + self.gamma*max_q_s2a2 - old_qsa)
                self.deltas[(state[0], state[1])][a][it] = np.abs(old_qsa - self.Q[(state[0], state[1])][a])
                # pdb.set_trace()
                state = finalState
                i = i + 1

            # print(i)
            self.episodeLenght[it] = i
    
    # Each agent has a Q table
    # No collisions
    # They have independent motion, can capture either target. 
    # If a target is reached, the other agent looks for another target.
    def update2Q(self):
        for it in tqdm(range(self.n_episodes)):
            self.initalState()
            state = self.initState
            state2 = self.initState2
            i = 0

            # assign random goals
            self.terminalGoal()
            while True:
            # for i in range(100):
                
                # both agents reached terminal states
                if self.goalCheck(state2, state):
                    break
                else:
                    # we reached the end
                    finalState = state
                    if finalState not in self.terminationStates:   
                        a, _ = self.max_dict(self.Q[(state[0], state[1])])
                        reward, finalState = self.nextStateReward(state, a)
                        self.agentReward[it] += reward 
                        # modify Q function
                        old_qsa = self.Q[(state[0], state[1])][a]
                        _, max_q_s2a2 = self.max_dict(self.Q[(finalState[0], finalState[1])])
                        self.Q[(state[0], state[1])][a] = old_qsa + self.lr*(reward + self.gamma*max_q_s2a2 - old_qsa)
                        self.deltas[(state[0], state[1])][a][it] = np.abs(old_qsa - self.Q[(state[0], state[1])][a])
                        # pdb.set_trace()
                        state = finalState

                    finalState2 = state2
                    if finalState2 not in self.terminationStates: 
                        a2, _ = self.max_dict(self.Q2[(state2[0], state2[1])])
                        reward2, finalState2 = self.nextStateReward(state2, a2)
                        self.agentReward2[it] += reward2 
                        # modify Q function
                        old_qsa2 = self.Q2[(state2[0], state2[1])][a2]
                        _, max_q_s2a2 = self.max_dict(self.Q2[(finalState2[0], finalState2[1])])
                        self.Q2[(state2[0], state2[1])][a2] = old_qsa2 + self.lr2*(reward2 + self.gamma2*max_q_s2a2 - old_qsa2)
                        self.deltas2[(state2[0], state2[1])][a2][it] = np.abs(old_qsa2 - self.Q2[(state2[0], state2[1])][a2])
                        # pdb.set_trace()
                        state2 = finalState2
   
                i = i + 1

            # print(i)
            self.episodeLenght[it] = i
    
    def goalCheck2(self, finalState, goal):
        if finalState == goal:
            return True
        else:
            return False

    # Each agent has a Q table
    # No collisions
    # They share rewards. 
    # If a target is reached, the other agent looks for another target.
    def updateQ3(self):
        for it in tqdm(range(self.n_episodes)):
            # initialize goal
            self.initalState()
            state = self.initState
            state2 = self.initState2
            i = 0

            # assign random goals
            self.terminalGoal()
            a, _ = self.max_dict(self.Q[(state[0], state[1])])
            reward, finalState = self.nextStateReward(state, a)
            a2, _ = self.max_dict(self.Q2[(state2[0], state2[1])])
            reward2, finalState2 = self.nextStateReward(state2, a2)

            while True:
                # both agents reached terminal states
                if self.goalCheck(state2, state):
                    break
                else:
                    # we reached the end
                    finalState = state
                    if finalState not in self.terminationStates:   
                        a, _ = self.max_dict(self.Q[(state[0], state[1])])
                        reward, finalState = self.nextStateReward(state, a)
                        self.agentReward[it] += reward 
                        # modify Q function
                        old_qsa = self.Q[(state[0], state[1])][a]
                        _, max_q_s2a2 = self.max_dict(self.Q[(finalState[0], finalState[1])])
                        self.Q[(state[0], state[1])][a] = old_qsa + self.lr*(reward2+reward + self.gamma*max_q_s2a2 - old_qsa)
                        self.deltas[(state[0], state[1])][a][it] = np.abs(old_qsa - self.Q[(state[0], state[1])][a])
                        # pdb.set_trace()
                        state = finalState

                    finalState2 = state2
                    if finalState2 not in self.terminationStates: 
                        a2, _ = self.max_dict(self.Q2[(state2[0], state2[1])])
                        reward2, finalState2 = self.nextStateReward(state2, a2)
                        self.agentReward2[it] += reward2 
                        # modify Q function
                        old_qsa2 = self.Q2[(state2[0], state2[1])][a2]
                        _, max_q_s2a2 = self.max_dict(self.Q2[(finalState2[0], finalState2[1])])
                        self.Q2[(state2[0], state2[1])][a2] = old_qsa2 + self.lr2*(reward+reward2 + self.gamma2*max_q_s2a2 - old_qsa2)
                        self.deltas2[(state2[0], state2[1])][a2][it] = np.abs(old_qsa2 - self.Q2[(state2[0], state2[1])][a2])
                        # pdb.set_trace()
                        state2 = finalState2
   
                i = i + 1

            # print(i)
            self.episodeLenght[it] = i
    
    def goalCheck(self, a, b):
        if (a and b) in self.terminationStates:
            return True
        else:
            return False

    def results(self, ax, ax2, epj):
        # make data
        x = np.linspace(0, self.n_episodes,num=self.n_episodes)
        y = self.episodeLenght
        # plot
        ax[epj].plot(x, y, linewidth=1.0)

        y2 = self.agentReward
        # plot
        ax2[epj].plot(x, y2, linewidth=1.0)

    def resultsMulti(self, ax, ax2, ax3, epj):
        # make data
        x = np.linspace(0, self.n_episodes,num=self.n_episodes)
        y = self.episodeLenght
        # plot
        ax[epj].plot(x, y, linewidth=1.0)

        y2 = self.agentReward
        # plot
        ax2[epj].plot(x, y2, linewidth=1.0)

        y3 = self.agentReward2
        # plot
        ax3[epj].plot(x, y3, linewidth=1.0)
