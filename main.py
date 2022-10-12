
'''
3 times each scenario
One agent
episode length vs episode
episode reward with time

change epsilon
change alpha
change discount

two agents two targets
episode length vs episode
agents reward with time

change epsilon - for one; for both
change alpha
change discount

2a 2t global reward
episode length vs episode
episode reward with time

change epsilon - for one; for both
change alpha
change discount
'''

from env import Qtable
import matplotlib.pyplot as plt

if __name__ == "__main__":
    #number of episode we will run
    # decreased from 10000
    n_episodes = 3500

    # figure
    figure, axis = plt.subplots(3,1)
    figure2, axis2 = plt.subplots(3,1)

    # for each epsilon
    k = 0
    axis[k].set_title("Episode length Vs Episode for varying epsilon")
    axis2[k].set_title("Agent Reward Vs Episode for varying epsilon")

    for j in [0.1, 0.5, 0.9]:
        # for each epoch
        for i in range(3):
            # grid creation
            Qobj = Qtable.QTable(n_ep=n_episodes, epsilon=j)

            # Value iteration
            Qobj.updateQ()

            # plot
            results = Qobj.results(axis, axis2, k)

        axis[k].set_xlabel('Episode Length')
        axis[k].set_ylabel('Episode No. \n for epsilon=%.2f' %j)
        axis[k].grid()
        figure.legend(['epoch 1', 'epoch 2', 'epoch 3'])

        axis2[k].set_xlabel('Total Agent Reward')
        axis2[k].set_ylabel('Episode No. \n for epsilon=%.2f '%j)
        axis2[k].grid()
        figure2.legend(['epoch 1', 'epoch 2', 'epoch 3'])
        k = k + 1
    
    # figure
    figure3, axis3 = plt.subplots(3,1)
    figure4, axis4 = plt.subplots(3,1)

    # for each lr
    k = 0
    axis3[k].set_title("Episode length Vs Episode for varying learning rate")
    axis4[k].set_title("Agent Reward Vs Episode for varying learning rate")

    for j in [0.001, 0.01, 0.1]:
        # for each epoch
        for i in range(3):
            # grid creation
            Qobj = Qtable.QTable(n_ep=n_episodes, lr=j)

            # Value iteration
            Qobj.updateQ()

            # plot
            results = Qobj.results(axis3, axis4, k)

        axis3[k].set_xlabel('Episode Length ')
        axis3[k].set_ylabel('Episode No.\n for lr=%.2f' %j )
        axis3[k].grid()
        figure3.legend(['epoch 1', 'epoch 2', 'epoch 3'])

        axis4[k].set_xlabel('Total Agent Reward')
        axis4[k].set_ylabel('Episode No. \n for lr=%.2f '%j)
        axis4[k].grid()
        figure4.legend(['epoch 1', 'epoch 2', 'epoch 3'])
        k = k + 1

    # figure
    figure5, axis5 = plt.subplots(3,1)
    figure6, axis6 = plt.subplots(3,1)

    # for each gamma
    k = 0
    axis5[k].set_title("Episode length Vs Episode for varying discount factor")
    axis6[k].set_title("Agent Reward Vs Episode for varying discount factor")

    for j in [0.5, 0.7, 0.9]:
        # for each epoch
        for i in range(3):
            # grid creation
            Qobj = Qtable.QTable(n_ep=n_episodes, gamma=j)

            # Value iteration
            Qobj.updateQ()

            # plot
            results = Qobj.results(axis5, axis6, k)

        axis5[k].set_xlabel('Episode Length')
        axis5[k].set_ylabel('Episode No. \n for gamma=%.2f' %j)
        axis5[k].grid()
        figure5.legend(['epoch 1', 'epoch 2', 'epoch 3'])

        axis6[k].set_xlabel('Total Agent Reward')
        axis6[k].set_ylabel('Episode No. \n for gamma=%.2f' %j)
        axis6[k].grid()
        figure6.legend(['epoch 1', 'epoch 2', 'epoch 3'])
        k = k + 1
    plt.show()
