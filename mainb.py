
'''
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
    n_episodes = 5000
    # increased to 10000 and kept the episode size larger
    # figure
    figure, axis = plt.subplots(3,1)
    figure2, axis2 = plt.subplots(3,1)
    figure3, axis3 = plt.subplots(3,1)

    # for each epsilon
    k = 0
    axis[k].set_title("Episode length Vs Episode")
    axis2[k].set_title('Agent "1" Reward Vs Episode')
    axis3[k].set_title('Agent "2" Reward Vs Episode')

    for j in [0.1, 0.5, 0.9]:    
        # for each epoch
        for i in range(3):
            # grid creation
            Qobj = Qtable.QTable(n_ep=n_episodes, epsilon=j)

            # Value iteration
            Qobj.update2Q()

            # plot
            results = Qobj.resultsMulti(axis, axis2, axis3, k)

        axis[k].set_xlabel('Episode Length')
        axis[k].set_ylabel('Episode No.')
        axis[k].grid()
        figure.legend(['epoch 1', 'epoch 2', 'epoch 3'])

        axis2[k].set_xlabel('Total Agent "1" Reward')
        axis2[k].set_ylabel('Total Reward')
        axis2[k].grid()
        figure2.legend(['epoch 1', 'epoch 2', 'epoch 3'])
        
        axis3[k].set_xlabel('Total Agent "2" Reward')
        axis3[k].set_ylabel('Total Reward')
        axis3[k].grid()
        figure2.legend(['epoch 1', 'epoch 2', 'epoch 3'])
        
        k = k + 1

    # for each discount factor
    figure4, axis4 = plt.subplots(3,1)
    figure5, axis5 = plt.subplots(3,1)
    figure6, axis6 = plt.subplots(3,1)
    k = 0

    axis4[k].set_title("Episode length Vs Episode")
    axis5[k].set_title('Agent "1" Reward Vs Episode')
    axis6[k].set_title('Agent "2" Reward Vs Episode')

    for j in [0.001, 0.01, 0.1]:    
        # for each epoch
        for i in range(3):
            # grid creation
            Qobj = Qtable.QTable(n_ep=n_episodes, gamma=j)

            # Value iteration
            Qobj.update2Q()

            # plot
            results = Qobj.resultsMulti(axis4, axis5, axis6, k)

        axis4[k].set_xlabel('Episode Length')
        axis4[k].set_ylabel('Episode No.')
        axis4[k].grid()
        figure4.legend(['epoch 1', 'epoch 2', 'epoch 3'])

        axis5[k].set_xlabel('Total Agent "1" Reward')
        axis5[k].set_ylabel('Total Reward')
        axis5[k].grid()
        figure5.legend(['epoch 1', 'epoch 2', 'epoch 3'])
        
        axis6[k].set_xlabel('Total Agent "2" Reward')
        axis6[k].set_ylabel('Total Reward')
        axis6[k].grid()
        figure6.legend(['epoch 1', 'epoch 2', 'epoch 3'])
        
        k = k + 1
    
    # for each learning rate
    figure7, axis7 = plt.subplots(3,1)
    figure8, axis8 = plt.subplots(3,1)
    figure9, axis9 = plt.subplots(3,1)
    k = 0

    axis7[k].set_title("Episode length Vs Episode")
    axis8[k].set_title('Agent "1" Reward Vs Episode')
    axis9[k].set_title('Agent "2" Reward Vs Episode')

    for j in [0.5, 0.7, 0.9]:    
        # for each epoch
        for i in range(3):
            # grid creation
            Qobj = Qtable.QTable(n_ep=n_episodes, lr=j)

            # Value iteration
            Qobj.update2Q()

            # plot
            results = Qobj.resultsMulti(axis7, axis8, axis9, k)

        axis7[k].set_xlabel('Episode Length')
        axis7[k].set_ylabel('Episode No.')
        axis7[k].grid()
        figure7.legend(['epoch 1', 'epoch 2', 'epoch 3'])

        axis8[k].set_xlabel('Total Agent "1" Reward')
        axis8[k].set_ylabel('Total Reward')
        axis8[k].grid()
        figure8.legend(['epoch 1', 'epoch 2', 'epoch 3'])
        
        axis9[k].set_xlabel('Total Agent "2" Reward')
        axis9[k].set_ylabel('Total Reward')
        axis9[k].grid()
        figure9.legend(['epoch 1', 'epoch 2', 'epoch 3'])
        
        k = k + 1
    plt.show()
