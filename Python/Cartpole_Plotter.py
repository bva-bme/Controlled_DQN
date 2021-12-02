import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pickle
import torch

matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams.update({'font.size': 18})


def Plot_Q(Q1, Q2):
    ax = plt.axes()
    x = np.linspace(0, len(Q1), len(Q1))
    ax.plot(x, Q1, label='$\displaystyle Q_1$')
    ax.plot(x, Q2, label='$\displaystyle Q_2$')
    ax.set_xlabel('Environment interactions')
    ax.set_ylabel('Q values')
    plt.legend()
    plt.show()


def Plot_Theta(Theta):
    ax = plt.axes()
    x = np.linspace(0, len(Q1), len(Q1))
    lb = np.ones(len(Q1)) * 200
    ub = np.ones(len(Q1)) * 500
    x = np.linspace(0, len(Q1), len(Q1))
    ax.plot(x, Theta, color='blue')
    ax.plot(x, lb, color='red')
    ax.plot(x, ub, color='red')
    ax.set_xlabel('Environment interactions')
    ax.set_ylabel('$\displaystyle \Theta_1$')
    plt.show()


def Plot_Control(u):
    ax = plt.axes()
    x = np.linspace(0, len(u), len(u))
    ax.plot(x, u)
    ax.set_xlabel('Environment interactions')
    ax.set_ylabel('Control input')
    plt.show()


def Plot_Scatter_Theta(theta1, theta2):
    ax = plt.axes()

    x = np.linspace(0, 1000, 1000)
    lb = np.ones(1000) * 200
    ub = np.ones(1000) * 500

    ax.scatter(theta1, theta2)

    ax.plot(lb, x, color='red')
    ax.plot(ub, x, color='red')
    ax.plot(0.8*x, x, color='red')
    ax.plot(1.2*x, x, color='red')

    ax.set_xlabel('$\displaystyle \Theta_1$')
    ax.set_ylabel('$\displaystyle \Theta_2$')
    plt.xlim([0, 1000])
    plt.ylim([0, 1000])
    plt.show()


data = pickle.load(open('cartpole_agent_HinfFix.p', 'rb'))

episode_gains = data[0]
mean_score = data[1]
Q1 = data[2]
Q2 = data[3]
T1 = data[4]
T2 = data[5]
u = data[6]
loss = data[7]

Plot_Q(Q1, Q2)
# Plot_Theta(T1)
# Plot_Scatter_Theta(T1, T2)
# Plot_Control(u)
