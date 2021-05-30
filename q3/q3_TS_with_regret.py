# -*- coding: utf-8 -*-
"""
Created on Mon May 17 18:57:12 2021

@author: maxime
"""

"""
Created on Sun May 16 18:08:13 2021

@author: Fabien
"""


import numpy as np 
import matplotlib.pyplot as plt

from Environment import Environment
from TS_Learner import TS_Learner
from UCB_Learner import UCB_Learner

from q1_functions import objective_function, conversion1
from q1_optimiser import p1_optimal

#Variables fixed

alpha = [0.25]*16
p2 = 40 # price before promotion
P1 = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55] #arms
n_clients_per_class = [50,40,25,15]
n_arms = len(P1)
p = np.mean(np.array([conversion1(p1) for p1 in P1]), axis = 1) # proba of each arm
p1_opt = p1_optimal(p1 = np.mean(P1), p2 = p2, alpha = alpha, n_clients_per_class = n_clients_per_class)
p1_opt = 45

T = 365

n_experiments = 100

ts_rewards_per_experiment = []
ts_pulled_arms_global = np.array([0]*len(P1))

ucb_rewards_per_experiment = []
ucb_pulled_arms_global = np.array([0]*len(P1))

for e in range(0, n_experiments):
    print(f"Experiment number {e}\n")
    env = Environment(n_arms = n_arms, probabilities = p)
    ts_learner = TS_Learner(n_arms, P1, p2, alpha, n_clients_per_class)
    ucb_learner = UCB_Learner(n_arms, P1, p2, alpha, n_clients_per_class)
    ts_memory_pulled_arm = []
    ucb_memory_pulled_arm = []
    
    for t in range(0, T):
        #TS Learner
        pulled_arm = ts_learner.pull_arm()
        ts_memory_pulled_arm.append(pulled_arm)
        
        cv_rate_1 = env.round(pulled_arm, n_clients_per_class)
        profit = objective_function(P1[pulled_arm], p2, alpha, n_clients_per_class)
        ts_learner.update(pulled_arm, cv_rate_1, profit)


        #UCB Learner #TODO : verify if it is correct
        pulled_arm = ucb_learner.pull_arm()
        ucb_memory_pulled_arm.append(pulled_arm)
        
        cv_rate_1 = env.round(pulled_arm, n_clients_per_class)
        profit = objective_function(P1[pulled_arm], p2, alpha, n_clients_per_class)
        ucb_learner.update(pulled_arm, cv_rate_1, profit)

        
    n_pulled_arms = [((np.array(ts_memory_pulled_arm) == i).sum()) for i in range(len(P1))]
    ts_pulled_arms_global += np.array(n_pulled_arms)

    ucb_n_pulled_arms = [((np.array(ucb_memory_pulled_arm) == i).sum()) for i in range(len(P1))]
    ucb_pulled_arms_global += np.array(ucb_n_pulled_arms)


    print(f" for the experiment number {e} the number of time each arm has been pulled are \n {n_pulled_arms} \n")
    most_pulled_arm = np.argmax(n_pulled_arms)
    n_pulled_arms[most_pulled_arm] = 0
    second_most_pulled_arm = np.argmax(n_pulled_arms)
    print(f"so the 2 most pulled arms are {P1[most_pulled_arm]} and {P1[second_most_pulled_arm]}")


    ts_rewards_per_experiment.append(ts_learner.total_profit)
    ucb_rewards_per_experiment.append(ucb_learner.total_profit)

ts_pulled_arms_global = 1300* ts_pulled_arms_global / max(ts_pulled_arms_global)
ucb_pulled_arms_global = 1300* ucb_pulled_arms_global / max(ucb_pulled_arms_global)
    
opt = objective_function(p1_opt, p2, alpha, n_clients_per_class)

plt.figure(0)
plt.xlabel("t")
plt.ylabel("Regret")
plt.plot(np.cumsum(np.mean(opt - ts_rewards_per_experiment, axis = 0)), 'b')
plt.legend(["TS"])

plt.figure(1)
plt.xlabel("t")
plt.ylabel("Regret")
plt.plot(np.cumsum(np.mean(opt - ucb_rewards_per_experiment, axis = 0)), 'b')
plt.legend(["UCB"])

plt.figure(2)
X = np.linspace(10, 60, 100)
Y = [objective_function(x, p2, alpha, n_clients_per_class) for x in X]
plt.xlabel('TS : prize of item 1')
plt.ylabel('profit over the day')
plt.scatter(P1, ts_pulled_arms_global)
plt.plot(X, Y, 'g')

plt.figure(3)
X = np.linspace(10, 60, 100)
Y = [objective_function(x, p2, alpha, n_clients_per_class) for x in X]
plt.xlabel('UCB : prize of item 1')
plt.ylabel('profit over the day')
plt.scatter(P1, ucb_pulled_arms_global)
plt.plot(X, Y, 'g')
plt.show()

