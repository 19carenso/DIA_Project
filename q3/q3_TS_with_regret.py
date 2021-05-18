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
from q1_optimiser import p1_optimal
from q1_functions import objective_function

#Variables fixed

alpha = [0.25]*16
p2 = 40 # price before promotion
P1 = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55] #arms
n_clients_per_class = [50,40,25,15]
n_arms = len(P1)
p = np.array([1/len(P1)]*len(P1)) #proba of each arm
p1_opt = p1_optimal(p1 = np.mean(P1), p2 = p2, alpha = alpha, n_clients_per_class = n_clients_per_class)

T = 365

n_experiments = 100

ts_rewards_per_experiment = []


for e in range(0, n_experiments):
    print(f"Experiment number {e}\n")
    env = Environment(n_arms = n_arms, probabilities = p)
    ts_learner = TS_Learner(n_arms, P1, p2, alpha, n_clients_per_class)
    memory_pulled_arm = []
    
    for t in range(0, T):
        #TS Learner
        pulled_arm = ts_learner.pull_arm()
        memory_pulled_arm.append(pulled_arm)
        
        cv_rate_1 = env.round(pulled_arm)
        profit = objective_function(P1[pulled_arm], p2, alpha, n_clients_per_class)
        ts_learner.update(pulled_arm, cv_rate_1, profit)        
    n_pulled_arms = [((np.array(memory_pulled_arm) == i).sum()) for i in range(len(P1))]
    print(f" for the experiment number {e} the number of time each arm has been pulled are \n {n_pulled_arms} \n \n ")    
    ts_rewards_per_experiment.append(ts_learner.total_profit)
    
opt = objective_function(p1_opt, p2, alpha, n_clients_per_class)

plt.figure(0)
plt.xlabel("t")
plt.ylabel("Expected profit")
plt.plot(np.cumsum(np.mean(ts_rewards_per_experiment, axis = 0)), 'r')
plt.legend(["TS"])

plt.figure(1)
plt.xlabel("t")
plt.ylabel("Regret")
plt.plot(np.cumsum(np.mean(opt - ts_rewards_per_experiment, axis = 0)), 'b')
plt.legend(["TS"])
plt.show()