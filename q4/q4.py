"""
Created on Sun May 16 18:08:13 2021

@author: Fabien
"""


import numpy as np 
import matplotlib.pyplot as plt
from Environment import Environment
from TS_Learner import TS_Learner
from q1_no_array import objective_function

#Variables fixed

alpha = [0.25]*16
p2 = 40 # price before promotion
P1 = [10,15,25,40] #arms
n_clients_per_class = [50,40,25,15]
n_arms = len(P1)
p = np.array([0.5, 0.25, 0.15, 0.1]) #proba of each arm


T = 365

n_experiments = 100

ts_rewards_per_experiment = []


for e in range(0, n_experiments):
    print(e)
    env = Environment(n_arms = n_arms, probabilities = p)
    ts_learner = TS_Learner(n_arms, P1, p2, alpha, n_clients_per_class)
  
    
    for t in range(0, T):
        #TS Learner
        pulled_arm = ts_learner.pull_arm()
        cv_rate_1 = env.round(pulled_arm)
        profit = objective_function(P1[pulled_arm], p2, alpha, n_clients_per_class)
        ts_learner.update(pulled_arm, cv_rate_1, profit)
        
        
    ts_rewards_per_experiment.append(ts_learner.total_profit)
    

plt.figure(0)
plt.xlabel("t")
plt.ylabel("Expected profit")
plt.plot(np.cumsum(np.mean(ts_rewards_per_experiment, axis = 0)), 'r')
plt.legend(["TS"])
plt.show()

