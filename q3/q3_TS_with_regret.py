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

from q1_functions import objective_function, conversion1, conversion2
from q1_optimiser import p1_optimal

#Variables fixed

alpha = [0.25]*16 # promotions are distributed uniformly with no distinction of class
p2 = 40 # price before promotion
p2_after_promo = [p2 * (1 - P) for P in [0, 0.10, 0.20, 0.30]] #price after promotions, be careful that the promos are the same than in obj_fun. 
P1 = [20, 30, 35, 40, 45, 50] #arms

n_clients_per_class = [50,40,25,15]
n_arms = len(P1)
p_1 = np.array([conversion1(p1) for p1 in P1])

p1_opt = p1_optimal(p1 = np.mean(P1), p2 = p2, alpha = alpha, n_clients_per_class = n_clients_per_class)
p_2 = conversion2(p2_after_promo)
T = 365

n_experiments = 50

ts_rewards_per_experiment = []
ts_pulled_arms_global = np.array([0]*len(P1))

ucb_rewards_per_experiment = []
ucb_pulled_arms_global = np.array([0]*len(P1))


for e in range(0, n_experiments):
    print(f"Experiment number {e}\n")
    env = Environment(n_arms = n_arms, probabilities_1 = p_1, probabilities_2 = p_2)
    ts_learner = TS_Learner(n_arms, P1, p2, alpha, n_clients_per_class)
    ucb_learner = UCB_Learner(n_arms, P1, p2, alpha, n_clients_per_class)
    ts_memory_pulled_arm = []
    ucb_memory_pulled_arm = []
    
    for t in range(0, T):
        #TS Learner
        pulled_arm = ts_learner.pull_arm()
        ts_memory_pulled_arm.append(pulled_arm)
        cv_rate_1 = env.round(pulled_arm,n_clients_per_class)
        profit = objective_function(P1[pulled_arm], p2, alpha, n_clients_per_class)
        ts_learner.update(pulled_arm, cv_rate_1, profit)
        
        #UCB
        pulled_arm = ucb_learner.pull_arm()
        ucb_memory_pulled_arm.append(pulled_arm)
        cv_rate_1 = env.round(pulled_arm,n_clients_per_class)
        profit = objective_function(P1[pulled_arm], p2, alpha, n_clients_per_class)
        ucb_learner.update(pulled_arm, cv_rate_1, profit)
        
        
        
    ts_n_pulled_arms = [((np.array(ts_memory_pulled_arm) == i).sum()) for i in range(len(P1))]
    ts_pulled_arms_global += np.array(ts_n_pulled_arms)
    print(f" for the experiment number {e} the number of time each arm has been pulled are \n {ts_n_pulled_arms} \n")
    ts_most_pulled_arm = np.argmax(ts_n_pulled_arms)
    ts_n_pulled_arms[ts_most_pulled_arm] = 0
    ts_second_most_pulled_arm = np.argmax(ts_n_pulled_arms)
    print(f"so the 2 most pulled arms are {P1[ts_most_pulled_arm]} and {P1[ts_second_most_pulled_arm]} \n \n -------------------------- \n")


    ucb_n_pulled_arms = [((np.array(ucb_memory_pulled_arm) == i).sum()) for i in range(len(P1))]
    ucb_pulled_arms_global += np.array(ucb_n_pulled_arms)
    print(f" for the experiment number {e} the number of time each arm has been pulled are \n {ucb_n_pulled_arms} \n")
    ucb_most_pulled_arm = np.argmax(ucb_n_pulled_arms)
    ucb_n_pulled_arms[ucb_most_pulled_arm] = 0
    ucb_second_most_pulled_arm = np.argmax(ucb_n_pulled_arms)
    print(f"so the 2 most pulled arms are {P1[ucb_most_pulled_arm]} and {P1[ucb_second_most_pulled_arm]} \n \n -------------------------- \n")



    ts_rewards_per_experiment.append(ts_learner.total_profit)
    ucb_rewards_per_experiment.append(ucb_learner.total_profit)

opt = objective_function(p1_opt, p2, alpha, n_clients_per_class)
ts_pulled_arms_global = opt* ts_pulled_arms_global / max(ts_pulled_arms_global)  #we use the optimal value just as a scaling factor so that the scale of the points matches the scale of the optimal curve
ucb_pulled_arms_global = opt* ucb_pulled_arms_global / max(ucb_pulled_arms_global)


'''
        l'UCB a tendence à "butiner" les bras. Cela se transcrit sur les graphs
        par un regret plus élevé que TS mais une bien meilleure approximation
        des bras sous_optimaux.
'''    


plt.figure(0)
plt.xlabel("t")
plt.ylabel("Regret")
plt.plot(np.cumsum(np.mean(opt - ts_rewards_per_experiment, axis = 0)), 'b')
plt.plot(np.cumsum(np.mean(opt - ucb_rewards_per_experiment, axis = 0)), 'r')
plt.legend(["TS","UCB"])

plt.figure(1)
X = np.linspace(10, 60, 100)
Y = [objective_function(x, p2, alpha, n_clients_per_class) for x in X]
plt.xlabel('prize of item 1')
plt.ylabel('profit over the day')
plt.scatter(P1, ts_pulled_arms_global)
plt.legend(["TS"])
plt.plot(X, Y, 'g')

plt.figure(2)
X = np.linspace(10, 60, 100)
Y = [objective_function(x, p2, alpha, n_clients_per_class) for x in X]
plt.xlabel('prize of item 1')
plt.ylabel('profit over the day')
plt.scatter(P1, ucb_pulled_arms_global)
plt.legend(["UCB"])
plt.plot(X, Y, 'g')
plt.show()