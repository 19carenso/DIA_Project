# -*- coding: utf-8 -*-
"""
Created on Sun Sep 12 10:25:08 2021

@author: Fabien
"""

import numpy as np 
import matplotlib.pyplot as plt

from Environment2 import Environment
from TS_Learner2 import TS_Learner

from q1_functions import objective_function, conversion1, conversion2
from q1_optimiser import p1_optimal
from q5 import assignment

#Variables fixed of the problem
alpha = [0.25]*16
p2 = 50 # price before promotion
p2_after_promo = [p2 * (1 - P) for P in [0, 0.10, 0.20, 0.30]] 
n_clients_per_class =  [20, 50, 40, 15]


#Variables fixed of the testing
P1 = [20, 30, 42, 50, 60] #j'ai changé les bras, c'est moins ambigu pour voir quel est le meilleur
n_arms = len(P1)
p_1 = np.array([conversion1(p1) for p1 in P1]) # conversion rate of item 1 for each arm and for each class
p_2 = conversion2(p2_after_promo) #attention à l'utilisation les 16taux de conversion sont tous les uns après les autres dans une même liste de 16éléments

T = 365
n_experiments = 50

#Variables fixed of the solution
p1_opt = p1_optimal(p1 = np.mean(P1), p2 = p2, alpha = alpha, n_clients_per_class = n_clients_per_class)


#Paramters to randomize the number of customers per class
n_clients_per_class_rnd = [0] * len(n_clients_per_class)
std = 1

for e in range(0, n_experiments):
    #storing data for plot
    ts_rewards_per_experiment = []
    pulled_arms_global = np.array([0]*len(P1))
    
    print(f"Experiment number {e}\n")
    env = Environment(n_arms = n_arms, probabilities_1 = p_1, probabilities_2 = p_2)
    ts_learner = TS_Learner(n_arms, P1, p2, alpha, n_clients_per_class)    
    ts_memory_pulled_arm = []
    
    for t in range(0, T):
        #The number of customer per class is a random variable
        n_clients_per_class_rnd = list(np.round(n_clients_per_class + np.random.normal(0,std,4)).astype(int)) #simulation of the number of customer per class according to a normal distribution

        
        #TS Learner
        pulled_arm = ts_learner.pull_arm()
        ts_memory_pulled_arm.append(pulled_arm)
        
        cv_rate_1, cv_rate_2 = env.round(pulled_arm, n_clients_per_class_rnd)
        profit = objective_function(P1[pulled_arm], p2, alpha, n_clients_per_class) #On est d'accord que le nombre de clients n'est pas connu du Learner donc il utilise la moyenne ?
        ts_learner.update(pulled_arm, cv_rate_1, cv_rate_2, profit)

        #Matching
        #We use the hungarian algorithm to find the optimal assignment using as p1 the arm pulled by the learner
        alpha = assignment(P1[pulled_arm],p2,n_clients_per_class_rnd)[0]
        alpha = alpha.flatten() #la fonction assignement renvoie un array (4,4) or le Learner a besoin d'une liste

       
        
    n_pulled_arms = [((np.array(ts_memory_pulled_arm) == i).sum()) for i in range(len(P1))]
    pulled_arms_global += np.array(n_pulled_arms)
    
    print(f" for the experiment number {e} the number of time each arm has been pulled are \n {n_pulled_arms} \n")
    most_pulled_arm = np.argmax(n_pulled_arms)
    p1 = P1[most_pulled_arm]
    print(f"so the most pulled arm is {P1[most_pulled_arm]} \n \n -------------------------- \n")
    ts_rewards_per_experiment.append(ts_learner.total_profit)
    
    


#On utilise le p1 optimal (42) comme p1 pour l'algo hongrois qui détermine la meilleure affectation des promos  
assignment_opt = assignment(p1_opt,p2,n_clients_per_class)
opt = objective_function(p1_opt, p2, assignment_opt[0].flatten(), n_clients_per_class)

pulled_arms_global = opt* pulled_arms_global / max(pulled_arms_global) #we use the optimal value just as a scaling factor so that the scale of the points matches the scale of the optimal curve
    


plt.figure(0)
plt.xlabel("t")
plt.ylabel("Regret")
plt.plot(np.cumsum(np.mean(opt - ts_rewards_per_experiment, axis = 0)), 'b')
plt.legend(["TS"])

plt.figure(1)
X = np.linspace(10, 70, 100)
Y = [objective_function(x, p2, assignment_opt[0].flatten(), n_clients_per_class) for x in X]
plt.xlabel('prize of item 1')
plt.ylabel('profit over the day')
plt.scatter(P1, pulled_arms_global)
plt.plot(X, Y, 'g')

plt.show()

