# -*- coding: utf-8 -*-
import numpy as np 
import matplotlib.pyplot as plt

from Environment2 import Environment
from TS_Learner2 import TS_Learner
from UCB_Learner2 import UCB_Learner
from q1_functions import objective_function, conversion1, conversion2, experiment_obj_fun
from q1_optimiser import p1_optimal

#Variables fixed of the problem
alpha = [0.25]*16
P1 = [120,140,160,180,200] #arms
p2 = 150 # price before promotion
p2_after_promo = [p2 * (1 - P) for P in [0, 0.10, 0.20, 0.30]] #price after promotions, be careful that the promos are the same than in obj_fun. 
n_clients_per_class = [50, 20, 10, 5]

n_arms = len(P1)
p_1 = np.array([conversion1(p1) for p1 in P1]) # conversion rate of item 1 for each arm and for each class
p_2 = conversion2(p2_after_promo) #attention à l'utilisation les 16taux de conversion sont tous les uns après les autres dans une même liste de 16éléments
p1_opt = p1_optimal(p1 = np.mean(P1), p2 = p2, alpha = alpha, n_clients_per_class = n_clients_per_class)

T = 365
n_experiments = 10

#storing data for plot
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
        cv_rate_1, cv_rate_2 = env.round(pulled_arm, n_clients_per_class)
        profit = experiment_obj_fun(P1[pulled_arm], cv_rate_1, p2_after_promo, cv_rate_2)
        ts_learner.update(pulled_arm, cv_rate_1, cv_rate_2, profit)
        
        #UCB Learner
        pulled_arm = ucb_learner.pull_arm()
        ucb_memory_pulled_arm.append(pulled_arm)        
        cv_rate_1, cv_rate_2 = env.round(pulled_arm, n_clients_per_class)
        profit = experiment_obj_fun(P1[pulled_arm], cv_rate_1, p2_after_promo, cv_rate_2) 
        ucb_learner.update(pulled_arm, cv_rate_1, cv_rate_2, profit)

    #TS Arms        
    n_pulled_arms = [((np.array(ts_memory_pulled_arm) == i).sum()) for i in range(len(P1))]
    ts_pulled_arms_global += np.array(n_pulled_arms)
    print(f" for the experiment number {e} the number of time each arm has been pulled are \n {n_pulled_arms} \n")
    most_pulled_arm = np.argmax(n_pulled_arms)
    n_pulled_arms[most_pulled_arm] = 0
    second_most_pulled_arm = np.argmax(n_pulled_arms)
    print(f"so the 2 most pulled arms are {P1[most_pulled_arm]} and {P1[second_most_pulled_arm]} \n \n -------------------------- \n")
    
    ts_rewards_per_experiment.append(ts_learner.total_profit)
    
    #UCB Arms
    ucb_n_pulled_arms = [((np.array(ucb_memory_pulled_arm) == i).sum()) for i in range(len(P1))]
    ucb_pulled_arms_global += np.array(ucb_n_pulled_arms)
    print(f" for the experiment number {e} the number of time each arm has been pulled are \n {ucb_n_pulled_arms} \n")
    ucb_most_pulled_arm = np.argmax(ucb_n_pulled_arms)
    ucb_n_pulled_arms[ucb_most_pulled_arm] = 0
    ucb_second_most_pulled_arm = np.argmax(ucb_n_pulled_arms)
    print(f"so the 2 most pulled arms are {P1[ucb_most_pulled_arm]} and {P1[ucb_second_most_pulled_arm]} \n \n -------------------------- \n")

    ucb_rewards_per_experiment.append(ucb_learner.total_profit)

opt = objective_function(p1_opt, p2, alpha, n_clients_per_class)


ts_pulled_arms_global = opt * ts_pulled_arms_global / max(ts_pulled_arms_global) #we use the optimal value just as a scaling factor so that the scale of the points matches the scale of the optimal curve
ucb_pulled_arms_global = opt * ucb_pulled_arms_global / max(ucb_pulled_arms_global) #we use the optimal value just as a scaling factor so that the scale of the points matches the scale of the optimal curve
   

plt.figure(0)
plt.xlabel("t")
plt.ylabel("Regret")
plt.plot(np.cumsum(np.mean(opt - ts_rewards_per_experiment, axis = 0)), 'b')
plt.plot(np.cumsum(np.mean(opt - ucb_rewards_per_experiment, axis = 0)), 'r')
plt.legend(["TS", "UCB"])

plt.figure(1)
X = np.linspace(80, 200, 100)
Y = [objective_function(x, p2, alpha, n_clients_per_class) for x in X]
plt.xlabel('prize of item 1')
plt.ylabel('profit over the day')
plt.scatter(P1, ts_pulled_arms_global, None, 'b')
plt.scatter(P1, ucb_pulled_arms_global, None, 'r')
plt.legend(["TS", "UCB"])
plt.plot(X, Y, 'g')

plt.show()

