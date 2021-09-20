# -*- coding: utf-8 -*-
import numpy as np 
import matplotlib.pyplot as plt

from Non_Stationnary_Environment import Non_Stationnary_Environment
from TS_Learner2 import TS_Learner
from SWTS_Learner import SWTS_Learner
from q1_functions import objective_function,objective_function_mod, conversion1, conversion2, conversion1_mod
from q1_optimiser import p1_optimal, p1_optimal_mod
from q5 import assignment

#Variables fixed of the problem
alpha = [0.25]*16
P1 = [120,140,160,180,200]
p2 = 150
p2_after_promo = [p2 * (1 - P) for P in [0, 0.10, 0.20, 0.30]] 
n_clients_per_class =  [50, 20, 10, 5]

n_arms = len(P1)
p_1 = np.array([conversion1(p1) for p1 in P1])
p_1_mod = np.array([conversion1_mod(p1) for p1 in P1])
p_2 = conversion2(p2_after_promo) 

T = 364
n_experiments = 10

#Paramters to randomize the number of customers per class
n_clients_per_class_rnd = [0] * len(n_clients_per_class)
std = 1

window_size = 30
n_phases = 4
phases_len = int(T/n_phases)

#storing data for plot
ts_rewards_per_experiment = []
swts_rewards_per_experiment = []

for e in range(0, n_experiments):
   print(f"Experiment number {e}\n")
   ts_env = Non_Stationnary_Environment(n_arms = n_arms, probabilities_1 = [p_1,p_1_mod,p_1,p_1_mod], probabilities_2 = p_2,horizon=T)
   swts_env = Non_Stationnary_Environment(n_arms = n_arms, probabilities_1 = [p_1,p_1_mod,p_1,p_1_mod], probabilities_2 = p_2,horizon=T)

   ts_learner = TS_Learner(n_arms, P1, p2, alpha, n_clients_per_class)    
   ts_memory_pulled_arm = []
    
   swts_learner = SWTS_Learner(n_arms, P1, p2, alpha, n_clients_per_class,window_size)    
   swts_memory_pulled_arm = []
    
   for t in range(0, T):
      phase_size = T/n_phases
      current_phase = int(t/phase_size)
      
      #The number of customer per class is a random variable
      n_clients_per_class_rnd = list(np.round(n_clients_per_class + np.random.normal(0,std,4)).astype(int)) #simulation of the number of customer per class according to a normal distribution
      #The gaussian is truncated. If the number of customers in one class is negative then we set it to 0.
      n_clients_per_class_rnd = [0 if n_clients<0 else n_clients for n_clients in n_clients_per_class_rnd]

      #TS Learner
      pulled_arm_ts = ts_learner.pull_arm()
      ts_memory_pulled_arm.append(pulled_arm_ts)        
      
      #SWTS Learner
      pulled_arm_swts = swts_learner.pull_arm()
      swts_memory_pulled_arm.append(pulled_arm_swts)

      
      cv_rate_1, cv_rate_2 = ts_env.round(pulled_arm_ts, n_clients_per_class_rnd)
      if((current_phase==0) or (current_phase==2)):
         profit_ts = objective_function(P1[pulled_arm_ts], p2, alpha, n_clients_per_class) #On est d'accord que le nombre de clients n'est pas connu du Learner donc il utilise la moyenne ?
      else : 
         profit_ts = objective_function_mod(P1[pulled_arm_ts], p2, alpha, n_clients_per_class) #On est d'accord que le nombre de clients n'est pas connu du Learner donc il utilise la moyenne ?
      ts_learner.update(pulled_arm_ts, cv_rate_1, cv_rate_2, profit_ts)
      
      cv_rate_1, cv_rate_2 = swts_env.round(pulled_arm_swts, n_clients_per_class_rnd)
      if((current_phase==0) or (current_phase==2)):
         profit_swts = objective_function(P1[pulled_arm_swts], p2, alpha, n_clients_per_class) 
      else : 
         profit_swts = objective_function_mod(P1[pulled_arm_swts], p2, alpha, n_clients_per_class)    
      swts_learner.update(pulled_arm_swts, cv_rate_1, cv_rate_2, profit_swts)

      #Matching
      #We use the hungarian algorithm to find the optimal assignment using as p1 the arm pulled by the learner
      alpha = assignment(P1[pulled_arm_swts],p2,n_clients_per_class_rnd)[0]
      alpha = alpha.flatten() 

       
        
   swts_rewards_per_experiment.append(swts_learner.total_profit)
   ts_rewards_per_experiment.append(ts_learner.total_profit)

swts_instantaneous_regret = np.zeros(T)
ts_instantaneous_regret = np.zeros(T)
optimum_per_round = np.zeros(T)

#Variables fixed of the solution
p1_opt = p1_optimal(p1 = np.mean(P1), p2 = p2, alpha = alpha, n_clients_per_class = n_clients_per_class)
p1_opt_mod = p1_optimal_mod(p1 = np.mean(P1), p2 = p2, alpha = alpha, n_clients_per_class = n_clients_per_class)

#Optimal price and optimal reward per phase
opt_price_per_phases = [p1_opt,p1_opt_mod,p1_opt,p1_opt_mod]
opt_per_phases = [objective_function(p1_opt, p2, alpha, n_clients_per_class), objective_function_mod(p1_opt_mod, p2, alpha, n_clients_per_class)]*2


#Compute the instantaneous_regret for both learner
for i in range(n_phases):
   t_index = range(i*phases_len,(i+1)*phases_len)
   optimum_per_round[t_index] = opt_per_phases[i]
   swts_instantaneous_regret[t_index] = opt_per_phases[i] - np.mean(swts_rewards_per_experiment,axis = 0)[t_index]
   ts_instantaneous_regret[t_index] = opt_per_phases[i] - np.mean(ts_rewards_per_experiment,axis = 0)[t_index]

plt.figure(0)
plt.plot(np.cumsum(swts_instantaneous_regret),'r')
plt.plot(np.cumsum(ts_instantaneous_regret),'b')
plt.legend(['SWTS','TS'])
plt.show()
