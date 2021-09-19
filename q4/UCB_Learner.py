# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 15:48:16 2021

@author: maxime
"""

from Learner import Learner
from q1_optimiser import p1_optimal
from q1_functions import objective_function
import math

import numpy as np 

alpha = [0.25]*16
p2 = 200 # price before promotion
p2_after_promo = [p2 * (1 - P) for P in [0, 0.10, 0.20, 0.30]] #price after promotions, be careful that the promos are the same than in obj_fun. 
n_clients_per_class = [50,40,25,15]

P1 = [20, 30, 35, 40, 45, 50] #arms
p1_opt = p1_optimal(p1 = np.mean(P1), p2 = p2, alpha = alpha, n_clients_per_class = n_clients_per_class)

opt = objective_function(p1_opt, p2, alpha, n_clients_per_class)



class UCB_Learner(Learner):
    
    
    def __init__(self, n_arms, p1, p2, alpha, n_clients_per_class):
        '''
        Il est important d'initialiser les valeurs des moyennes avec une 
        valeure supérieure au profit optimum et non avec un vecteur de 0
        Le problème se situe dans l'update. Puisque notre profit a une grande valeur
        et non 0 ou 1, l'upperbound du bras choisi est largement réhaussée à chaque update
        ce qui ammène à tirer toujours le même bras.
        '''        
        super().__init__(n_arms, p1, p2, alpha, n_clients_per_class)    
        self.empirical_means = np.array([0.1]*n_arms)
        self.confidence = np.array([np.inf]*n_arms)
                
    def pull_arm(self): 
        upper_conf = self.empirical_means + self.confidence
        return np.random.choice(np.where(upper_conf == upper_conf.max())[0])
    
    def update(self, pulled_arm, cv_rate_1, cv_rate_2, profit): 
        #print(f"profit avant {profit}")
        self.update_observations(pulled_arm, cv_rate_1, cv_rate_2, profit) # although normalizing we still want to remember the result 
        min_profit = np.percentile(self.total_profit, 30)
        max_profit = np.percentile(self.total_profit, 70)
        profit = (profit - min_profit )/ (max_profit - min_profit) # min-max norm
        if math.isnan(profit) == True : profit = 0 # We don't want the first experience to count in this case :/ 
        # profit = 2*np.arcsin(profit)/np.pi # This way we change the norm between 0 and 1. 
        #print(profit)
        self.t += 1
        self.empirical_means[pulled_arm] = (self.empirical_means[pulled_arm] * (self.t - 1) + profit) / self.t ## updating mean reward of that arm  
        for a in range(self.n_arms):
            n_samples = len(self.cv_rate_1_per_arm[a])
            self.confidence[a] = (2*np.log(self.t)/n_samples)**0.5 if n_samples > 0 else np.inf
        