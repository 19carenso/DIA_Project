# -*- coding: utf-8 -*-
"""
Created on Sun May 16 18:08:13 2021

@author: Fabien
"""

import numpy as np 

class Learner():
    def __init__(self, n_arms, p1, p2, alpha, n_clients_per_class):
        self.n_arms = n_arms
        self.p1 = p1
        self.p2 = p2
        self.alpha = alpha
        self.n_clients_per_class = n_clients_per_class
        self.t = 0
        self.cv_rate_1_per_arm = [[] for i in range(n_arms)] #collected reward for each round and for each arm
        #TODO: redefine this variable, should contain the reward, not the conversion rate (it is used in UCB) 
        self.collected_cr_1 = np.array([]) #conversion rate obtained at each round = do we sell item 1 at each round
        self.total_profit = np.array([])
        
    def update_observations(self, pulled_arm, cv_rate_1, profit): #update the list above
        self.cv_rate_1_per_arm[pulled_arm].append(cv_rate_1)
        self.collected_cr_1 = np.append(self.collected_cr_1, cv_rate_1)
        self.total_profit =  np.append(self.total_profit, profit)