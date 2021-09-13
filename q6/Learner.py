# -*- coding: utf-8 -*-
"""
Created on Sun May 16 18:08:13 2021

@author: Fabien
"""

import numpy as np 

class Learner():
    def __init__(self, n_arms, P1, p2, alpha, n_clients_per_class):
        self.n_arms = n_arms
        self.P1 = P1
        self.p2 = p2
        self.alpha = alpha
        self.n_clients = np.sum(n_clients_per_class)
        self.t = 0
        self.cv_rate_1_per_arm = [[] for i in range(n_arms)] #collected reward for each arm per round where they've been pulled
        self.collected_cr_1 = np.array([]) #conversion rate obtained at each round = do we sell item 1 at each round
        self.collected_cr_2 = np.array([]) #conversion rate per promo obtained at each round
        self.total_profit = np.array([])
        
    def update_observations(self, pulled_arm, cv_rate_1, cv_rate_2, profit): #update the list above
        self.cv_rate_1_per_arm[pulled_arm].append(cv_rate_1) #what is cv_rate_1 here? a list of the succes and fails? the mean of the succes aka a real converstion rate?
        self.collected_cr__1 = np.append(self.collected_cr_1, cv_rate_1)
        self.collected_cr_2 = np.append(self.collected_cr_2, cv_rate_2)
        self.total_profit =  np.append(self.total_profit, profit)